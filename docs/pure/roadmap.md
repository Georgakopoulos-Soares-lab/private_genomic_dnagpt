# Roadmap — Scheme A (pure non-interactive CKKS), frozen

Frozen 2026-07-25 as the paper's non-interactive ablation baseline after three chained-
composition attempts failed closed on a root-caused GPU memory wall (not accuracy). No
further Scheme A runs are planned; this is a historical record, not an active plan. See
[../roadmap.md](../roadmap.md) for the shared acceptance contract and rules that still
apply, and [../hybrid/roadmap.md](../hybrid/roadmap.md) for the active Scheme B plan
that replaced this one.

## Completed: GPU backend parity

The existing block schedule was implemented in C++/CUDA with current
[FIDESlib](https://github.com/CAPS-UMU/FIDESlib) and its patched OpenFHE interop. FIDESlib
2.1.3 exposes CKKS multiplication, hoisted rotation, bootstrap, CUDA acceleration, and
multi-GPU support; the completed parity implementation kept server-side evaluation GPU
resident.

The small block verified:

- ciphertext layout and rotations match the Python anchor;
- scale/level bookkeeping survives OpenFHE↔FIDESlib conversion;
- no secret key or intermediate decrypt enters server evaluation; and
- wall time, level trace, fixed polynomial domains, image digest, and operation
  schedule are recorded.

Use CPU fallback only for an operator that fails the gate in the current GPU library.
A mixed CPU/GPU path is not a milestone by itself because transfers can dominate.
Do not implement cryptographic CUDA kernels from scratch.

## Real-width block: closed under Scheme A

Close one released-weight DNAGPT block at `D=768`, 12 heads, and `T=2` before sequence
scaling. `T=2` is the smallest non-degenerate causal-attention graph and exercises every
real block-0 matrix while containing memory. It uses public, query-independent
polynomial domains and never feeds a decrypted diagnostic into evaluation.

The complete released block 0 (LayerNorm-1, QKV/12-head attention/projection, residual,
LayerNorm-2, the full 3072-wide GELU MLP, and output packing) now closes in one
encrypted lineage inside the `4e-2` gate: `packed_output_level=41` of the 43-level
chain, `rel_inf=6.41e-6`, one final decrypt
(`fhe_fides_real_d768_t2_block0_sigmoid13_a100_asymfix2_20260724.json`). Only 2 levels
remain unconsumed at that point, so a same-lineage native refresh must be triggered
before block 1, not after further block-0 work.

The native GPU block-boundary refresh has separately passed as a staged gate on the
same fixed released block-0 activation fixture: it restores 21 levels and carries a
post-refresh nonlinear tail (square + LayerNorm epsilon) inside the `4e-2` gate with one
final decrypt (`fhe_fides_refresh_d768_t2_native_a100_asymfix2_20260724.json`). Both
gates are validated in isolation on the same activation fixture, not yet chained
end-to-end from one continuous ciphertext.

The `gpu_multiblock` CUDA module (block 0 -> public conditioning -> native encrypted
bootstrap -> block 1, one lineage, one final decrypt, `multiplicative_depth=64`,
`large_digits=4` HYBRID key switching) is compiled, its fixture staged, and its own
fail-closed launcher/scheduler now exist (`fhe/gpu_multiblock/launch_brev_multiblock.sh`,
`schedule_multiblock_when_free.sh`). The first run crashes (`SIGSEGV`, exit 139) before
its own first log line (`fhe_fides_gpu_multiblock_blocks0_1_refresh_FAIL_20260724.json`).

Root-caused through a sequence of diagnostics, all on uncommitted local edits reverted
after each test:

1. Per-call setup logging (`std::unitbuf` plus a print after each of
   `GenCryptoContext`/`Enable`/`EvalBootstrapSetup`/`KeyGen`/`EvalMultKeyGen`/
   `EvalRotateKeyGen`/`EvalBootstrapKeyGen`/`LoadContext`/`Synchronize`) localized the
   crash to inside `LoadContext`.
2. `compute-sanitizer` memcheck reported zero device-memory errors, ruling out an
   illegal GPU access; `strace`/`gdb`-based tracing is not viable on this binary
   (CUDA's driver-internal signal handling produces a runaway `SIGSEGV` storm under
   `ptrace`).
3. Lowering `multiplicative_depth` to 50 (same `batch_slots=4096`, 63 rotation keys,
   `large_digits=4`) got much further into `LoadContext` and failed with an explicit
   CUDA allocation error at `81131/81920 MiB`
   (`fhe_fides_gpu_multiblock_bisect_depth50_FAIL_20260724.json`).
4. Lowering `large_digits` to 2 instead was an independent dead end: it crashed at the
   same early point as the original `depth=64`/`large_digits=4` failure, at *both*
   depth 64 and depth 50 -- `large_digits=2` is itself broken/incompatible with this
   rotation-key configuration on this FIDESlib build, unrelated to memory.
5. At `multiplicative_depth=58` (`large_digits=4`), `addr2line` on the crash backtrace
   pinpointed the exact call chain: `AddBootstrapPlaintexts -> Plaintext::load ->
   RNSPoly::loadConstant -> LimbPartition::generateLimbConstant -> Limb -> VectorGPU ->
   GPUmalloc` (`fhe_fides_gpu_multiblock_bisect_depth58_backtrace_FAIL_20260724.json`)
   -- the same bootstrap-precomputation-plaintext-loading code path that step 3 reached
   and failed in cleanly.

Unified conclusion: every failure across `multiplicative_depth` in `{50, 58, 64}` and
`large_digits` in `{2, 4}` traces back to GPU memory exhaustion during rotation-key or
bootstrap-precomputation-plaintext loading inside `LoadContext`. Which specific
allocation fails first -- and whether FIDESlib reports it as a clean CUDA
out-of-memory exception or crashes on an unchecked `GPUmalloc` -- depends on the exact
configuration, but the underlying cause is the same: this context's total GPU memory
footprint does not fit in one A100's 80GB. Tuning `multiplicative_depth` or
`large_digits` alone cannot fix it. `[U]` A real fix needs fewer `batch_slots` and/or
fewer rotation keys (the two levers that directly shrink the number/size of
GPU-resident keys and plaintexts), or splitting the computation across multiple GPUs --
none attempted yet. After the two-block gate passes, increase sequence length.

The first full attempt is measured and retained as a failure: exp-plus-reciprocal
attention creates a 13-level causal-branch gap, so token 1 needs packed level 46
against a depth-43 chain. The `T=2` sigmoid score-difference identity replacing that
path resolves it: the full block closes at level 41 instead of failing at level 46,
first validated as a staged LN1+attention sub-gate at `packed_output_level=22`
(`fhe_fides_real_d768_t2_attention_sigmoid13_a100_asymfix2_20260724.json`) before the
full-block retry above. Raising depth to 49 without changing the graph remains the
fallback/control, not the preferred performance path, and is not needed now that the
sigmoid schedule closes block 0 at depth 43.

The twelve-block plaintext preflight rejected blind block-0 interval reuse and selected
public per-block domains, T=2 sigmoid attention, public-scaled LayerNorm, and full-domain
GELU. The intervals are `[A]` until calibrated on a broader public set; they are never
adapted from a decrypted private query.

This is the point at which the project pivoted to Scheme B — see
[../hybrid/roadmap.md](../hybrid/roadmap.md).
