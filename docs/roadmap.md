# Execution plan

## One rule

Every step must add a verified claim. Skip any run that only repeats established
arithmetic on a slower or irrelevant path. An optimization is retained only when it
passes the unchanged oracle and improves measured wall time, peak memory, ciphertext
count, rotation count, or multiplicative depth.

Two architectures are tracked (`docs/feasibility/05_architecture_options.md`):

- **Scheme A (frozen baseline):** every encrypted correctness run uses
  `HEStd_128_classic`, one uninterrupted ciphertext lineage, no intermediate decryption
  feeding evaluation, finite output, and global plus worst-token `rel_inf <= 4e-2`. No
  further Scheme A runs are planned; its existing evidence stands as the paper's
  non-interactive ablation.
- **Scheme B (active):** every encrypted correctness run uses `HEStd_128_classic`,
  keeps linear algebra in one uninterrupted encrypted GPU lineage, and permits decrypt
  only at pre-declared nonlinearity boundaries performed solely by the data-owning
  client (secret-key holder) on its own data. Boundaries are fixed before the run, never
  chosen adaptively from decrypted content. Same `4e-2` oracle gate as Scheme A, plus
  recorded round-trip count and a wall-time split between GPU-encrypted and client-side
  plaintext compute.

Every retained result gets a new immutable run JSON and manifest entry, tagged with its
scheme.

## Verified foundation

- `[V]` The three plaintext tasks pass and provide frozen task oracles.
- `[V]` Every required CKKS primitive and bootstrap refresh passes independently.
- `[V]` One complete `D=8`, `T=4`, two-head DNAGPT-shaped block passes with global
  rel-inf `1.49e-3`, zero intermediate decrypt attempts, and one final decrypt.
- `[V]` The complete C++/CUDA block passes on one A100 with global rel-inf
  `1.8423e-4`, worst-token rel-inf `4.8125e-4`, and `93.405 s [gpu]` evaluation.
- `[V]` Released block-0 weights pass the encrypted `D=768`, `T=2` LayerNorm gate at
  global rel-inf `3.8193e-10`.
- `[V]` The released block-0 `D=768`, `T=2`, 12-head encrypted attention/projection
  gate passes at global and worst-token rel-inf `7.0183e-10` in `1,347.062 s [gpu]`.
- `[V]` The independent plaintext/export contract matches all 12 released blocks and
  the classifier head; `[V]` block-0 domains fail beginning at block 1.
- `[V/A]` A fixed public T=2 range-control schedule passes all 12 block outputs and
  the classifier with worst-token rel-inf `8.1174e-3` and zero domain violations.
- `[V]` The required FIDESlib asymmetric-Chebyshev correction is measured before and
  after patching (`0.292889` fail to `1.5282e-5` pass).
- `[V]` Local optimization probes preserve correctness:
  - BSGS reduces `D=16` rotations from 15 to 6.
  - hoisted baby rotations preserve the same encrypted result.
  - numerator-first attention saves two levels.
  - GELU degree 5 uses four levels and has `4.26e-3` sample-grid rel-inf.

The complete block is the CPU arithmetic anchor. The optimization probes are screening
evidence; each selected change still has to pass the complete-block and real-weight
oracles.

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

## Current: real-width block

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

## Scale and optimize

Increase sequence length only after the real-width block passes. `T=2` is not an
application milestone. Validate the packed layout at `T=8/16`, then use capacity gates
`T=32, 64, 103`; `T=103` is the current GSR task-representative target. Skip other
sizes unless a correctness, memory, or throughput boundary needs resolution. Test one
change at a time:

1. token/head packing and batched nonlinear evaluation;
2. parallel BSGS plus hoisted/double-hoisted rotations;
3. pre-encoded weight reuse and public-mask fusion;
4. numerator-first attention;
5. the lowest polynomial degrees that pass the full oracle;
6. kernel fusion, streams, and full GPU residency;
7. multi-GPU sharding only when one A100 cannot hold the live set.

Two timing repeats are required for a directional speedup; three or more are required
for a reported stable speedup. Always report wall time, operation split, bootstrap
count, host RAM, per-GPU VRAM, and ciphertext layout.

## Compose only after one block scales

Test two blocks first to validate refresh placement and accumulated CKKS error. If it
passes, advance directly to the twelve-block 0.1b backbone unless a four-block run is
needed to locate an error or memory boundary. Then add final LayerNorm and the GSR
readout head. The end gate is the encrypted embedded-vector-to-task-output graph at
`T=103`, not a T=2 block-only demonstration.

Input scope remains explicit:

- `[V]` The transformer path accepts encrypted embedded numeric vectors.
- `[U]` Encrypted token-index embedding lookup is separate.
- `[U]` Production key custody, transport, and client-only final decryption need a
  deployment harness after arithmetic closure.

## Scheme B: hybrid client-assisted CKKS

Adopted 2026-07-25 after three independent chained-composition failures
(`multiplicative_depth` in `{50, 58, 64}`) all traced to GPU memory exhaustion during
rotation-key/bootstrap-plaintext loading, not accuracy — i.e. the "measured GPU
correctness failure" condition below is met. Full comparison and rationale in
`docs/feasibility/05_architecture_options.md`. Plan:

1. Reuse the existing real-weight block-0 CKKS/FIDESlib graph unchanged for every linear
   op (Q/K/V projection, attention matmul, output projection, FFN, residual adds).
2. Remove `EvalChebyshevFunction` calls for LayerNorm invsqrt, the T=2 sigmoid attention
   identity, and GELU. Replace each with an explicit decrypt (client, secret-key holder,
   own data only) -> exact plaintext function -> re-encrypt boundary.
3. Re-measure against the unchanged Phase-A oracle and `4e-2` gate; record round-trip
   count and the GPU-encrypted vs. client-plaintext wall-time split.
4. Because Scheme B removes the bootstrap-driven depth requirement, re-derive the
   minimum viable `multiplicative_depth`/`batch_slots` before assuming depth 43 is still
   needed; a smaller context may clear the exact memory wall that blocked Scheme A.
5. Scale sequence length and block count only after one Scheme B block gate passes,
   mirroring the "Scale and optimize" and "Compose only after one block scales" order
   above.

## Explicit skips

- no full twelve-layer Python/OpenFHE CPU run;
- no duplicate “CPU container” full pass after one complete block has closed;
- no custom CUDA cryptography while current FIDESlib supplies the operation;
- `[done, 2026-07-25]` no CPU/GPU hybrid milestone unless a measured GPU correctness
  failure forces it — three independent chained-composition failures traced to GPU
  memory exhaustion (not accuracy) forced the Scheme B pivot; see
  `docs/feasibility/05_architecture_options.md`. This unblocks explicit client-side
  decrypt boundaries under Scheme B's own contract above, not a silent change to
  Scheme A;
- no multi-GPU work until single-GPU memory or throughput is measured;
- no task-scale twelve-block run before a real-width block and a two-block refresh gate pass.
