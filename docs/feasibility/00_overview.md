# FHE feasibility overview

## Question

Can DNAGPT evaluate encrypted embedded genomic-token vectors and return encrypted
outputs so an untrusted compute provider never sees plaintext DNA?

Two architectures are tracked as of 2026-07-25
(see [05_architecture_options.md](05_architecture_options.md) for the full comparison):

- **Scheme A (frozen baseline):** pure non-interactive CKKS, one uninterrupted
  ciphertext lineage, zero intermediate decrypt.
- **Scheme B (active path):** hybrid client-assisted CKKS — linear algebra stays
  encrypted end to end on GPU; only the data-owning client (already the secret-key
  holder) decrypts, at pre-declared nonlinearity boundaries, its own data only.

The current answer is deliberately split:

- `[V]` The required CKKS primitives and a depth-refresh operation work independently
  at 128-bit security.
- `[V]` The complete `D=8`, `T=4`, two-head toy block passes the arithmetic-closure
  gate with no intermediate decrypt.
- `[V]` The same complete toy graph passes in C++/CUDA on one A100 with one final
  decrypt and `93.405 s [gpu]` encrypted evaluation.
- `[V]` Released DNAGPT block-0 weights pass encrypted `D=768`, `T=2` LayerNorm
  and complete 12-head attention/projection gates.
- `[V/A]` A fixed, publicly calibrated approximation schedule passes the plaintext
  oracle through all 12 released blocks and the GSR head; broader calibration and
  encrypted composition remain open.
- `[V]` Local screens validate rotation/depth-reducing schedules without changing the
  encrypted result.
- `[U]` The rest of the real-width block, multi-block composition, and task-level
  encrypted inference have not yet closed under Scheme A; Scheme A is now frozen as the
  paper's non-interactive baseline/ablation after three chained-composition attempts
  failed closed on a root-caused GPU memory wall (not accuracy). Work continues under
  Scheme B.
- `[U]` Encrypted token-index embedding lookup is outside the current backbone input
  boundary.

The toy proof starts after embedding: the client encrypts numeric embedding vectors.
The server-side arithmetic is LayerNorm, Q/K/V projections, multi-head causal softmax
attention, output projection, second LayerNorm, GELU MLP, and residual additions.

## Acceptance contract

Every encrypted correctness gate uses:

- OpenFHE CKKS with `HEStd_128_classic`
- one context and key lineage
- no intermediate `Decrypt` call
- one final decrypt for oracle measurement
- finite output
- global and worst-token relative-infinity error no greater than `4e-2`

The result harness homomorphically packs all toy token outputs into one ciphertext before
that final decrypt. `fhe/oracle.py` supplies the plaintext reference and gate.

## Evidence progression

| Result | New claim | State | Evidence |
|---|---|---|---|
| Primitive closure | Every required operator and bootstrap refresh works in isolation | `[V]` complete | `fhe_operator_matrix_d8_20260724.json`, `fhe_bootstrap_d8_20260724.json` |
| Complete toy block | A DNAGPT-shaped block closes in one encrypted lineage | `[V]` complete | `fhe_toy_block.json` |
| Optimization screens | BSGS/hoisting, numerator-first attention, and GELU degree candidates preserve correctness | `[V]` complete | `fhe_local_optimizations_20260724.json` |
| GPU parity | The complete toy arithmetic survives C++/CUDA execution | `[V]` complete | `fhe_fides_toy_a100_asymfix2_20260724.json` |
| Real-width block | Released 0.1b block-0 gates work at `D=768`, 12 heads, real weights | `[V]` LayerNorm + attention complete (original schedule) | `fhe_fides_real_d768_t2_attention_a100_asymfix2_20260724.json` |
| Original real full-block schedule | Determine whether exp-plus-reciprocal attention fits depth 43 | `[V]` fails closed at token-1 MLP; no decrypt | `fhe_fides_real_d768_t2_block0_depth43_FAIL_20260724.json` |
| Sigmoid-schedule attention gate | Replace exp-plus-reciprocal with the T=2 sigmoid identity to recover depth for the MLP | `[V]` LN1+attention pass at `packed_output_level=22/43`, 709.3s vs 1347.1s | `fhe_fides_real_d768_t2_attention_sigmoid13_a100_asymfix2_20260724.json` |
| Complete real-width block | Close block 0 (LN1+attention+MLP+LN2+pack) at depth 43 with the sigmoid schedule | `[V]` complete: packed at level 41/43, rel_inf `6.41e-6`, `2461.8 s [gpu]` | `fhe_fides_real_d768_t2_block0_sigmoid13_a100_asymfix2_20260724.json` |
| Twelve-block nonlinear schedule | Fixed public domains, stable T=2 attention, scaled LayerNorm, and full-domain GELU preserve all blocks/head | `[V/A]` plaintext preflight passes; not FHE | `fhe_range_control_t2_12block_optimized_v2_20260724.json` |
| Scale and composition | Refreshes, two encrypted blocks, all 12 blocks, and task head close | `[V]` native-GPU refresh gate and the complete block-0 sigmoid gate both pass in isolation; `[V]` the chained two-block (`gpu_multiblock`, depth 64) gate fails closed -- root cause confirmed via a symbolized backtrace (`AddBootstrapPlaintexts -> GPUmalloc`) as GPU memory footprint (batch_slots=4096, 63 rotation keys) exceeding one A100's 80GB, not depth- or digit-count-specific; `[U]` 12-block closure remains | `fhe_fides_refresh_d768_t2_native_a100_asymfix2_20260724.json`, `fhe_fides_real_d768_t2_block0_sigmoid13_a100_asymfix2_20260724.json`, `fhe_fides_gpu_multiblock_blocks0_1_refresh_FAIL_20260724.json`, `fhe_fides_gpu_multiblock_bisect_depth50_FAIL_20260724.json`, `fhe_fides_gpu_multiblock_bisect_depth58_backtrace_FAIL_20260724.json` |

A twelve-layer CPU run is intentionally skipped: it would repeat arithmetic already
established by the complete block while measuring a rejected performance path.

## Performance path

Python remains the oracle and evidence harness. The performance implementation is
C++/CUDA using OpenFHE/FIDESlib interoperability. The pinned FIDESlib 2.1.3 commit needs
the repository's two-branch asymmetric-Chebyshev correction; the unpatched GPU path
failed at `rel_inf=0.292889`, while the corrected path passes at `1.5282e-5`.
The preferred path keeps the complete server pass GPU resident. CPU fallback is used
only if a current GPU operator fails the same correctness gate.

GPU parity is complete. The real-width implementation now uses 4096 power-of-two slots,
32×32 BSGS, hoisted baby rotations, and numerator-first `T=2` attention. Each further
change keeps the same oracle and receives a new immutable result. Longer-sequence
packing, refresh placement, encoded-weight reuse, kernel fusion, and multi-GPU sharding
remain later gates.

The original full-block schedule has a measured depth failure rather than an accuracy
failure: its token-1 branch needs packed level 46 with a depth-43 chain. The next
version preserves T=2 softmax exactly as a single sigmoid of the score difference,
which removes the reciprocal polynomial and is expected to fit within depth 43.

See [roadmap.md](../roadmap.md) for the gate definitions,
[01_backend_selection.md](01_backend_selection.md) for the backend decision,
[02_operator_matrix.md](02_operator_matrix.md) for primitive measurements, and
[05_architecture_options.md](05_architecture_options.md) for the Scheme A/B/C
comparison and the active Scheme B decision. The passing block and derived 0.1b
boundary are in [03_measurements.md](03_measurements.md).
