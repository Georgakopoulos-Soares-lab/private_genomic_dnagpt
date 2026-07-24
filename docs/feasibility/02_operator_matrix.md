# CKKS operator matrix

## Contract

The isolated operators use deterministic `D=8` inputs, OpenFHE CKKS at
`HEStd_128_classic`, and a relative-infinity error gate of `4e-2`. They decrypt only
after each isolated operator completes. These runs establish arithmetic support and
depth cost; they do not predict full-model latency.

## Measurements

| Operation | CKKS implementation | Levels | max abs error | rel-inf | Wall time | Verdict |
|---|---|---:|---:|---:|---:|---|
| `D×D` linear | Halevi-Shoup diagonal BSGS | 1 | `1.94e-13` | `4.73e-13` | `2.486 s [emu]` | `[V]` pass |
| LayerNorm | mean subtraction + Chebyshev inverse square root, degree 27 | 10 | `6.83e-6` | `4.54e-6` | `6.784 s [emu]` | `[V]` pass |
| causal-softmax primitive | Chebyshev exponential degree 13 + reciprocal degree 27 | 12 | `1.50e-6` | `6.59e-6` | `7.837 s [emu]` | `[V]` pass |
| GELU broad profile | tanh GELU Chebyshev degree 13 on `[-4,4]` | 5 | `9.73e-4` | `6.04e-4` | `3.764 s [emu]` | `[V]` pass |
| GELU toy-block profile | tanh GELU Chebyshev degree 7 on `[-2,2]` | 5 | `1.10e-3` | `6.84e-4` | `1.632 s [native-cpu]` | `[V]` pass |
| bootstrap refresh | OpenFHE approximate CKKS bootstrap, budget `[3,3]` | restores 10 | `1.67e-5` | `2.56e-5` | `53.814 s [emu]` | `[V]` pass |

Sources:

- `results/runs/fhe_operator_matrix_d8_20260724.json`
- `results/runs/fhe_gelu_toy_profile_brev_cpu_20260724.json`
- `results/runs/fhe_bootstrap_d8_20260724.json`

The first four timings and bootstrap timing are x86-64 Docker under Apple Silicon
emulation. The narrowed GELU timing is native x86-64 CPU on Brev. Timing labels must not
be compared as a speedup because the environments and allocated CPU counts differ.

## Depth result

The standalone bootstrap started at level 29, returned at level 19, then supported nine
additional multiplicative levels before the one final decrypt. `[V]` This proves usable
refresh, not a full-block bootstrap schedule.

The complete toy block uses a longer leveled chain because it is cheaper and simpler for
this one-block correctness gate. Cross-block composition must introduce refreshes and
measure their number, placement, error, and latency rather than infer them from this
standalone demonstration.

## Domain boundary

All nonlinear domains are public and source-frozen:

- LayerNorm variance plus epsilon: `[0.03, 0.9]`
- attention logits: `[-2, 2]`
- causal softmax denominator: `[0.5, 4.5]` in the toy block
- toy MLP pre-GELU value: `[-2, 2]`

The deterministic toy oracle is inside those bands. `[U]` Real DNAGPT inputs need public
calibration and fail-closed diagnostic range validation before a real-width result can
be load-bearing.

## Local optimization screens

`fhe/optimizations/local_ckks_probe.py` compares encrypted schedules at the same
security level. Each row decrypts only its completed circuit for a NumPy comparison.
Mac timings are `[emu-directional]`; operation/depth changes and correctness are the
load-bearing results.

| Probe | Baseline | Candidate | Result |
|---|---:|---:|---|
| `D=16` diagonal matmul | 15 rotations | BSGS: 6 rotations | `[V]` same level, rel-inf `7.07e-13` |
| BSGS key-switch work | 6 decompositions | hoisted baby steps: 4 | `[V]` rel-inf `9.18e-13`; GPU timing still needed |
| attention aggregation | materialized softmax: 9 levels | numerator-first: 7 levels | `[V]` two levels saved; parity rel-inf `2.59e-11` |
| GELU on `[-2,2]` | degree 7: 5 levels | degree 5: 4 levels | `[V]` sample-grid rel-inf `4.26e-3` |

Degree 3 also passes the isolated grid at `3.30e-2`, but it is too close to the `4e-2`
gate to select before a complete-block and real-weight accuracy run. Degree 5 is the
preferred next candidate. BSGS is already retained. Hoisting is retained only if CUDA
wall time improves; it was not faster under Mac x86-64 emulation.

Evidence: `results/runs/fhe_local_optimizations_20260724.json`.

The optimization order follows primary-source guidance: OpenFHE exposes fast-rotation
precomputation, [parallel BSGS](https://eprint.iacr.org/2024/883) reduces rotations when
packing permits, and [FIDESlib](https://arxiv.org/abs/2507.04775) emphasizes hoisting,
fusion, on-chip reuse, and memory bandwidth on GPUs.
