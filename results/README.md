# Evidence contract

`results/` is the immutable evidence store. It is the only place performance numbers are considered
real, and it is what the paper cites.

## Rules

- **One run = one file.** Each evaluation writes `runs/<tag>.json` (metrics + config + provenance)
  and, where per-example outputs matter, `runs/<tag>_preds.csv`. These are append-only — never edit
  or overwrite a completed run; a re-run gets a new `<tag>`.
- **Every run has a manifest row** in `manifest.yaml`: task, weight, dataset+source, device, n,
  metrics, reference, and a `[V]`/`[U]` verdict.
- **A number without a run file does not exist.** Do not quote metrics from chat or logs.
- Per-example prediction CSVs are the frozen **oracle** for later encrypted-inference acceptance.
- FHE correctness runs also record the security level, context shape, executing source
  hashes where available, approximation domains, level trace, decrypt counts, and an
  explicit environment/latency label.

## Current runs

| tag | task | headline |
|---|---|---|
| `gsr_aataaa_human_full` | GSR | acc 0.9124, F1 0.916 (n=22,604) |
| `mrna_xpresso_human_1ktest` | mRNA regression | r² 0.562, Pearson 0.753 (n=1,000) |
| `mrna_smoke_regression_sh` | mRNA pipeline check | 2 author examples within ~0.04 |
| `gue_prom_core_all` | GUE core promoter | MCC 0.680, acc 0.840 |
| `gue_prom_300_all` | GUE promoter 300bp | MCC 0.897, acc 0.948 |
| `gue_splice_reconstructed` | GUE splice (3-class) | MCC 0.831, acc 0.895 |
| `fhe_operator_matrix_d8_20260724` | CKKS operator matrix | linear/LN/softmax/GELU all pass `4e-2` |
| `fhe_bootstrap_d8_20260724` | CKKS bootstrap | restores 10 levels, rel-inf `2.56e-5` |
| `fhe_gelu_toy_profile_brev_cpu_20260724` | narrowed GELU | rel-inf `6.84e-4 [native-cpu]` |
| `fhe_toy_block` | complete encrypted toy block | global `1.49e-3`, worst-token `2.21e-3`, 0 mid-decrypts |
| `fhe_0p1b_extrapolation_20260724` | derived 0.1b boundary | `[A/U]` counts and lower bounds, not a run |
| `fhe_local_optimizations_20260724` | local CKKS optimization screens | BSGS, numerator-first attention, GELU sweep pass |
| `fhe_runtime_projection_20260724` | derived runtime boundary | `[A/U]` planning range, not a benchmark |
| `fhe_gpu_asymmetric_chebyshev_prepatch_20260724` | FIDESlib GPU backend gate | `[V]` fail, rel-inf `0.292889` |
| `fhe_gpu_asymmetric_chebyshev_asymfix2_20260724` | patched FIDESlib GPU backend gate | `[V]` pass, rel-inf `1.5282e-5` |
| `fhe_fides_toy_a100_asymfix2_20260724` | complete CUDA toy block | global `1.8423e-4`, worst-token `4.8125e-4`, `93.405 s [gpu]` |
| `fhe_fides_real_d768_t2_ln1_a100_asymfix2_20260724` | real block-0 D768 LayerNorm | global `3.8193e-10`, `1.779 s [gpu]` |
| `fhe_multiblock_plaintext_contract_20260724` | 12 released blocks + GSR-head oracle | `[V]` NumPy/torch contract passes; block-0 domains fail at block 1 |
| `fhe_fides_real_d768_t2_attention_a100_asymfix2_20260724` | real block-0 D768 12-head attention | global/worst-token `7.0183e-10`, `1,347.062 s [gpu]` |
| `fhe_range_control_t2_12block_optimized_v2_20260724` | fixed 12-block nonlinear preflight | `[V/A]` worst-token `8.1174e-3`, zero domain violations; not FHE |
| `fhe_fides_real_d768_t2_block0_depth43_FAIL_20260724` | original real full-block schedule | `[V]` fails closed from level exhaustion; zero decrypts |
| `fhe_measured_t2_attention_boundary_20260724` | derived current-schedule runtime boundary | `[A]` `4.49 h` for 12× measured T=2 attention only; excludes MLP/refresh/head |
| `fhe_fides_refresh_d768_t2_native_a100_asymfix2_20260724` | native-GPU encrypted bootstrap refresh + nonlinear tail, block-0 | `[V]` pass, rel-inf `1.045e-3`, restores 21 levels; container exits `139` post-decrypt (teardown crash, non-blocking) |
| `fhe_fides_real_d768_t2_attention_sigmoid13_a100_asymfix2_20260724` | T=2 sigmoid-identity LN1+attention gate, block-0 | `[V]` pass, rel-inf `9.91e-9`, `709.3 s [gpu]` (vs `1347.1 s` exp-schedule), packed at level 22/43 |

The CPU and CUDA toy-block results are the current complete-graph arithmetic anchors.
Their encrypted inputs begin after embedding, and their one final decrypt exists only
to measure the oracle. The real-width LayerNorm and attention results are staged gates,
not a complete block. The twelve-block range-control result is a plaintext preflight,
not an encrypted run. The derived 0.1b and runtime files must never be presented as
measured full-model performance.
