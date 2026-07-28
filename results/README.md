# Evidence contract

`results/` is the immutable evidence store. It is the only place performance numbers are considered
real, and it is what the paper cites.

## Rules

- **One run = one file.** Each evaluation writes `runs/<tag>.json` (metrics + config + provenance)
  and, where per-example outputs matter, `runs/<tag>_preds.csv`. These are append-only — never edit
  or overwrite a completed run; a re-run gets a new `<tag>`. All run files stay flat under `runs/`
  regardless of scheme (many harnesses hardcode this path).
- **Every run has a manifest row** in the manifest for its scheme: `pure/manifest.yaml` (Scheme A,
  frozen), `hybrid/manifest.yaml` (Scheme B, active), or `shared/manifest.yaml` (Phase-A plaintext
  baselines and scheme-agnostic FHE scaffolding). Each row records task, weight, dataset+source,
  device, n, metrics, reference, and a `[V]`/`[U]` verdict.
- **A number without a run file does not exist.** Do not quote metrics from chat or logs.
- Per-example prediction CSVs are the frozen **oracle** for later encrypted-inference acceptance.
- FHE correctness runs also record the security level, context shape, executing source
  hashes where available, approximation domains, level trace, decrypt counts, and an
  explicit environment/latency label.

## Current runs

### Shared — Phase-A plaintext baselines + scheme-agnostic FHE scaffolding

| tag | task | headline |
|---|---|---|
| `gsr_aataaa_human_full` | GSR | acc 0.9124, F1 0.916 (n=22,604) |
| `mrna_xpresso_human_1ktest` | mRNA regression | r² 0.562, Pearson 0.753 (n=1,000) |
| `mrna_smoke_regression_sh` | mRNA pipeline check | 2 author examples within ~0.04 |
| `gue_prom_core_all` | GUE core promoter | MCC 0.680, acc 0.840 |
| `gue_prom_300_all` | GUE promoter 300bp | MCC 0.897, acc 0.948 |
| `gue_splice_reconstructed` | GUE splice (3-class) | MCC 0.831, acc 0.895 |
| `fhe_multiblock_plaintext_contract_20260724` | 12 released blocks + GSR-head oracle | `[V]` NumPy/torch contract passes; block-0 domains fail at block 1 |
| `fhe_range_control_t2_12block_optimized_v2_20260724` | fixed 12-block nonlinear preflight | `[V/A]` worst-token `8.1174e-3`, zero domain violations; not FHE |

### Pure — Scheme A (pure non-interactive CKKS), frozen baseline

| tag | task | headline |
|---|---|---|
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
| `fhe_fides_real_d768_t2_attention_a100_asymfix2_20260724` | real block-0 D768 12-head attention | global/worst-token `7.0183e-10`, `1,347.062 s [gpu]` |
| `fhe_fides_real_d768_t2_block0_depth43_FAIL_20260724` | original real full-block schedule | `[V]` fails closed from level exhaustion; zero decrypts |
| `fhe_measured_t2_attention_boundary_20260724` | derived current-schedule runtime boundary | `[A]` `4.49 h` for 12× measured T=2 attention only; excludes MLP/refresh/head |
| `fhe_fides_refresh_d768_t2_native_a100_asymfix2_20260724` | native-GPU encrypted bootstrap refresh + nonlinear tail, block-0 | `[V]` pass, rel-inf `1.045e-3`, restores 21 levels; container exits `139` post-decrypt (teardown crash, non-blocking) |
| `fhe_fides_real_d768_t2_attention_sigmoid13_a100_asymfix2_20260724` | T=2 sigmoid-identity LN1+attention gate, block-0 | `[V]` pass, rel-inf `9.91e-9`, `709.3 s [gpu]` (vs `1347.1 s` exp-schedule), packed at level 22/43 |
| `fhe_fides_real_d768_t2_block0_sigmoid13_a100_asymfix2_20260724` | complete released block-0 (LN1+attention+MLP+LN2+pack), T=2 sigmoid schedule | `[V]` pass, rel-inf `6.41e-6`, packed at level 41/43, `2461.8 s [gpu]`; resolves the earlier depth-43 FAIL |
| `fhe_fides_gpu_multiblock_blocks0_1_refresh_FAIL_20260724` | two-block same-lineage gate (block0 -> refresh -> block1), depth 64 | `[V]` fails closed (crash, exit `139`) during context/key setup; zero decrypts, no evidence written |
| `fhe_fides_gpu_multiblock_bisect_depth50_FAIL_20260724` | same two-block gate, depth 64->50 diagnostic bisection | `[V]` explicit CUDA OOM at `81131/81920 MiB`; supersedes the depth-64 diagnosis -- footprint doesn't fit one A100 regardless of depth |
| `fhe_fides_gpu_multiblock_bisect_depth58_backtrace_FAIL_20260724` | same two-block gate, depth=58 symbolized backtrace + digits=2 side-experiment | `[V]` crash traced to `AddBootstrapPlaintexts -> GPUmalloc`, same code path as the depth-50 OOM; unifies all observed failure modes as one GPU-memory-footprint cause |
| `fhe_fides_gpu_multiblock_multigpu_2gpu_sigsegv_setupconstants_FAIL_20260725` | two-block gate, first real 2-GPU sharding attempt | `[V]` fails closed with a distinct SIGSEGV inside FIDESlib's own multi-GPU `SetupConstants`/`ContextData` path (not OOM, not the depth50/58 code path); confirmed via `cuda-gdb` backtrace |

### Hybrid — Scheme B (hybrid client-assisted CKKS), active

| tag | task | headline |
|---|---|---|
| `fhe_toy_block_scheme_b_native_cpu_20260725` | Scheme B (hybrid) toy complete block | `[V]` pass, global `2.22e-12`, 16 client round trips, depth 20/`ring_dim` 65536 (vs Scheme A toy's depth 49/131072); `1433.4 s [native-cpu]` on a heavily contended shared host, not used for GPU extrapolation |
| `fhe_fides_real_d768_t2_ln1_scheme_b_a100_20260725` | Scheme B (hybrid) real block-0 D768 LayerNorm | `[V]` pass, global `2.35e-10`, 2 client round trips, depth 16/`ring_dim` 65536 (vs Scheme A's depth 43/131072); `2.407 s [gpu]` total = `1.037 s` server GPU linear algebra + `1.370 s` client boundary crossings |
| `fhe_fides_real_d768_t2_attention_scheme_b_a100_20260725` | Scheme B (hybrid) real block-0 D768 LN1+12-head attention | `[V]` pass, global `2.87e-10`, 14 client round trips; `240.29 s [gpu]` total = `236.61 s` server + `3.68 s` client boundary |
| `fhe_fides_real_d768_t2_block0_scheme_b_a100_20260725` | Scheme B (hybrid) real complete block-0 (LN1+attention+MLP+LN2+pack) | `[V]` pass, global `3.32e-10`, 24 client round trips; `372.27 s [gpu]` total = `365.78 s` server + `6.49 s` client boundary (vs Scheme A equivalent `2461.8 s`) |
| `fhe_fides_real_d768_t2_attention_batched_scheme_b_a100_20260726` | Scheme B batched real block-0 D768 LN1+12-head attention correctness gate | `[V]` pass, global `2.18e-10`, 3 physical round trips / 14 logical boundary instances, zero intermediate decrypts; `[U]` timing excluded because shared-host load surged after launch |
| `fhe_fides_real_d768_t2_block0_batched_scheme_b_a100_20260726` | Scheme B batched real complete block-0 correctness gate | `[V]` pass, global `3.51e-10`, 7 physical round trips / 24 logical boundary instances, zero intermediate decrypts; `[U]` timing excluded because shared-host load surged after launch |
| `fhe_fides_real_d768_t2_full_cached_2x_scheme_b_a100_20260727` | Scheme B in-process context/key caching: same block-0 full gate evaluated 2x sharing one setup | `[V]` both iterations pass (global `3.73e-10`, `1.98e-10`); one-time setup measured `5.86s`; `[A]` saves `~1.4%` of the 12-block extrapolation; `[U]` per-iteration timing (`467.24s`/`342.70s`) is contention noise, not compared as a speedup |
| `fhe_fides_real_d768_t2_full_serialized_writer/reader_scheme_b_a100_20260727` | Scheme B cross-process context/key serialization: separate writer/reader `docker run` processes sharing only a bind-mounted state dir | `[V]` reader passes, global `3.46e-10`, zero `GenCryptoContext(`/`EvalMultKeyGen(`/`EvalRotateKeyGen(` in its source; `[V]` honest negative result -- reader's deserialize+LoadContext (`7.88s`) cost *more* than the in-process cached run's full setup (`5.86s`), since deserialization (`3.82s`) exceeds the keygen (`1.27s`) it replaces; no speedup claimed |
| `fhe_fides_real_d768_t2_full_profiled_scheme_b_a100_20260727` | Scheme B GPU-side profiling of `server_linear_algebra_seconds` (24 matmul call sites + one detailed giant-step breakdown) | `[V]` pass, global `2.51e-10`; the 6 matmul stages are 99.5% of server time; within one call, ct-pt multiply is ~100% of cost vs `<0.1%` for rotation/keyswitch; QKV-stage calls average `~41.8s/call` vs `mlp_projection`'s `~13.3s/call` (3.15x), motivating the fix and warm-up investigations below |
| `fhe_fides_real_d768_t2_full_profiled_scheme_b_a100_20260728_control` | Control run: byte-identical profiled binary re-run to bracket the diagcache fix's 6 GPU crash attempts | `[V]` pass, global `2.87e-10`, `515.09s` -- decisive evidence the diagcache fix's 6/6 crashes are a real code issue, not host contention |
| `fhe_fides_real_d768_t2_full_warmup_scheme_b_a100_20260728` | Warm-up-prelude hypothesis test (untimed dummy pass before the real evaluate() call) | `[V]` pass, global `2.93e-10` (3rd independent pass this session); `[U]` timing inconclusive -- run coincided with the session's worst sustained host contention (load 800-1218 for ~43 min), swamping the signal being measured |
| `fhe_fides_real_d768_t2_full_lintransform_scheme_b_a100_20260728` | Isolated single-call-site gate: FIDESlib's native batched `LinearTransform` for the QKV query projection only | `[V]` pass, global `4.59e-10`; `[V]` timing regresses -- the converted call is `1.6-1.9x` slower than its unconverted sibling calls in the same run; lever closed as measured-negative |
| `fhe_fides_real_d768_t3_attention_general_attention_scheme_b_a100_20260728` | First general (T>2) causal-attention Scheme B circuit: real-weight D=768/T=3 LN1 + 12-head general causal attention gate | `[V]` pass, global `3.19e-10`, 6 round trips / 39 logical boundary instances, zero intermediate decrypts; `[U]` no speed claim -- not comparable to any T=2 run |
| `fhe_fides_real_d768_t3_full_general_attention_scheme_b_a100_20260728` | First general (T>2) causal-attention Scheme B circuit: real-weight D=768/T=3 complete block-0 gate | `[V]` pass, global `3.65e-10`, 12 round trips / 54 logical boundary instances, `519.57s [gpu]` = `516.45s` server + `3.12s` client; `[U]` T=3 vs T=2's `372.27s` is not a speed comparison (different token/matmul counts, host load uncontrolled) |

**Diagonal-plaintext-cache fix (`real_dnagpt_fides_scheme_b_diagcache.cpp`): implemented, contract-tested, abandoned as unsafe.** 6 consecutive real-GPU crashes across 6 different physical GPUs, all `Cuda failure ... out of memory`, `addr2line`-confirmed at `Ciphertext::multPt -> Plaintext::adjustPlaintextToCiphertext -> RNSPoly::grow -> GPUmalloc`, bracketed by 2 clean passes of the unmodified baseline the same day -- ruling out pure host contention. Root cause narrowed to reusing a GPU-resident `Plaintext` object across two `multPt` calls (not to "unbounded cache growth," which does not hold up quantitatively: the fix registers *fewer* total distinct plaintexts, `12288` vs the original's `24576`). No `results/runs/` row exists for any diagcache attempt since none produced valid evidence; full account in `docs/hybrid/tasks.md`.

**Native `LinearTransform` primitive (`real_dnagpt_fides_scheme_b_lintransform.cpp`): implemented, contract-tested, correctness confirmed, timing regresses.** Scoped as compatible (three concrete adaptations required, not a drop-in: reaching native FIDESlib types via the wrapper's public device-object accessors, pre-encoding diagonal plaintexts at the ciphertext's exact level, an explicit level-match check since the library's own assert is compiled out under this build's Release/NDEBUG). Converted exactly one of 24 matmul calls (QKV query projection) to FIDESlib's native batched BSGS primitive; the other 23 stayed on the manual path. Real-GPU gate passes correctness (`global_rel_inf=4.59e-10`) but the converted call measures `1.6-1.9x` *slower* than its unconverted sibling calls in the identical run -- the opposite of the hypothesized direction. Converting the remaining 23 call sites was not attempted given this negative signal; full account in `docs/hybrid/tasks.md`.

**General causal-attention Scheme B circuit (`real_dnagpt_fides_scheme_b_general_attention.cpp`): T>2 design gap closed.** Every Scheme B result above was measured against the T=2-only closed-form sigmoid identity, which does not generalize. This circuit replaces it with an exact per-row causal softmax evaluated at the client boundary (`context_i = v0 + sum_{j=1}^{i} weight_ij*(v_j-v0)`), reusing `Client::cross_boundary_impl` unchanged and adding exactly one new `Client` method. Round trips grow as `T*(T-1)/2`, which reduces to the frozen file's `1` at T=2 -- evidence this is a strict generalization, not a coincidentally-agreeing different algorithm. Numpy-proven against a real T=3 oracle (same checkpoint/FASTA as the T=2 fixture) before any C++; both real-GPU gates (`attention`, `full`) pass at the same `~1e-10` precision band as every T=2 run. `[U]` No speed claim is made -- T=3 is not compared against the T=2 anchor (different token/matmul counts), and `O(T)`/`O(T^2)` growth beyond T=3 is unmeasured; full account in `docs/hybrid/tasks.md`.

The CPU and CUDA toy-block results are the current complete-graph arithmetic anchors.
Their encrypted inputs begin after embedding, and their one final decrypt exists only
to measure the oracle. The real-width LayerNorm and attention results are staged gates,
not a complete block. The twelve-block range-control result is a plaintext preflight,
not an encrypted run. The derived 0.1b and runtime files must never be presented as
measured full-model performance.
