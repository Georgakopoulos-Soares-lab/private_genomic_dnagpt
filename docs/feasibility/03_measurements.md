# Toy-block measurement and 0.1b boundary

## CUDA progression — current measured state

The C++/CUDA path now has three load-bearing passes on Brev A100s:

| Gate | Result | Accuracy | Encrypted evaluation |
|---|---:|---:|---:|
| complete formulaic block, `D=8`, `T=4`, 2 heads | `[V]` pass | global `1.8423e-4`; worst token `4.8125e-4` | `93.405 s [gpu]` |
| real block-0 LayerNorm, `D=768`, `T=2` | `[V]` pass | global `3.8193e-10`; worst token `8.0169e-10` | `1.779 s [gpu]` |
| real block-0 LayerNorm + QKV + 12-head attention + projection, `D=768`, `T=2` | `[V]` pass | global/worst token `7.0183e-10` | `1,347.062 s [gpu]` |

All three runs use `HEStd_128_classic`, encrypted embedded numeric inputs, one evaluator
context/key lineage, zero intermediate decrypt attempts, and one final oracle decrypt.
The complete toy run includes LayerNorm, QKV, two-head causal attention, projection,
both residuals, and the 4D GELU MLP. The real-width gate uses released DNAGPT block-0
weights and a power-of-two 4096-slot layout.

The first FIDESlib GPU nonlinear probe failed at `rel_inf=0.292889` despite using the
upstream asymmetric-interval fix. `[V]` The GPU code centered `[a,b]` at `(b-a)/2`;
the correct affine map uses `(a+b)/2`. The pinned two-branch patch in
`docker/patches/fideslib-asymmetric-chebyshev.patch` reduces the same inverse-square-root
probe to `1.5282e-5` and is required by every retained GPU result.

Evidence:

- `results/runs/fhe_gpu_asymmetric_chebyshev_prepatch_20260724.json`
- `results/runs/fhe_gpu_asymmetric_chebyshev_asymfix2_20260724.json`
- `results/runs/fhe_fides_toy_a100_asymfix2_20260724.json`
- `results/runs/fhe_fides_real_d768_t2_ln1_a100_asymfix2_20260724.json`
- `results/runs/fhe_fides_real_d768_t2_attention_a100_asymfix2_20260724.json`

## Twelve-block range contract — `[V/A]` plaintext preflight

The released-weight export now covers all 12 blocks plus the GSR classifier. An
independent NumPy float64 lineage matches upstream torch after every block and the
head. At `T=2`, block 11 has rel-inf `1.1401e-6` and the head has `1.5380e-6`.
This freezes the oracle but is not an encrypted run.

`[V]` Blindly repeating the block-0 polynomial intervals is invalid beginning at
block 1. On the fixed public sample, variance-plus-epsilon reaches `486.059`,
attention delta spans `[-20.136, 9.735]`, the old exp/reciprocal denominator reaches
`16,891.851`, and GELU input spans `[-13.922, 38.343]`.

A fixed plaintext approximation preflight then tests the CKKS-suitable repairs:

- T=2 attention uses the exact identity
  `softmax([s0,s1])[1] = sigmoid(s1-s0)`, eliminating the unstable exp-sum
  denominator.
- LayerNorm uses the exact public scaling identity
  `inv_sqrt(v) = inv_sqrt(v/s) / sqrt(s)` before a fixed polynomial.
- GELU uses a fixed full-domain polynomial including both tails; it never clips.
- every domain is selected from the immutable public fixture before evaluation;
  an out-of-domain value fails closed.

All 12 block outputs and the classifier head pass with zero domain violations.
The worst block global rel-inf is `1.0600e-3`, worst-token rel-inf is
`8.1174e-3`, worst head-stage rel-inf is `6.1849e-3`, and the `N/A` label is
preserved. Per-block nonlinear depth proxies fall in `[14,24]`; these are not
measured CKKS levels.

Evidence:

- `results/runs/fhe_multiblock_plaintext_contract_20260724.json`
- `results/runs/fhe_range_control_t2_12block_optimized_v2_20260724.json`

The per-block-domain approach follows encrypted-transformer practice: OpenFHE maps a
caller-supplied Chebyshev interval but does not clamp inputs, while
[Orion](https://arxiv.org/abs/2311.03470) calibrates nonlinear modules and separately
optimizes bootstrap placement. Stable shifted/scaled softmax and interval-aware
inverse-square-root designs are described by
[NEXUS](https://eprint.iacr.org/2024/136) and
[THOR](https://eprint.iacr.org/2024/1881). These sources motivate the circuit, but
the measured DNAGPT result above is repository-local evidence.

## Complete toy block — `[V]` pass

The native Brev run completed one `D=8`, `T=4`, two-head DNAGPT-shaped block:

```text
encrypted embedded vectors
  -> LayerNorm
  -> Q/K/V
  -> two-head causal softmax attention
  -> output projection + residual
  -> LayerNorm
  -> 4D tanh-GELU MLP + residual
  -> homomorphic output pack
  -> one final decrypt for oracle measurement
```

| Property | Measurement |
|---|---:|
| security | `HEStd_128_classic` |
| ring dimension | 131072 |
| multiplicative depth | 49 |
| global rel-inf | `1.4908e-3` |
| worst-token rel-inf | `2.2093e-3` |
| gate | `4e-2` |
| intermediate decrypt attempts | 0 |
| final decrypt calls | 1 |
| context and key generation | `8.841 s [native-cpu]` |
| encrypted evaluation | `330.741 s [native-cpu]` |
| verdict | `[V]` pass |

Evidence: `results/runs/fhe_toy_block.json`, SHA-256
`6238979e566fc02127cc8249e94bb5308309c3b02df4d82f5a52f35f28a935e8`.
The result records the executing script and oracle hashes, OpenFHE and NumPy versions,
Python version, image digest, polynomial degrees, and full level trace.

The output-mask multiplication initially exceeded the level-49 chain. The passing
schedule folds each token's output mask into the plaintext diagonals of the final MLP
projection and then packs the already-disjoint ciphertexts with additions. A separate
encrypted shallow probe was used during implementation; the complete immutable block
run is the load-bearing validation because it passes the unchanged oracle.

## What the pass proves

- `[V]` The arithmetic graph of a complete DNAGPT-shaped block is expressible in
  OpenFHE CKKS without intermediate decryption at this toy shape.
- `[V]` Multi-head causal attention, both LayerNorms, the 4D GELU MLP, both residuals,
  and final encrypted output packing coexist in one context and key lineage.
- `[V]` The approximation error is about 18 times below the declared worst-token gate.
- `[V]` The OpenFHE Python correctness path is CPU-only; this run did not use an A100.

It does not prove:

- `[V]` LayerNorm and attention correctness at `D=768`, 12 heads, real block-0 weights
- `[U]` encrypted MLP/full-block correctness at `D=768`, or any `T>2`
- `[U]` correctness across two or more blocks and bootstrap boundaries
- `[U]` practical latency or memory for the 0.1b model
- `[U]` encrypted token-index embedding lookup
- `[U]` client/server key custody, transport, or client-only result decryption

## Derived 0.1b work boundary — `[A/U]`

`fhe/extrapolate.py` applies the measured toy depth and standalone bootstrap refresh to
a 12-block, `D=768`, 12-head, `T=512` target. It deliberately emits work counts and
lower bounds, not a total latency estimate.

| Derived quantity | Value |
|---|---:|
| per-token `D×D` BSGS plan | 32 baby × 24 giant steps |
| rotations per `D×D` linear | 54 |
| linear evaluations, 12 blocks | 73,728 |
| linear rotations, 12 blocks | 3,981,312 |
| causal query/key pairs, 12 blocks | 18,911,232 |
| score + value ciphertext products, 12 blocks | 37,822,464 |
| assumed sequential critical-path levels | 588 |
| optimistic critical-path refresh lower bound | 56 |
| ideal dense activation ciphertext minimum | 6 |
| fresh activation coefficient-byte lower bound | 629,145,600 bytes (600 MiB) |

Evidence: `results/runs/fhe_0p1b_extrapolation_20260724.json`. The file hashes both
measured inputs.

The refresh count is only an optimistic critical-path lower bound. A real attention
schedule must refresh many parallel ciphertexts; it is not a total bootstrap count. The
600 MiB figure is coefficient storage for six ideally packed fresh ciphertexts only. It
excludes guard slots, evaluation keys, object overhead, layouts, and live intermediates.
It also assumes the toy run's ring dimension and 50-tower fresh chain; real-width
security parameters may increase or change that value.

The standalone bootstrap took `53.814 s [emu]`, but multiplying emulated toy or
bootstrap time alone is not a valid forecast. Packing, hoisting, concurrency,
host/device transfers, and ciphertext residency fundamentally change the schedule.

## Runtime boundary

`fhe/runtime_projection.py` produced the pre-real-width planning diagnostics below.
They remain useful schedule-rejection evidence, but the time ranges are historical
and superseded for the current T=2 graph by the measured-input boundary that follows.

| Path | Derived time | Meaning |
|---|---:|---|
| repeat the toy CPU layout by causal-pair ratio | `9.91 years` | `[A]` schedule rejection; it even ignores `D=8 → 768` widening |
| substitute published FIDESlib RTX 4090 primitive times into the current per-token counts | `12.95 hours` | `[A]` unoptimized GPU subtotal; excludes nonlinearities, bootstrap, transfers, and memory stalls |
| first complete unoptimized GPU implementation | order of `10+ hours` | `[A]` working expectation |
| optimized, packed, GPU-resident `D=768/T=512/12` pass | roughly `0.5–4 hours` | `[A]` planning interval to replace with measurements |

Evidence: `results/runs/fhe_runtime_projection_20260724.json`.

The CPU number is not how long a good implementation should take; it shows why the
per-token Python schedule must not be scaled. The 12.95-hour subtotal directly applies
FIDESlib's published 1.107 ms rotation, 1.084 ms ciphertext multiplication, and 21.74 µs
plaintext multiplication to the unoptimized work counts. Its GPU and CKKS parameters
differ from the planned A100, and packing can remove or amortize much of that work.

The first runtime boundary grounded in this repository's real-width A100 measurement is:

| Measured-input derivation | Time | Qualification |
|---|---:|---|
| 12 × block-0 `D=768/T=2` LayerNorm + QKV + 12-head attention + projection | `4.49 h` | `[A]` current-schedule lower boundary; excludes every MLP, refresh, task head, and later-block domain change |

Evidence: `results/runs/fhe_measured_t2_attention_boundary_20260724.json`, derived
from the passing attention result with encrypted evaluation time `1,347.062388 s
[gpu]`. It is not a measured 12-block run. The current graph is specialized to
`T=2`, so the document intentionally leaves task-representative `T=103` runtime
`[U]` rather than inventing linear or quadratic scaling.

The strongest external comparator is
[EncryptedLLM](https://proceedings.mlr.press/v267/de-castro25a.html), which reports over
200× acceleration of a 12-block, `D=768` encrypted GPT-2 small forward from several
hours to a few minutes on an A100 80GB. Its benchmark generates token 128 after earlier
input processing is amortized away, so it is evidence that the optimized regime exists,
not a DNAGPT `T=512` runtime prediction. It also reports about 550 ms to refresh 20
levels at 128-bit security.

The current expectation is therefore `[A]` **well over 4.49 hours** for the first
unoptimized T=2 full pass and **hours, not seconds** for the first useful pass. A
future packed and GPU-resident schedule may still target tens of minutes to a few
hours, but that is an optimization target—not a claim supported by the present
per-token schedule.
The first `D=768`, `T=8/16` block benchmark must replace this range with a measured
operation-level model before a `T=512` run is approved.

## Optimization evidence

The local encrypted screen already establishes:

- `[V]` BSGS reduces `D=16` rotations from 15 to 6 at the same one-level depth.
- `[V]` hoisted baby rotations preserve the result and reduce modeled key-switch
  decompositions from 6 to 4; CUDA must determine whether that becomes a speedup.
- `[V]` numerator-first attention produces the same weighted sum while saving two
  levels.
- `[V]` degree-5 GELU on `[-2,2]` saves one level versus degree 7 with `4.26e-3`
  sample-grid rel-inf. It is only a candidate until the full-block oracle passes.

Evidence: `results/runs/fhe_local_optimizations_20260724.json`.

## Verdict and next measurement

Initial evidence is positive for **arithmetic feasibility**: every primitive, refresh,
the CPU and GPU complete toy blocks, and the first released-weight `D=768` GPU gate pass
with substantial error margin. `[V]` The formulaic GPU block is about 3.5× faster than
the native CPU block at this tiny shape, but `93.405 s` is still not practical latency.

Practicality remains `[U]` until the full released-weight block, encrypted refresh,
block composition, and longer sequence layouts pass. Do not run twelve blocks on the
Python CPU path. Complete the staged real-weight `D=768`, 12-head, `T=2` full-block
gate, then the scaled block-boundary refresh and two-block composition gates. Every retained optimization must
pass the unchanged global and worst-token oracle and write a new immutable run.

The first full released-weight attempt is retained as a `[V]` failure:
`fhe_fides_real_d768_t2_block0_depth43_FAIL_20260724.json`. It reached token-0
block output, but token 1 entered LayerNorm2 at level 37; the remaining MLP plus
output pack required level 46 while the chain ended at 43. No final decrypt occurred.
The active replacement uses the exact two-way identity
`w1=sigmoid(s11-s10); context=v0+w1*(v1-v0)`, removing the reciprocal polynomial
and approximately five critical-path levels. A depth-49 retry of the old graph is
only a fallback/control.
