# Released-weight DNAGPT blocks 0–1 with encrypted refresh

This module is the next correctness gate after the single real block. It
evaluates released `dna_gpt0.1b_m` blocks 0 and 1 at `D=768`, `T=2` in one CKKS
context and key lineage:

```text
encrypted token+position embeddings
  -> released block 0
  -> public /20 conditioning
  -> native FIDESlib bootstrap
  -> public *20 restoration
  -> released block 1
  -> encrypted packed output
  -> exactly one final oracle decrypt
```

There is no intermediate decrypt, client round trip, new context, ciphertext
serialization, or key-lineage conversion.

## Single-context bootstrap API audit

The pinned FIDESlib commit
`786c7600fb2f16b724e0acf73df367b27b8afed6` exposes the required operations on
the same `CryptoContext` used by the block graph:

- `Enable(FHE)`;
- `EvalBootstrapSetup(..., slots=4096, ...)`;
- `EvalBootstrapKeyGen(secretKey, 4096)` during client/key setup;
- `EvalBootstrap(ciphertext, 1, 0, false)` inside the keyless evaluator.

FIDESlib's pinned bootstrap examples use `FLEXIBLEAUTO`, and the CUDA
implementation has an explicit `FLEXIBLEAUTO` modulus-raise path. The current
API can therefore express the two-block lineage without a client decrypt.
There is no API gap requiring a speculative adapter.

The graph has two token ciphertexts, so one logical block-boundary refresh
requires two bootstrap primitive calls. Both calls use the same context and
evaluation-key set. Native FIDESlib ignores requested iterative-bootstrap
precision; this module makes no iterative-accuracy claim.

## Fail-closed status

This source is compile/static ready, not GPU evidence. No PASS is claimed until
the pinned program is compiled and the complete encrypted lineage satisfies:

- all outputs finite;
- global `rel_inf <= 4e-2`;
- worst-token `rel_inf <= 4e-2`;
- exactly zero intermediate decrypts and one final decrypt;
- actual recorded levels remain within depth 64.

The public `/20` bound covers the immutable block-0 fixture output
`[-15.7543375453, 8.67853367362]`. The restored bootstrap error is not assumed
to be acceptable; only the final block-1 oracle gate may establish that.
The pinned plaintext polynomial schedule reaches block 1 with
`global_rel_inf = worst-token rel_inf = 2.66383756824e-5`. This is a positive
approximation baseline, not a prediction that CKKS plus bootstrap will pass.

## Public approximation contract

The module pins:

- multiblock manifest SHA-256
  `3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c`;
- optimized range-control result SHA-256
  `b148e30b42c430405a0a2c41401ab295701e0e1c3655ef00f06984ff9f8faba2`;
- the 15 binary files consumed by this gate in `fixture.sha256`.

Domains and degrees are fixed public assumptions, never selected from a
private value:

| Block | Function | Public domain | Degree |
|---:|---|---:|---:|
| 0 | LN1 inverse square root | `[0.00300306994, 0.01010613479]` | 7 |
| 0 | attention sigmoid | `[-6.06986674334, 0.25]` | 9 |
| 0 | LN2 inverse square root | `[0.50424597027, 1.28958570115]` | 7 |
| 0 | tanh GELU | `[-3.46661639687, 3.46661639687]` | 15 |
| 1 | LN1 inverse square root | `[0.72318097013, 1.50631295770]` | 7 |
| 1 | attention sigmoid | `[-8.98140764102, 0.25]` | 11 |
| 1 | LN2 inverse square root | `[0.74877708323, 1.34400849652]` | 7 |
| 1 | tanh GELU | `[-9.90561224372, 9.90561224372]` | 39 |

Both attention graphs use the exact `T=2` identity:

```text
w1 = sigmoid(s11 - s10)
context = v0 + w1 * (v1 - v0)
```

The deeper sigmoid ciphertext is the first multiplication operand. Block-0 LN1
also uses the range-control contract's public power-of-two variance scale
`0.00390625`; all other two-block LayerNorm scales are one.

## Predicted depth and refresh contract

Parameters are `FLEXIBLEAUTO`, 59-bit scale, depth 64, and bootstrap level
budget `[4,4]`. Based on FIDESlib/OpenFHE's default bootstrap modulus-reduction
depth and the measured single-block level schedule, the static prediction is:

| Stage | Predicted maximum level |
|---|---:|
| block-0 output | 34 |
| conditioned bootstrap input | 35 |
| native bootstrap output | approximately 17 |
| public-scale restored output | approximately 18 |
| block-1 output | approximately 52 |
| packed output | approximately 53 |

These values are `[A]` schedule predictions, not measurements. The executable
records every actual stage level and fails before block MLPs or final packing
when insufficient depth remains. The contract has one refresh boundary and two
bootstrap primitive calls.

## Local validation

```bash
source .venv/bin/activate
python fhe/gpu_multiblock/validate_contract.py
python -m unittest fhe.gpu_multiblock.test_static_contract
bash -n \
  fhe/gpu_multiblock/build_in_fideslib.sh \
  fhe/gpu_multiblock/run_two_block.sh
```

## Build and run in the pinned image

No remote launcher is included: GPU occupancy and launch authorization remain
an orchestration concern.

```bash
FIDESLIB_ARCH=80-real BUILD_JOBS=2 \
  fhe/gpu_multiblock/build_in_fideslib.sh

FIXTURE=checkpoints/fhe_exports/gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0
FIDES_CONTAINER_IMAGE='dnagpt-fideslib:786c-asymfix2@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4' \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_multiblock/run_two_block.sh \
  0 "$FIXTURE" \
  results/runs/fhe_fides_real_d768_t2_blocks0_1_refresh_a100_YYYYMMDD.json
```

The runner verifies the pinned backend commit, manifest, range-control file,
all consumed fixture hashes, unique evidence tag, and immutable output path
before execution.
