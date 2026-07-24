# Real-weight DNAGPT T=2 sigmoid GPU optimization

This is a versioned optimization of the frozen `fhe/gpu_real/` block-0 gate.
It preserves the same released `dna_gpt0.1b_m` fixture, CKKS parameters,
encrypted-input boundary, final-only decryption boundary, and `4e-2` dual
accuracy gate. It changes only the two-source causal-attention schedule.

For the second row of a `T=2` causal softmax, the exact plaintext algebra is:

```text
delta   = s11 - s10
w1      = sigmoid(delta)
context = v0 + w1 * (v1 - v0)
```

The encrypted implementation approximates sigmoid directly with one fixed
degree-13 Chebyshev series over the public `[-6, 0]` contract. The deeper
sigmoid ciphertext is the first operand of the ciphertext multiplication.
This replaces two nonlinear series and two ciphertext multiplications in each
head with one nonlinear series and one ciphertext multiplication.

The evaluator receives no secret key, performs no decryption, and returns an
encrypted packed output. The client calls `Decrypt` exactly once after the
selected gate. Encrypted token-index embedding lookup remains out of scope;
the encrypted numeric input is token-plus-position embeddings.

## Public approximation and level contract

| Function | Domain | Degree |
|---|---:|---:|
| LN1 inverse square root | `[0.002, 0.012]` | 27 |
| LN2 inverse square root | `[0.50, 1.20]` | 15 |
| attention sigmoid | `[-6, 0]` | 13 |
| tanh GELU | `[-3, 3]` | 15 |

These are `[A]` frozen public fixture/calibration assumptions, never
private-query-derived bounds. The unchanged multiplicative-depth budget is 43.
The predicted optimized maximum levels are:

| Stage | Predicted maximum level |
|---|---:|
| attention context | 20 |
| attention projection | 21 |
| LN2 | 32 |
| block output | 40 |
| packed output | 41 |

The executable records the actual level trace. It fails closed before each MLP
unless `ln2_level + 8 <= 43`, and before final packing unless
`max_output_level + 1 <= 43`.

Run the local plaintext preflight:

```bash
source .venv/bin/activate
python fhe/gpu_real_sigmoid/validate_sigmoid_fixture.py
python -m unittest fhe.gpu_real_sigmoid.test_static_contract
```

The default fixture is the ignored reproducible export at
`checkpoints/fhe_exports/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0`.
`run_sigmoid.sh` checks the pinned manifest and every consumed binary against
`fixture.sha256` before evaluation.

## Pinned image build and Brev run

Required backend: patched FIDESlib commit
`786c7600fb2f16b724e0acf73df367b27b8afed6`, installed as
`fideslib::fideslib`. The default target is A100 `sm_80`.

```bash
chmod +x \
  fhe/gpu_real_sigmoid/build_in_fideslib.sh \
  fhe/gpu_real_sigmoid/run_sigmoid.sh \
  fhe/gpu_real_sigmoid/launch_brev_sigmoid.sh \
  fhe/gpu_real_sigmoid/schedule_when_free.sh

FIDESLIB_ARCH=80-real BUILD_JOBS=2 \
  fhe/gpu_real_sigmoid/build_in_fideslib.sh
```

Run gates from cheap to expensive. Use new immutable tags; do not reuse
evidence paths from `fhe/gpu_real/`.

```bash
FIXTURE=/work/checkpoints/fhe_exports/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_sigmoid/run_sigmoid.sh 0 attention "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_attention_sigmoid13_a100_20260724.json

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_sigmoid/run_sigmoid.sh 0 full "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_block0_sigmoid13_a100_20260724.json
```

Each gate independently requires finite output, global `rel_inf <= 4e-2`, and
worst-token `rel_inf <= 4e-2`. Static and plaintext preflight results are not
FHE evidence; only a pinned GPU run may validate the predicted levels, error,
and latency.

On a shared Brev host, `schedule_when_free.sh` accepts a comma-separated GPU
allowlist and launches only after one GPU has less than 100 MiB allocated,
less than 10% utilization, and zero compute processes for two consecutive
30-second polls. `launch_brev_sigmoid.sh` repeats the zero-process check
immediately before launch, so a race fails closed instead of sharing a GPU:

```bash
fhe/gpu_real_sigmoid/schedule_when_free.sh \
  0,1,2,3,4,5,6,7 \
  /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  fhe/gpu_real_sigmoid \
  real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0 \
  dnagpt-fideslib:786c-asymfix2 \
  attention \
  fhe_fides_real_d768_t2_attention_sigmoid13_a100_asymfix2_20260724 \
  1440
```

## Packing and operation metadata

- `D=768`, `T=2`, 12 heads, MLP width 3072.
- Each token is one ciphertext with four repeated 1024-slot blocks.
- Real matrices are zero-padded to 1024 and use `32 x 32` BSGS.
- Q/K/V and the four MLP-expansion products reuse hoisted baby rotations.
- JSON records matrix-product count, all helper-mediated ciphertext-ciphertext
  and ciphertext-plaintext multiplications, actual level ranges, predicted
  maxima, and the two fail-closed remaining-depth guards.
- The depth-43 graph uses no bootstrap; multi-block composition needs the
  separately validated encrypted refresh boundary.
