# Real-weight DNAGPT block-0 GPU gate

This directory moves the no-mid-decrypt proof from the formulaic `D=8` toy to
released DNAGPT `dna_gpt0.1b_m` block-0 weights:

```text
encrypted token+position embeddings (T=2, D=768)
  -> LayerNorm -> QKV -> 12-head causal attention -> projection -> residual
  -> LayerNorm -> 3072-wide GELU MLP -> residual -> encrypted output
  -> one final oracle Decrypt
```

The fixture is the ignored, reproducible export at
`checkpoints/fhe_exports/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0`.
`run_real.sh` checks the manifest and every consumed binary against
`fixture.sha256` before evaluation. The evaluator class receives no secret key.

## Why T=2 first

`T=2` is the smallest non-degenerate causal-attention gate: row 0 has one source
and row 1 has two. It exercises real width, all 12 heads, every real block-0
weight, both nonlinear LayerNorms, softmax normalization, GELU, and both
residuals while limiting ciphertext count. Longer sequence length is a separate
scaling gate after this graph passes.

Attention uses a stable numerator-first T=2 identity. For row 1 it subtracts
the encrypted source-0 score from both logits, then computes:

```text
(v0 + exp(s1-s0) * v1) * reciprocal(1 + exp(s1-s0))
```

This exposes neither score and materializes no probability ciphertext. The
domains are fixed public contract values; they are never selected from a
decrypted private query.

## Public approximation contract

The compiled domains cover the frozen public fixture preflight:

| Function | Domain | Degree |
|---|---:|---:|
| LN1 inverse square root | `[0.002, 0.012]` | 27 |
| LN2 inverse square root | `[0.50, 1.20]` | 15 |
| attention `exp(s1-s0)` | `[-6, 0]` | 13 |
| attention reciprocal | `[1, 2.05]` | 13 |
| tanh GELU | `[-3, 3]` | 15 |

These are `[A]` public fixture/calibration assumptions, not universal bounds for
every DNAGPT input. A representative public calibration and a range-control
strategy are required before private deployment. The executable records this
boundary in every JSON result.

Run the local plaintext preflight:

```bash
source .venv/bin/activate
python fhe/gpu_real/validate_fixture.py
```

## Build and run in the pinned FIDESlib image

Required backend: FIDESlib commit
`786c7600fb2f16b724e0acf73df367b27b8afed6`, installed as the CMake package
`fideslib::fideslib`. The default target is an A100 (`sm_80`).

```bash
chmod +x fhe/gpu_real/build_in_fideslib.sh fhe/gpu_real/run_real.sh
FIDESLIB_ARCH=80-real BUILD_JOBS=2 fhe/gpu_real/build_in_fideslib.sh
```

Run cheap-to-expensive staged gates. Each is an independent encrypted lineage
with exactly one final decrypt and an immutable output path:

```bash
FIXTURE=/work/checkpoints/fhe_exports/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c \
FIDES_RUN_ENVIRONMENT='Brev A100 80GB [gpu]' \
fhe/gpu_real/run_real.sh 0 ln1 "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_ln1_a100_20260724.json

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c \
FIDES_RUN_ENVIRONMENT='Brev A100 80GB [gpu]' \
fhe/gpu_real/run_real.sh 0 attention "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_attention_a100_20260724.json

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c \
FIDES_RUN_ENVIRONMENT='Brev A100 80GB [gpu]' \
fhe/gpu_real/run_real.sh 0 full "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_block0_a100_20260724.json
```

Every gate requires finite output, global `rel_inf <= 4e-2`, and worst-token
`rel_inf <= 4e-2`. An earlier gate provides a fail-closed diagnostic boundary;
it is not evidence that the later graph passed.

## Implementation notes

- BSGS is `32 x 32`: 31 baby rotations are issued through FIDESlib's vector
  hoisting overload, followed by 31 giant rotations. The real `768 x 768`
  matrices are zero-padded to `1024 x 1024`.
- Each token is one ciphertext with four repeated 1024-slot blocks. A block
  contains 768 model values and 256 zeros. Power-of-two packing makes rotations
  and `AccumulateSum(..., 1024, 1)` explicit; the first two blocks are masked
  into one ciphertext for the single final decrypt.
- Q/K/V and the four MLP-expansion products reuse hoisted baby rotations.
- This block is leveled at depth 43 and uses no bootstrap. Composition requires
  a separately validated encrypted refresh boundary.
- Encrypted token-index embedding lookup remains out of scope; the numeric
  token-plus-position embedding is the encrypted input boundary.
