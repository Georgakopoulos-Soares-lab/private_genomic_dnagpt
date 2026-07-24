# FIDESlib/OpenFHE encrypted refresh gate

This gate isolates the depth-reset boundary needed to compose real-width DNAGPT blocks. It
uses the exported `D=768`, `T=2` block-0 output as a representative activation. The 1,536
active values are zero-padded into 2,048 CKKS slots because the bootstrap transform requires
a power-of-two slot count:

`encrypted real block output → public /20 conditioning → bootstrap → encrypted square + ε → one final decrypt`

The square is the first ciphertext nonlinearity required to form the next block's LayerNorm
variance. A pass therefore proves that refreshed data remains usable; it is stronger than a
bootstrap-only round trip.

## Security and backend contract

- FIDESlib commit `786c7600fb2f16b724e0acf73df367b27b8afed6` with the repository's
  asymmetric-Chebyshev/OpenFHE patch
- `HEStd_128_classic`, ring dimension exactly `131072`, `FIXEDMANUAL`, 59-bit scale
- the evaluator receives the context, evaluation keys, public mode, and ciphertext only
- no evaluator-side private key and no intermediate decrypt
- exactly one final decrypt for the oracle gate
- immutable JSON (`O_EXCL`) with source, activation, backend, image, level, error, and timing
  metadata

The query-independent public bootstrap bound is `20`, which covers the frozen fixture range
`[-15.7543375453, 8.67853367362]` with about 27% headroom and keeps the conditioned message
away from the CKKS bootstrap boundary at ±1. The input is encrypted one level above exhaustion;
public conditioning consumes that level, so bootstrap sees the representative value at the
depleted-chain boundary.

## Ordered execution

Run the native CUDA primitive first:

```bash
python3 fhe/gpu_bootstrap/validate_static.py
fhe/gpu_bootstrap/build_in_fideslib.sh

ACTIVATION=checkpoints/fhe_exports/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0/oracle__block_output.bin
FIDES_CONTAINER_IMAGE='dnagpt-fideslib:786c-asymfix2@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4' \
FIDES_RUN_ENVIRONMENT='Brev A100 80GB [gpu]' \
fhe/gpu_bootstrap/run_refresh.sh \
  0 native-gpu "${ACTIVATION}" \
  results/runs/fhe_fides_refresh_native_a100_20260724.json
```

Success requires both global and worst-token `rel_inf <= 4e-2`. The pinned native GPU API
performs one core bootstrap pass and ignores its iteration/precision arguments. Do not claim
iterative refinement for that mode.

Only if immutable native evidence fails the accuracy gate, run the explicit fallback as a new
evidence tag:

```bash
FIDES_CONTAINER_IMAGE='dnagpt-fideslib:786c-asymfix2@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4' \
FIDES_RUN_ENVIRONMENT='Brev A100 80GB [gpu] + host OpenFHE fallback' \
fhe/gpu_bootstrap/run_refresh.sh \
  0 cpu-iterative-interop "${ACTIVATION}" \
  results/runs/fhe_fides_refresh_cpu_iter2_a100_20260724.json
```

The fallback downloads the ciphertext coefficients without decrypting, runs OpenFHE's
two-iteration bootstrap in the same crypto context and key lineage, normalizes its noise-scale
degree, uploads the refreshed ciphertext, and continues the square on GPU. JSON separates
download, CPU bootstrap, upload, and GPU tail latency. The fallback is not a client round trip:
it uses evaluation keys only and never receives the secret key.

Neither mode selects a fallback by decrypting an intermediate result. Mode choice is explicit
and must be justified by prior immutable native evidence.

On a shared Brev host, use the fail-closed scheduler rather than sharing a GPU:

```bash
fhe/gpu_bootstrap/schedule_refresh_when_free.sh \
  0,1,2,3,4,5,6,7 \
  /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  refresh_native_v2 \
  real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0/oracle__block_output.bin \
  dnagpt-fideslib:786c-asymfix2 \
  native-gpu \
  fhe_fides_refresh_d768_t2_native_a100_asymfix2_20260724 \
  1440
```

The scheduler requires less than 100 MiB allocated, less than 10% utilization,
and zero compute processes for two consecutive 30-second polls. The launcher
rechecks all three conditions immediately before starting the pinned container.
