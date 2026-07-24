# FIDESlib GPU toy block

This directory is the first CUDA gate for the encrypted DNAGPT path. It evaluates one
deterministic `D=8`, `T=4`, two-head DNAGPT block with a `4D=32` GELU MLP:

`encrypted embeddings → LN → QKV → causal attention → projection → residual → LN → MLP → residual → encrypted packed output`

The evaluator receives no private key. The four encrypted outputs are made disjoint during
the final projection and added into one ciphertext. `main` performs one final readout for
the plaintext-oracle gate; there is no evaluator-side readout or client round trip.

## Pinned backend

- FIDESlib commit: `786c7600fb2f16b724e0acf73df367b27b8afed6` (v2.1.3)
- security: `HEStd_128_classic`
- scaling: `FLEXIBLEAUTO`, 50-bit scale, depth 49
- GPU bootstrap: not used by this one-block leveled circuit

The executable rejects any other FIDESlib commit. It uses the public v2.1.3 API:
`AccumulateSum`, `GetChebyshevCoefficients`, `EvalChebyshevSeries`, and the vector
`EvalFastRotation` overload. It does not use `EvalDivide`.

The formulaic fixture and exact plaintext graph can be checked without FIDESlib:

```bash
fhe/gpu/validate_fixture_local.sh
```

## Build and run in the FIDESlib image

The image must contain the pinned checkout at `/opt/FIDESlib` and its patched OpenFHE 1.5.1
installation. FIDESlib itself must be installed so the package exists at
`/usr/local/share/fideslib/cmake/fideslibConfig.cmake`; the toy target consumes
`fideslib::fideslib` and does not rebuild the backend.

```bash
chmod +x fhe/gpu/build_in_fideslib.sh fhe/gpu/run_toy.sh
FIDESLIB_ARCH=80-real BUILD_JOBS=2 fhe/gpu/build_in_fideslib.sh

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:2.1.3 \
FIDES_RUN_ENVIRONMENT='Brev A100 80GB [gpu]' \
fhe/gpu/run_toy.sh 0 /work/results/runs/fhe_fides_toy_a100_20260724.json
```

The output path is immutable: both the launcher and executable refuse to overwrite it.
Success requires finite output, global `rel_inf <= 4e-2`, and worst-token
`rel_inf <= 4e-2`.

## Design notes

- Inputs are embedded numeric token vectors, not encrypted token indices. Private embedding
  lookup remains a separate problem.
- Token ciphertexts repeat their eight values across four slot blocks. This makes every
  `AccumulateSum(..., 8, 1)` a per-token broadcast and allows one final packed readout.
- Matrix products use `2 × 4` BSGS. Baby rotations use the public vector overload, which
  invokes FIDESlib's native GPU hoisting at the pinned commit.
- Attention uses the locally validated numerator-first schedule and a Chebyshev reciprocal;
  no encrypted division primitive is assumed.
- The encrypted evaluator applies no query-derived/public softmax shift. The plaintext oracle
  uses the conventional max shift only for floating-point stability; exact softmax is invariant
  to that shift.
- Approximation domains are fixed public contract values. Changing the fixture, domains, or
  polynomial degrees requires a new evidence tag rather than overwriting a prior run.
