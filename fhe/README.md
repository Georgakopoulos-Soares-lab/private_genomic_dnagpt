# Encrypted DNAGPT block study

This directory tests whether DNAGPT's transformer arithmetic can remain encrypted from
embedded input vectors to block output. It contains a Python/OpenFHE oracle path, a
C++/CUDA FIDESlib block, released-weight `D=768` gates, a 12-block export contract,
and a plaintext-first range-control schedule. It is not a deployed private-inference
service.

## Scope

Included:

- BSGS linear maps for Q, K, V, attention projection, and the first MLP
  projection; a diagonal final MLP projection that folds in output packing
- LayerNorm through a Chebyshev inverse-square-root approximation
- multi-head causal softmax through Chebyshev exponential and reciprocal
- DNAGPT's `GELU("tanh")`
- residual additions
- one OpenFHE context, no intermediate decryption, and one final output decryption

Not included:

- encrypted token-index embedding lookup
- all twelve 0.1b blocks
- physical client/server key custody and transport

The toy input is the already-embedded numeric vector. An encrypted token index cannot be
used as a normal array index; private embedding lookup remains separate work. HE-LRM
([arXiv:2506.18150](https://arxiv.org/abs/2506.18150)) describes this as a substantial
FHE subproblem and evaluates specialized lookup and packing techniques.

## Acceptance contract

`fhe/oracle.py` defines the plaintext graph and the frozen gate:

```text
global rel_inf <= 4e-2
worst-token rel_inf <= 4e-2
all decrypted values finite
```

The block uses fixed polynomial domains declared in
`fhe/block/toy_block_ckks.py`. They cover the deterministic toy input. Coverage for
unseen genomic inputs is unresolved and would require a public calibration set plus a
fail-closed range policy.

`EvalOnlyContext` blocks access to `Decrypt` during evaluation. After the block finishes,
the token outputs are homomorphically masked into one packed ciphertext. The harness
then makes one final `Decrypt` call for oracle validation.

## Run

Build the pinned image as described in `docker/README.md`, then:

```bash
# Combined operator matrix
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/ops/op_matrix_ckks.py

# Bootstrap refresh
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/ops/bootstrap_ckks.py

# Correctness/depth screens for GPU-targeted schedules
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/optimizations/local_ckks_probe.py

# Assumption-labeled full-model runtime boundary
python3 fhe/runtime_projection.py
```

Pass `--tag <new-tag>` to write immutable evidence under `results/runs/`.
Each isolated operator also has its own entry point:
`matmul_ckks.py`, `layernorm_ckks.py`, `softmax_ckks.py`, and `gelu_ckks.py`.

The complete toy block's canonical correctness run is native x86-64 CPU on the existing
Brev node; see [the Brev runbook](../docs/pure/brev_runbook.md). The OpenFHE
Python wheel is CPU-only, so assigning an A100 does not accelerate it.

The GPU path is under `fhe/gpu/`, `fhe/gpu_real/`, and `fhe/gpu_bootstrap/`.
The all-block oracle and nonlinear preflight live under `fhe/multiblock/` and
`fhe/range_control/`. It requires the pinned image from
`docker/Dockerfile.fideslib`, including the audited asymmetric-Chebyshev patch. The
unpatched backend failed a load-bearing inverse-square-root gate at `rel_inf=0.292889`;
the patched image passes at `1.5282e-5`.

The canonical block passed with global rel-inf `1.49e-3`, worst-token rel-inf
`2.21e-3`, zero intermediate decrypt attempts, one final decrypt, and
`330.741 s [native-cpu]` evaluation. See
`results/runs/fhe_toy_block.json` and
`docs/pure/measurements.md`.

The corresponding CUDA block passes with global rel-inf `1.8423e-4`, worst-token
rel-inf `4.8125e-4`, and `93.405 s [gpu]` encrypted evaluation. Released-weight
`D=768`, `T=2` LayerNorm and 12-head attention gates pass at `3.8193e-10` and
`7.0183e-10`, respectively. See the immutable
`fhe_fides_*_asymfix2_20260724.json` results for exact image and source hashes.
