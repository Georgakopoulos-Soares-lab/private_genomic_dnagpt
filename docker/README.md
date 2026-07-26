# OpenFHE container

The Python wheel used by this study targets Ubuntu x86-64. On an Apple Silicon Mac,
Docker runs it through `linux/amd64` emulation. Accuracy results remain useful; latency
is tagged `[emu]` and must not be treated as native CPU or GPU performance.

## Build

From the repository root:

```bash
docker build --platform linux/amd64 \
  -f docker/Dockerfile.openfhe \
  -t dnagpt-openfhe .
```

The image pins OpenFHE Python `1.5.1.0.24.4` and NumPy `2.5.1`. It contains no model
weights, datasets, or credentials.

## Sanity check

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/ops/matmul_ckks.py
```

## Mac primitive measurements

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/ops/op_matrix_ckks.py \
  --tag fhe_operator_matrix_d8_20260724

docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/ops/bootstrap_ckks.py \
  --tag fhe_bootstrap_d8_20260724
```

These are accuracy and depth measurements. Their latency is `[emu]`.

## Native Brev correctness run

The complete toy block uses the same Dockerfile built natively on the existing x86-64
Brev node. The exact staging, build, launch, evidence-copy, and inspection commands are
in [docs/pure/brev_runbook.md](../docs/pure/brev_runbook.md).
It is a CPU correctness anchor: the OpenFHE Python wheel does not use the A100s.
The canonical run passed in `330.741 s [native-cpu]` with global rel-inf
`1.49e-3`; see `results/runs/fhe_toy_block.json`.

```bash
docker run --rm --cpus 32 --memory 64g \
  -e OMP_NUM_THREADS=32 -e OPENBLAS_NUM_THREADS=1 \
  -v /data/christos/private_genomic_ml/dnagpt_fhe_20260724:/work \
  -w /work dnagpt-openfhe:20260724 \
  python3 -u fhe/block/toy_block_ckks.py \
  --tag fhe_toy_block \
  --environment "Brev OCI DGXC x86_64 CPU, 32-vCPU container [native]" \
  --latency-label "[native-cpu]"
```

Run tags are immutable. If a tagged JSON already exists, the harness refuses to
overwrite it; use a new tag for a new measurement.

## FIDESlib CUDA image

Build the A100 image on an x86-64 CUDA host:

```bash
docker build \
  -f docker/Dockerfile.fideslib \
  -t dnagpt-fideslib:786c-asymfix2 .
```

The image pins FIDESlib commit `786c7600...`, CUDA 13.0, and its patched OpenFHE
dependency. It installs FIDESlib as a CMake package so the DNAGPT targets link through
`fideslib::fideslib`.

`docker/patches/fideslib-asymmetric-chebyshev.patch` is required. The upstream commit's
GPU interval normalization uses the half-width instead of the midpoint for asymmetric
Chebyshev domains. The repository retains both the failed prepatch measurement and the
passing corrected measurement under `results/runs/`.

The canonical Brev image digest is
`sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4`.
