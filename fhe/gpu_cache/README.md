# FIDESlib public evaluation-key cache parity gate

This module tests whether DNAGPT's expensive real-width FIDESlib context and
evaluation keys can be provisioned once, then imported by a clean GPU process.
It is a bounded parity gate, not a claim that key generation itself is faster.

The sealed cache contains exactly:

- `crypto-context.bin` and FIDESlib's `crypto-context.bin.dev` sidecar
- `public-key.bin`
- `eval-mult.bin`
- `eval-automorphism.bin`
- project-owned `manifest.json`

It never contains the private key. `cache_contract.py` rejects extra files,
symlinks, hash/size drift, parameter drift, image drift, source drift, and any
private-key-like filename. The manifest binds the pinned FIDESlib commit, the
patched image digest, full real-gate CKKS parameters, source hashes, and every
required rotation.

## Rotation contract

`rotation_contract.py` derives the key vector from
`../gpu_real/src/real_dnagpt_fides.cpp`: BSGS baby rotations, BSGS giant
rotations, and the base-4 `AccumulateSum(1024, 1)` keys are canonicalized,
deduplicated, and sorted. The result is 63 exact rotations. `[V]` This matches
the frozen executable and measured attention-gate JSON. An earlier 65-key hand
estimate is superseded; this module will not invent two unused keys.

The pinned CUDA image intentionally has no Python. Therefore the exact generated
vector is committed as `include/generated_rotation_contract.hpp`; CMake consumes
that header directly. Developer/Brev-host static checks regenerate it in memory
and fail if it differs from the real source:

```bash
python3 fhe/gpu_cache/rotation_contract.py \
  --source fhe/gpu_real/src/real_dnagpt_fides.cpp \
  --check-cpp fhe/gpu_cache/include/generated_rotation_contract.hpp
```

## Security boundary

Provisioning needs the private key to create evaluation keys. It serializes
that key only to an explicit client/oracle path outside the cache, with a
process `umask` of `077` and mode `0600`. A production client should own this
path. The Brev host runner uses a fresh `/tmp` client directory, mounts it
separately from the public cache, and removes it at exit.

The clean reload sequence is load-bearing:

```text
context -> public key -> eval-mult -> eval-automorphism -> GPU LoadContext
```

OpenFHE evaluation-key maps are process-global. FIDESlib copies the imported
keys to GPU in `LoadContext`, so calling it earlier would silently produce an
incomplete GPU context.

`ReloadEvaluator` receives only the context, rotation vector, and ciphertext.
It executes one ciphertext multiplication and every one of the 63 rotations.
Each result is masked into a separate slot of one ciphertext. The client key is
loaded only after encrypted evaluation finishes, and exactly one final decrypt
checks all 64 values. There is no intermediate decrypt.

## Build in the pinned image

Required image:

```text
dnagpt-fideslib:786c-asymfix2@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4
```

The build path inside that image is Python-free:

```bash
FIDESLIB_ARCH=80-real BUILD_JOBS=2 \
  fhe/gpu_cache/build_in_fideslib.sh
```

Run the Python static/manifest tools on the Brev host, not in the container.
Do not install Python or otherwise mutate/rebuild the pinned image.

## Brev host orchestrator

`run_brev_host.sh` runs on the Brev VM host. It verifies the exact local image
ID, checks GPU occupancy, runs host-side Python audits, builds in the pinned
image, and starts two separate clean GPU containers:

1. provision C++ binary: writes the public cache and the external client key to
   distinct mounts;
2. reload C++ binary: mounts the sealed public cache read-only and the temporary
   client directory at a separate read-only path.

Manifest seal/verification occurs on the host between those containers. The
client key is never written or mounted inside the public-cache directory.
Cache and evidence paths must not already exist.

```bash
PROJECT_ROOT=/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724
IMAGE_TAG=dnagpt-fideslib:786c-asymfix2

"${PROJECT_ROOT}/fhe/gpu_cache/run_brev_host.sh" \
  7 \
  "${PROJECT_ROOT}" \
  "${PROJECT_ROOT}/cache/full-real-v1" \
  "${PROJECT_ROOT}/results/fhe_fides_cache_reload_a100_YYYYMMDD.json" \
  "${IMAGE_TAG}"
```

`parity_gate.sh` is a compatibility alias with the same five arguments. The
orchestrator passes logical GPU `0` inside each single-GPU container while its
first argument selects the physical GPU. Do not copy `/tmp` client material
into repository evidence; keep only the sealed public cache and immutable JSON.

`provision_cache.sh` and `reload_cache.sh` are Python-free thin wrappers for
manual use inside an already prepared container. Manifest seal/verification
still must be performed on the host between them.

## Pass criteria

The reload gate exits successfully only when:

- the manifest and all five public artifacts match their SHA-256/size contract;
- the context sidecar contains the exact sorted 63-step vector;
- multiplication and all rotations execute after a clean reload;
- output is finite;
- global relative-infinity error, worst rotation relative error, and
  multiplication relative error are each at most `4e-2`;
- unselected-slot leakage is at most `4e-2`;
- static audit confirms zero decrypt in the evaluator and one final decrypt.

The result separates deserialization and GPU-load time, making the cache's
benefit directly comparable with fresh `context_keygen_load` measurements.
