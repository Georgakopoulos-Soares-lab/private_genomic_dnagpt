# Brev FHE runbook

This runbook reuses the existing `awesome-gpu-name` workspace. It does not create, stop,
or delete an instance and does not inspect credentials.

## Safety and capacity check

```bash
brev ls --json
brev exec awesome-gpu-name -- \
  "nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader"
```

The historical toy correctness run is CPU-only even though the node has A100s. New
performance work starts with the C++/CUDA FIDESlib parity gate.

## Stage source and build

From the local repository root, copy only the FHE harness and container definition into
the isolated directory:

```bash
COPYFILE_DISABLE=1 tar -czf /private/tmp/dnagpt_fhe_src_20260724.tgz \
  docker fhe
brev exec awesome-gpu-name -- \
  "test ! -e /data/christos/private_genomic_ml/dnagpt_fhe_20260724 && \
   mkdir -p /data/christos/private_genomic_ml/dnagpt_fhe_20260724"
brev copy /private/tmp/dnagpt_fhe_src_20260724.tgz \
  awesome-gpu-name:/data/christos/private_genomic_ml/dnagpt_fhe_20260724/source.tgz
brev exec awesome-gpu-name -- \
  "cd /data/christos/private_genomic_ml/dnagpt_fhe_20260724 && \
   tar -xzf source.tgz && \
   docker build -f docker/Dockerfile.openfhe -t dnagpt-openfhe:20260724 ."
```

Before a measured run, compare local and remote hashes for the executing script and
oracle. The result JSON also records both hashes and the pinned package versions.

```bash
sha256sum fhe/block/toy_block_ckks.py fhe/oracle.py
brev exec awesome-gpu-name -- \
  "cd /data/christos/private_genomic_ml/dnagpt_fhe_20260724 && \
   sha256sum fhe/block/toy_block_ckks.py fhe/oracle.py && \
   docker image inspect dnagpt-openfhe:20260724 --format '{{.Id}}'"
```

## Canonical native-CPU toy block

The immutable tag must not already exist:

```bash
brev exec awesome-gpu-name -- \
  "cd /data/christos/private_genomic_ml/dnagpt_fhe_20260724 && \
   test ! -e results/runs/fhe_toy_block.json && \
   docker run --rm --cpus 32 --memory 64g \
     -e OMP_NUM_THREADS=32 -e OPENBLAS_NUM_THREADS=1 \
     -v /data/christos/private_genomic_ml/dnagpt_fhe_20260724:/work \
     -w /work dnagpt-openfhe:20260724 \
     python3 -u fhe/block/toy_block_ckks.py \
       --tag fhe_toy_block \
       --environment 'Brev OCI DGXC x86_64 CPU, 32-vCPU container [native]' \
       --latency-label '[native-cpu]' \
       --container-image \
       'dnagpt-openfhe:20260724@sha256:6260e3f4ea6364c40c91299936b0ddbe5be11038059001d6842666cb5fabf603'"
```

Copy the completed evidence back without overwriting a local result:

```bash
test ! -e results/runs/fhe_toy_block.json
brev copy \
  awesome-gpu-name:/data/christos/private_genomic_ml/dnagpt_fhe_20260724/results/runs/fhe_toy_block.json \
  results/runs/fhe_toy_block.json
```

## GPU progression

Do not run the unchanged Python harness "on a GPU": the wheel has no GPU execution path.
Build the pinned, locally corrected FIDESlib image and use C++/CUDA:

1. `[V]` prove complete `D=8`, `T=4` C++/CUDA parity once;
2. `[V]` validate the patched asymmetric-Chebyshev backend gate;
3. `[V]` advance to released weights at `D=768`, 12 heads, `T=2`;
4. `[V]` attention is closed; close the sigmoid-schedule 3072-wide MLP/full block
   before increasing sequence length;
5. keep linears, rotations, nonlinear polynomials, and bootstrap GPU resident;
6. record setup separately from encrypted evaluation and reject silent per-operator transfers;
7. fall back to OpenFHE CPU only for a GPU operation that fails the accuracy gate;
8. optimize packing and scheduling one change per immutable run;
9. allocate multiple A100s only after a measured single-GPU memory or throughput limit.

An optimization is retained only if the unchanged global and worst-token `4e-2` oracle
gate passes.

The canonical backend is:

```text
dnagpt-fideslib:786c-asymfix2
sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4
```

Build and smoke locally in the remote workspace:

```bash
docker build -f docker/Dockerfile.fideslib \
  -t dnagpt-fideslib:786c-asymfix2 .

cmake -S fhe/gpu/probes -B /tmp/dnagpt-cheb-build
cmake --build /tmp/dnagpt-cheb-build -j2
```

The Brev launchers enforce a sub-10GB/sub-10%-utilization preflight, refuse to overwrite
logs or evidence, map one physical GPU to container logical device 0, and run detached.
These are the exact historical launch commands for the retained toy and released-weight
numbers (they intentionally refuse to overwrite the already-present tags):

```bash
bash fhe/gpu/launch_brev_toy.sh \
  7 \
  /data/christos/private_genomic_ml/dnagpt_fides_786c_20260724/gpu_v2 \
  dnagpt-fideslib:786c-asymfix2 \
  dnagpt-fideslib:786c-asymfix2@sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4 \
  fhe_fides_toy_a100_asymfix2_20260724

bash fhe/gpu_real/launch_brev_real.sh \
  6 \
  /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  gpu_real_v3 \
  real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0 \
  dnagpt-fideslib:786c-asymfix2 \
  ln1 \
  fhe_fides_real_d768_t2_ln1_a100_asymfix2_20260724

bash fhe/gpu_real/launch_brev_real.sh \
  6 \
  /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  gpu_real_v3 \
  real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0 \
  dnagpt-fideslib:786c-asymfix2 \
  attention \
  fhe_fides_real_d768_t2_attention_a100_asymfix2_20260724

bash fhe/gpu_real/launch_brev_real.sh \
  5 \
  /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  gpu_real_v3 \
  real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0 \
  dnagpt-fideslib:786c-asymfix2 \
  full \
  fhe_fides_real_d768_t2_block0_a100_asymfix2_20260724
```

`context_keygen_load` is service setup. `encrypted_evaluation` is the forward latency.
Do not combine them when extrapolating per-query cost, and do not omit either from the
evidence.
