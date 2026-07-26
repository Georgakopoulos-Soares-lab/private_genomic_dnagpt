# Tasks — Scheme A (pure non-interactive CKKS), frozen

Frozen 2026-07-25 as the paper's non-interactive ablation baseline. No further Scheme A
runs are planned. See [../roadmap.md](../roadmap.md) for the shared acceptance contract,
[../shared/architecture_options.md](../shared/architecture_options.md) for why the
project pivoted away from this architecture, and [../hybrid/tasks.md](../hybrid/tasks.md)
for the active Scheme B evidence that replaced it.

Commands run from the repo root. Phase B uses the pinned Docker image or the documented
Brev container. Each reported number has exactly one command that regenerates it. Raw
outputs: `results/runs/<tag>.json` (+ `_preds.csv`).

## Phase B — encrypted operator and toy-block feasibility

Primitive and bootstrap latency below is `linux/amd64` OpenFHE running under Apple
Silicon emulation and is tagged `[emu]`. Build the pinned image once:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.openfhe -t dnagpt-openfhe .
```

### Operator matrix

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/ops/op_matrix_ckks.py \
  --gelu-profile broad \
  --tag fhe_operator_matrix_d8_20260724
```

Result: `results/runs/fhe_operator_matrix_d8_20260724.json`.

### Bootstrap refresh

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/ops/bootstrap_ckks.py \
  --tag fhe_bootstrap_d8_20260724
```

Result: `results/runs/fhe_bootstrap_d8_20260724.json`.

### Toy-block GELU profile on native Brev CPU

```bash
brev exec awesome-gpu-name -- \
  "docker run --rm --cpus 8 --memory 16g \
   -e OMP_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=1 \
   -v /data/christos/private_genomic_ml/dnagpt_fhe_20260724:/work \
   -w /work dnagpt-openfhe:20260724 \
   python3 -u fhe/ops/op_matrix_ckks.py \
     --only gelu --gelu-profile toy \
     --environment 'Brev OCI DGXC x86_64 CPU, 8-vCPU container [native]' \
     --latency-label '[native-cpu]' \
     --tag fhe_gelu_toy_profile_brev_cpu_20260724"
```

Result: `results/runs/fhe_gelu_toy_profile_brev_cpu_20260724.json`.

### Embedded-input-to-block-output encrypted toy block — `[V]` PASS

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

Result: `results/runs/fhe_toy_block.json`: global rel-inf `1.49e-3`,
worst-token rel-inf `2.21e-3`, zero intermediate decrypt attempts, one final decrypt,
and `330.741 s [native-cpu]`. The encrypted input is an embedded numeric vector;
encrypted token-index lookup is outside this run.

### Derived 0.1b boundary

```bash
python fhe/extrapolate.py --tag fhe_0p1b_extrapolation_20260724
```

Result: `results/runs/fhe_0p1b_extrapolation_20260724.json`. This is an `[A/U]` work-count
derivation from the measured toy and bootstrap runs, not an encrypted 0.1b execution.

### Local optimization correctness screen

```bash
docker run --rm --platform linux/amd64 \
  -v "$PWD":/work -w /work dnagpt-openfhe \
  python3 -u fhe/optimizations/local_ckks_probe.py \
  --tag fhe_local_optimizations_20260724
```

Result: `results/runs/fhe_local_optimizations_20260724.json`: all probes pass. BSGS
reduces rotations `15 → 6`; numerator-first attention saves two levels; degree-5 GELU
uses four levels with `4.26e-3` sample-grid rel-inf. Mac times are
`[emu-directional]`, not GPU speedups.

### Runtime boundary

```bash
python3 fhe/runtime_projection.py --tag fhe_runtime_projection_20260724
```

Result: `results/runs/fhe_runtime_projection_20260724.json`. This is an `[A/U]`
planning boundary, not a benchmark: the repeated toy CPU layout is rejected; direct
published-primitive substitution gives a `12.95 h` unoptimized GPU subtotal; the
`0.5–4 h` range is now a historical optimized target, superseded for the current
unoptimized T=2 graph by the real-width measurement below.

The first real-width measured-input boundary is reproduced with:

```bash
source .venv/bin/activate
python fhe/measured_runtime_boundary.py \
  --tag fhe_measured_t2_attention_boundary_20260724
```

Result: `results/runs/fhe_measured_t2_attention_boundary_20260724.json`. It repeats
the measured `D=768/T=2` block-0 attention time twelve times to obtain `4.49 h`.
This is an `[A]` lower boundary for the current unoptimized T=2 schedule: it excludes
all MLPs, refreshes, the task head, and later-block range handling. It is not a
12-block encrypted measurement and is not extrapolated to task-scale `T=103`.

### CUDA toy and released-weight gates

Build and run the pinned CUDA modules using the exact commands in
[`brev_runbook.md`](brev_runbook.md):

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

Immutable results:

- `fhe_fides_toy_a100_asymfix2_20260724.json` — `[V]` pass.
- `fhe_fides_real_d768_t2_ln1_a100_asymfix2_20260724.json` — `[V]` pass.
- `fhe_fides_real_d768_t2_attention_a100_asymfix2_20260724.json` — `[V]` pass.
- `fhe_fides_real_d768_t2_block0_depth43_FAIL_20260724.json` — `[V]` fail
  before final decrypt; the exact T=2 sigmoid replacement is the next run.

### Sigmoid-schedule attention gate

Fail-closed scheduled launch on the Brev GPU host (waits for a GPU to be
process-free for two consecutive 30 s polls, reserves it, then runs):

```bash
fhe/gpu_real_sigmoid/schedule_when_free.sh \
  0,1,2,3,4,5,6,7 \
  /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  gpu_real_sigmoid_v1/fhe/gpu_real_sigmoid \
  real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0 \
  dnagpt-fideslib:786c-asymfix2 \
  attention \
  fhe_fides_real_d768_t2_attention_sigmoid13_a100_asymfix2_20260724 \
  1440
```

Result: `fhe_fides_real_d768_t2_attention_sigmoid13_a100_asymfix2_20260724.json` —
`[V]` pass, `rel_inf=9.91e-9`, `709.3 s [gpu]` versus `1347.1 s` for the original
exp-plus-reciprocal schedule, packed at level 22 of the 43-level chain. The T=2
sigmoid score-difference identity (`w1=sigmoid(s11-s10)`) replaces the schedule that
exhausted depth 43 in `fhe_fides_real_d768_t2_block0_depth43_FAIL_20260724.json`.
The harness's own depth guards confirm 21 levels remain against an 8-level MLP
requirement plus a 1-level final pack — the full-block (`gate=full`) retry with this
schedule is the next run, not yet executed.

### Complete block-0 gate, sigmoid schedule

Fail-closed scheduled launch on the Brev GPU host (same script and fixture as the
sigmoid attention sub-gate above, `--gate full`):

```bash
fhe/gpu_real_sigmoid/schedule_when_free.sh \
  0,1,2,3,4,5,6,7 \
  /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  gpu_real_sigmoid_v1/fhe/gpu_real_sigmoid \
  real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0 \
  dnagpt-fideslib:786c-asymfix2 \
  full \
  fhe_fides_real_d768_t2_block0_sigmoid13_a100_asymfix2_20260724 \
  1440
```

Result: `fhe_fides_real_d768_t2_block0_sigmoid13_a100_asymfix2_20260724.json` — `[V]`
pass. The complete released block 0 (LayerNorm-1, 12-head T=2 sigmoid-identity
attention, projection, residual, LayerNorm-2, the full 3072-wide GELU MLP, and output
packing) closes in one encrypted lineage, packing at level 41 of the 43-level chain
with `rel_inf=6.41e-6`, `2461.8 s [gpu]`, one final decrypt. This resolves
`fhe_fides_real_d768_t2_block0_depth43_FAIL_20260724.json`, which exhausted the same
depth-43 chain using the exp-plus-reciprocal attention schedule. Only 2 levels remain
unconsumed, so the native refresh (below) must run immediately after this block in any
chained two-block gate.

### Two-block same-lineage gate (fails closed)

Fail-closed scheduled launch on the Brev GPU host:

```bash
fhe/gpu_multiblock/schedule_multiblock_when_free.sh \
  0,1,2,3,4,5,6,7 \
  /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  gpu_multiblock_v1 \
  multiblock_fixture_v1 \
  dnagpt-fideslib:786c-asymfix2 \
  fhe_fides_gpu_multiblock_blocks0_1_refresh_a100_asymfix2_20260724 \
  1440
```

Result: `fhe_fides_gpu_multiblock_blocks0_1_refresh_FAIL_20260724.json` — `[V]` FAIL.
The module crashed (`SIGSEGV`, exit `139`) strictly between the CUDA device banner and
its own first log line.

Root-caused via a sequence of diagnostics, all uncommitted local edits reverted after
each test: (1) per-call setup logging (`std::unitbuf` plus a print after each of
`GenCryptoContext`/`Enable`/`EvalBootstrapSetup`/`KeyGen`/`EvalMultKeyGen`/
`EvalRotateKeyGen`/`EvalBootstrapKeyGen`/`LoadContext`/`Synchronize`) localized the
crash to inside `LoadContext`, reproducibly; (2) `compute-sanitizer --tool memcheck`
reported zero device-memory errors, ruling out an illegal GPU access; (3) lowering
`multiplicative_depth` to 50 (`large_digits=4` unchanged) got much further into
`LoadContext` and failed with an explicit `Cuda failure ... 'out of memory'` at
`81131/81920 MiB` — see `fhe_fides_gpu_multiblock_bisect_depth50_FAIL_20260724.json`;
(4) lowering `large_digits` to 2 instead was an independent dead end — it crashed at
the same early point as the original failure at *both* depth 64 and depth 50, i.e.
`large_digits=2` is itself broken/incompatible with this rotation-key configuration,
unrelated to memory; (5) at `multiplicative_depth=58` (`large_digits=4`), `addr2line`
on the crash backtrace pinpointed the exact call chain —
`AddBootstrapPlaintexts -> Plaintext::load -> RNSPoly::loadConstant ->
LimbPartition::generateLimbConstant -> Limb -> VectorGPU -> GPUmalloc` — the same
bootstrap-precomputation-plaintext-loading path that step (3) reached and failed in
cleanly. See
`fhe_fides_gpu_multiblock_bisect_depth58_backtrace_FAIL_20260724.json`.

This supersedes the original record's "GPU memory pressure, unconfirmed" hypothesis:
the depth=64 run's ~2GB peak-usage measurement was real but misleading, since it
crashes before reaching the expensive allocations at all. The confirmed root cause is
that this context's memory footprint (`batch_slots=4096`, 63 rotation keys,
`large_digits=4` HYBRID key switching) does not fit in one A100's 80GB regardless of
depth in the 50-64 range or digit count; tuning `multiplicative_depth` or
`large_digits` alone does not fix it. Plain `strace`/`gdb` was not viable on this
binary (`ptrace` fights CUDA's driver-internal signal handling and produces a runaway
`SIGSEGV` storm); `cuda-gdb` (below) does not have this problem. `[U]` Fitting the
gate on one GPU needs fewer `batch_slots` and/or fewer rotation keys, or multi-GPU
sharding.

**Multi-GPU sharding attempt (2026-07-25):** `--gpu` was extended end-to-end
(CLI parsing, `SetDevices`, the launcher, and the scheduler) to accept a
comma-separated device list and reserve/launch on N GPUs atomically. The first
real run on 2 physical A100s (`fhe_fides_gpu_multiblock_multigpu_2gpu_sigsegv_setupconstants_FAIL_20260725.json`)
got further than every single-GPU attempt — both device banners printed, i.e.
per-device context enumeration succeeded — then crashed with a **different**
`SIGSEGV`, confirmed via a clean `cuda-gdb --batch -ex run -ex 'thread apply all
bt'` backtrace: `main -> LoadContext -> GenCryptoContextGPU ->
ContextData::ContextData(devices) -> FIDESlib::SetupConstants -> cudaMemcpy ->
cuMemcpyHtoD_v2`, at only ~8.7 GiB/GPU (not OOM, and not the
`AddBootstrapPlaintexts` path above). This matches the standing risk flag that
FIDESlib's own examples and tests never exercise a real multi-GPU device list —
`SetupConstants`'s multi-device construction path itself appears to have a
genuine, previously-unexercised bug. `[U]` Multi-GPU sharding is not usable as
a lever until this is root-caused (needs a debug FIDESlib build or
`compute-sanitizer`) or worked around upstream; reducing `batch_slots`/rotation
keys on a single GPU remains the more tractable path for now.

**Docker CLI note:** Docker 29.3.1 / nvidia-container-toolkit 1.18.1 rejects a
bare `--gpus device=5,6` with `"cannot set both Count and DeviceIDs on device
request"`; it requires the value nested in a literal extra quote pair
(`--gpus '"device=5,6"'`) so Docker's own CSV parser treats the whole
comma-separated list as one value. `launch_brev_multiblock.sh` encodes this.

### Native GPU bootstrap refresh gate

Run on the Brev GPU host as a fail-closed scheduled launch (waits for a GPU to be
process-free for two consecutive 30 s polls, reserves it, then runs):

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

Result: `fhe_fides_refresh_d768_t2_native_a100_asymfix2_20260724.json` — `[V]` pass,
`rel_inf=1.045e-3` against the `4e-2` gate, restores 21 levels through a native CUDA
bootstrap plus a post-refresh nonlinear tail (square + LayerNorm epsilon), one final
decrypt, on the same fixed released block-0 activation fixture used by the LayerNorm
and attention gates above. The container exits `139` (`free(): invalid pointer`)
strictly after the decrypt and JSON write, during FIDESlib/CUDA teardown; this is a
recorded shutdown defect, not a correctness issue, and does not block the result.

### Twelve-block oracle and fixed nonlinear preflight

```bash
source .venv/bin/activate
python -m fhe.multiblock.export_fixture
python -m fhe.multiblock.validate_fixture \
  checkpoints/fhe_exports/gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0
python -m fhe.range_control.simulate
```

Evidence: `fhe_multiblock_plaintext_contract_20260724.json` and
`fhe_range_control_t2_12block_optimized_v2_20260724.json`. The latter is a
plaintext approximation preflight on one public T=2 fixture, not an encrypted
or task-representative run.
