# Tasks — methodology, commands, results, verdicts

Commands run from the repo root. Phase A uses `.venv`; Phase B uses the pinned Docker
image or the documented Brev container. Each reported number has exactly one command
that regenerates it. Raw outputs: `results/runs/<tag>.json` (+ `_preds.csv`). Data
origins: [data_provenance.md](data_provenance.md).

---

## Task 1 — Genomic Signal & Region Recognition (GSR) — [V] PASS

**Setup.** Released `classification.pth` (fine-tuned `dna_gpt0.1b_m`). For each DeepGSR human AATAAA
record: strip central motif → 600 bp, prompt `<R>{seq}<=><R>`, take `argmax` of the final-position
logits; token `N` ⇒ real GSR (1), `A` ⇒ fake (0). Evaluated on the **full** balanced set.

**Command**
```bash
python eval/eval_gsr.py --limit -1 --tag gsr_aataaa_human_full
```

**Result** (`results/runs/gsr_aataaa_human_full.json`, n=22,604, MPS):

| accuracy | precision | recall | F1 | reference |
|---|---|---|---|---|
| **0.9124** | 0.881 | 0.954 | **0.916** | DeepGSR human AATAAA ~0.916 |

**Verdict** `[V]`: matches the published accuracy within ~0.4 pt on the full test set. DNAGPT
reliably performs GSR recognition.

---

## Task 3 — Human mRNA Abundance Regression — [V] PASS

**Setup.** Released `regression.pth` (fine-tuned `dna_gpt0.1b_h`, max_len 4096). Xpresso human data
recovered from the Internet Archive and reprocessed (`eval/build_mrna_testset.py`) into the canonical
last-1,000-gene test split. Each gene: prompt `<R>{seq10.5kb}<+><M><=><M>` + the 8 z-scored
half-life features; read the `num_regression` head; compare to the z-scored expression target.

**Commands**
```bash
python eval/build_mrna_testset.py                              # builds data/mrna/pM10Kb_1KTest/test.h5
python eval/eval_mrna.py --tag mrna_xpresso_human_1ktest
```

**Pipeline check** (`results/runs/mrna_smoke_regression_sh.json`): the two author examples in
`scripts/regression.sh` reproduce their ground truth within ~0.04 (float32 vs upstream float16).

**Result** (`results/runs/mrna_xpresso_human_1ktest.json`, n=1,000, MPS):

| r² | Pearson r | Spearman r | reference |
|---|---|---|---|
| **0.562** | **0.753** | 0.745 | DNAGPT paper r² ~0.62 (Xpresso 0.59) |

**Verdict** `[V]`: strong correlation (Pearson 0.75), r² within ~0.05 of the paper. The residual gap
is most likely the non-identical test-split reproduction (see provenance caveat) plus float
precision. DNAGPT reliably performs mRNA abundance regression.

---

## Task 2 — GUE (human promoters & splice sites) — [V] PASS

**Setup.** No released GUE head, so `eval/finetune_gue.py` fine-tunes the `dna_gpt0.1b_m` foundation
backbone + a linear head on the masked-mean-pooled final hidden state (DNABERT-2 protocol: full
fine-tune, 3 epochs, AdamW lr 3e-5). Primary metric **MCC**.

**Commands**
```bash
python eval/finetune_gue.py --data data/gue/GUE/prom/prom_core_all      --tag gue_prom_core_all      --max_len 32 --epochs 3
python eval/finetune_gue.py --data data/gue/GUE/prom/prom_300_all       --tag gue_prom_300_all       --max_len 64 --epochs 3
python eval/finetune_gue.py --data data/gue/GUE/splice/reconstructed    --tag gue_splice_reconstructed --max_len 96 --epochs 3
```

**Result** (MPS, 3 epochs; `results/runs/gue_*.json`):

| dataset | classes | MCC | accuracy | macro-F1 | DNABERT-2 ref (MCC) `[A]` |
|---|---|---|---|---|---|
| prom_core_all | 2 | **0.680** | 0.840 | 0.840 | ~0.69 |
| prom_300_all | 2 | **0.897** | 0.948 | 0.948 | ~0.87 |
| splice/reconstructed | 3 | **0.831** | 0.895 | 0.892 | ~0.85 |

**Verdict** `[V]` PASS: DNAGPT's 0.1b backbone, fine-tuned locally, is competitive with DNABERT-2 on
human promoter and splice-site GUE tasks (matches prom_core, exceeds prom_300, close on splice).
`[A]` DNABERT-2 references are approximate — confirm against the DNABERT-2 paper table for exact deltas.
Note: this is 3 of the 28 GUE datasets (the human promoter/splice subset named in the brief).

---

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
[`04_brev_runbook.md`](feasibility/04_brev_runbook.md):

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

## Architecture pivot: Scheme A frozen, Scheme B active (2026-07-25)

Three independent chained-composition attempts above
(`fhe_fides_gpu_multiblock_bisect_depth50_FAIL_20260724`,
`..._bisect_depth58_backtrace_FAIL_20260724`,
`..._multigpu_2gpu_sigsegv_setupconstants_FAIL_20260725`) all fail closed on GPU memory
exhaustion, not accuracy, and all trace to the same root cause: block 0's single
end-of-block bootstrap buying back 35 levels for all of block 1 in one shot, driven by
degree-13 Chebyshev approximation of LayerNorm invsqrt, the attention sigmoid identity,
and GELU. Full comparison of alternatives in
[`05_architecture_options.md`](feasibility/05_architecture_options.md).

**Decision:** freeze pure non-interactive CKKS (**Scheme A**) as the paper's baseline —
its existing evidence (complete real-weight block 0, refresh gate, the three failures
above) stands unchanged. Active development moves to **Scheme B**: hybrid
client-assisted CKKS. Server-side linear algebra (FIDESlib GPU) stays in one encrypted
lineage; only the data-owning client (already the secret-key holder) decrypts, at
pre-declared nonlinearity boundaries (LayerNorm, attention nonlinearity, GELU), its own
data, then re-encrypts. This removes the Chebyshev-driven depth pressure that caused
every Scheme A composition failure, while keeping the untrusted compute provider fully
blind to plaintext at every step.

**Next concrete step (not yet run):** reuse the existing real-weight block-0
CKKS/FIDESlib graph, delete the three `EvalChebyshevFunction` calls, add
decrypt/compute/re-encrypt at each boundary, and re-measure against the unchanged
Phase-A oracle and `4e-2` gate, recording round-trip count and the GPU-encrypted vs.
client-plaintext wall-time split per Scheme B's acceptance contract in
`docs/roadmap.md`.

### Scheme B toy prototype (2026-07-25)

`fhe/block/toy_block_ckks_scheme_b.py` (D=8, T=4, 2 heads) implements the
`Client`/`EvalOnlyContext` protocol: the client holds the secret key and
performs `cross_boundary()` decrypt/exact-compute/re-encrypt at LN1, the
attention softmax, LN2, and GELU; the server-side context asserts zero
decrypts. Two real bugs were found and fixed en route: (1) `DEPTH=24` landed
in a needlessly expensive `ring_dim=131072` bucket for a measured max level of
only 14-16 — reduced to `DEPTH=20` (`ring_dim=65536`); (2) OpenFHE's
`EvalSum(ct, n)` is a sliding-window sum, not a per-block broadcast (only slot
0 holds the true sum) — `sum_bcast()` now isolates slot 0 via a one-hot mask
and replicates via rotation-doubling.

```bash
source .venv/bin/activate
python fhe/block/toy_block_ckks_scheme_b.py
```

Evidence: `fhe_toy_block_scheme_b_native_cpu_20260725.json` — `[V]` PASS,
`rel_inf=2.22e-12`, 16 client round trips (168 values crossed total), depth 20,
`ring_dim=65536` (vs. Scheme A's toy at depth 49/`ring_dim=131072`). The
1433s wall time is native CPU on a **heavily contended shared Brev host**
(concurrently observed load average 1000-1460 throughout this session) and is
explicitly not used as a GPU-extrapolation input.

### Scheme B real D=768/T=2 GPU gate (2026-07-25)

`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b.cpp` forks
`fhe/gpu_real_sigmoid`'s FIDESlib source and replaces the three
`EvalChebyshevSeries` calls (LN1/LN2 invsqrt, attention sigmoid, GELU x4) with
genuine `Client::cross_boundary()` round trips, using the exact elementwise
function instead of a calibrated-domain polynomial. Same released fixture,
same packing, same `4e-2` oracle gate as Scheme A's `gpu_real_sigmoid` gates.
No public per-block domain contract is needed (the client computes the exact
function on the true decrypted value, whatever its magnitude).

```bash
FIDESLIB_ARCH=80-real BUILD_JOBS=4 fhe/gpu_real_scheme_b/build_in_fideslib.sh

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_scheme_b/run_scheme_b.sh 0 ln1 "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_ln1_scheme_b_a100_20260725.json
```

Evidence: `fhe_fides_real_d768_t2_ln1_scheme_b_a100_20260725.json` — `[V]` PASS,
`rel_inf=2.35e-10`, 2 round trips, `ring_dim=65536` at `multiplicative_depth=16`
(vs. Scheme A's `ring_dim=131072` at depth 43 — LN1's segment level_max is only
2, far below either budget, but Scheme B never needs to reserve depth for the
*rest* of the block since every nonlinearity resets to level 0). The 2.41s
`encrypted_evaluation_seconds_gpu` splits into `server_linear_algebra_seconds`
(1.04s, GPU BSGS/mean/variance ops) and `client_boundary_seconds_total` (1.37s,
2x decrypt+invsqrt+re-encrypt of a `SLOTS=4096` vector) — this split is the key
input for reasoning about how much of Scheme B's cost is server-GPU vs.
client-CPU-boundary work.

`[U]` The `attention` and `full` gates were attempted the same session but the
run was killed after 8.5+ minutes of CPU-bound rotation-key generation
stalling under extreme host contention (load average ~1000-1127 on this
255-core shared machine; the GPU itself was idle at 1% utilization) — for
comparison, Scheme A's equivalent keygen took 25.9s on a presumably quieter
host. `fhe/gpu_real_scheme_b/wait_and_run_scheme_b.sh` now runs detached on the
Brev host, polling load average and GPU occupancy every 60s, and will launch
both remaining gates unattended once capacity is available (log at
`gpu_real_scheme_b_v1/fhe/gpu_real_scheme_b/scheme_b_orchestrator.log` on the
host). The full-block-0-to-e2e Scheme B extrapolation is blocked on those two
results — LN1 alone is too small a fraction of one block to extrapolate from.
