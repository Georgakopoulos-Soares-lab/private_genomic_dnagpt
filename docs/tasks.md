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
