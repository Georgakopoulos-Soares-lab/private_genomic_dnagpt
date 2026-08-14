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

**Verdict** `[V]`: the released head and local pipeline produce the expected aggregate behavior on
the full balanced 22,604-example source corpus. Because this all-example corpus includes records
used to train the released head, the result is model/pipeline-fidelity evidence rather than an
independent held-out generalization estimate.

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

## Phase B — encrypted feasibility

FHE evidence is split by architecture (see `CLAUDE.md` and
[shared/architecture_options.md](shared/architecture_options.md) for the full
comparison and decision rationale):

- **Pure non-interactive CKKS (legacy internal Scheme A, frozen baseline):** operator matrix, the toy
  block, CUDA gates, the sigmoid-schedule fix, and the chained-composition failures
  that forced the pivot — see [pure/tasks.md](pure/tasks.md).
- **Client-assisted CKKS (legacy internal Scheme B, active):** exact client nonlinearities, general
  causal attention, eight-token SIMD packing, a complete real-weight `D=768`/`T=103` block, short
  two-block composition, and the current optimization history — see [hybrid/tasks.md](hybrid/tasks.md)
  and [hybrid/roadmap.md](hybrid/roadmap.md).

Both share the same Phase-A acceptance rule. The complete 12-block plus GSR-head driver has executed
once at task length on a clean node and matched the recorded class label; it is a single selected
prompt and a single run, not encrypted task-set accuracy or a stable latency estimate. The remaining
execution order is in [roadmap.md](roadmap.md).
