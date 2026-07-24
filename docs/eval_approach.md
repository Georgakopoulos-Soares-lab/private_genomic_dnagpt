# Evaluation approach

How Phase A measures DNAGPT, why these choices, and how each number becomes an FHE oracle. Concrete
per-task commands + results live in [tasks.md](tasks.md); dataset origins in
[data_provenance.md](data_provenance.md). This file is the *methodology*.

## Design principles

1. **Reuse the upstream model verbatim.** `eval/common.py` imports the cloned `dna_gpt` package and
   replicates `test.py`'s exact forward paths (`classification`, `regression`). We add only batching,
   metrics, and a Mac-safe default (float32 on MPS/CPU; upstream defaults to float16+CUDA). No change
   to model math → the plaintext numbers are a faithful DNAGPT baseline, valid as an FHE oracle.
2. **Canonical test splits only.** GSR = full DeepGSR human AATAAA set; mRNA = Xpresso last-1,000
   split; GUE = the shipped `test.csv`. No cherry-picking.
3. **One command → one immutable run.** Each eval writes `results/runs/<tag>.json` (metrics + config)
   and, where per-example outputs matter, `<tag>_preds.csv`. Append-only (see `results/README.md`).
4. **Reference-anchored verdicts.** A task passes when DNAGPT's metric is at/near the published
   reference (DeepGSR / DNAGPT paper / DNABERT-2). Gaps are reported, never tuned away.

## Per-task method (why + how)

### Task 1 — GSR (classification head)
- **Model call:** released `classification.pth`; prompt `<R>{seq}<=><R>`; `argmax` of final-token
  logits over the vocab; token `N`→real, `A`→fake. This is exactly upstream `test.py:classification`.
- **Data transform:** DeepGSR 606 bp → strip central AATAAA motif `[300:306]` → 600 bp (DNAGPT format;
  confirmed against upstream `processData.m`).
- **Metric:** accuracy + precision/recall/F1 + confusion matrix. Binary, balanced classes → accuracy
  is meaningful; F1 guards against class skew. Matches how DeepGSR reports.

### Task 3 — mRNA abundance (regression head)
- **Model call:** released `regression.pth`; prompt `<R>{seq}<+><M><=><M>` + 8 z-scored half-life
  features spliced as number embeddings; read the `num_regression` scalar. Exactly upstream
  `test.py:regression`.
- **Data transform:** faithful py3 port of Xpresso `setup_training_files.py`
  (`eval/build_mrna_testset.py`): mask histone/chrY, `log10(x+0.1)`, shuffle `random_state=1`,
  z-score, take last 1,000 as test; promoter sliced to the 10.5 kb window `[3000:13500]`.
- **Metric:** r² (coefficient of determination, comparable to the paper's reported number) plus
  Pearson & Spearman r (rank-robust). r² is the headline; Pearson exposes correlation strength even
  if scale/offset drift.
- **Pipeline check:** the two author examples in `scripts/regression.sh` are reproduced first to prove
  the template/feature wiring before the full run.

### Task 2 — GUE (fine-tuned classifier — built here)
- **Why fine-tune:** DNAGPT ships no GUE head and these are supervised tasks; a foundation model has
  no zero-shot classifier. `eval/finetune_gue.py` adds a linear head on the **masked-mean-pooled**
  final hidden state of the `dna_gpt0.1b_m` backbone.
- **Protocol (DNABERT-2-comparable):** full fine-tune, 3 epochs, AdamW lr 3e-5, batch 32, dropout 0.1;
  per-dataset `max_len` sized to the sequence length (32/64/96 for core/300/splice).
- **Metric:** MCC (GUE primary metric — robust to class imbalance and multi-class) + accuracy + macro-F1.
- **Scope:** the 3 human promoter/splice datasets named in the brief (3 of 28 GUE datasets). Extending
  to all 28 is a documented Phase-B/optional step (GPU/Brev).

## Reproducibility

- Pinned env: `requirements.txt` (torch 2.13, etc.), Python 3.12, `.venv`.
- Determinism: fixed seeds in every harness (`--seed`). MPS kernels are not bit-identical to CPU/CUDA,
  so tiny metric drift across devices is expected and acceptable for a baseline oracle.
- Every headline number in [tasks.md](tasks.md) has a single copy-paste command that regenerates its
  run JSON.

## From baseline to FHE oracle (why this matters for Phase B)

The encrypted (FHE) DNAGPT path in Phase B must reproduce the **plaintext prediction**, not just the
metric. So the per-example `results/runs/*_preds.csv` are the frozen acceptance oracle: for a chosen
task and input, the encrypted output must match the recorded plaintext output within a declared
tolerance (mirroring `../evo2`'s oracle→encrypted contract). The lightest task (GSR: 600 bp,
single-token readout) is the natural first encrypted end-to-end target.

## Known limitations (honest boundaries)

- `[U]` mRNA test-split gene identity vs Xpresso's original 1,000 is unverified (pandas `sample`
  reproducibility); likely source of the r² 0.562 vs paper ~0.62 gap.
- `[A]` DNABERT-2 GUE reference MCCs are approximate — confirm against the paper table for exact deltas.
- `[U]` GUE covers 3/28 datasets by design (the promoter/splice subset).
- Metrics are MPS/float32; not bit-reproducible across hardware.
