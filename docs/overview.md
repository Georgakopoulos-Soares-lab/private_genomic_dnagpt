# Overview

## Purpose

Evaluate whether **DNAGPT** (arXiv 2307.05628) inference can run under **FHE** (encrypted operations
on an encrypted genome, compute-provider side) — the DNAGPT counterpart of `../evo2`.

**Phase A (this repo's current work):** measure DNAGPT locally on three downstream tasks, confirm it
is a good, reproducible model, and freeze its per-example predictions as the **plaintext oracle** the
encrypted path must later reproduce. **Phase B:** encrypted operators on the 0.1b backbone (see
[roadmap.md](roadmap.md)).

## What DNAGPT ships

Inference only: `test.py` + `dna_gpt/`. Released fine-tuned heads (Google Drive, 0.1b):
`classification.pth` (GSR, on `dna_gpt0.1b_m`) and `regression.pth` (mRNA, on `dna_gpt0.1b_h`,
max_len 4096). No datasets, no fine-tuning code, no GUE head — we supply those.

## Status

| # | Task | Verdict | Result (local) | Reference |
|---|------|---------|----------------|-----------|
| 1 | GSR — human AATAAA PAS | **[V] PASS** | acc **0.9124**, F1 **0.916** (n=22,604) | DeepGSR ~0.916 |
| 3 | mRNA abundance regression | **[V] PASS** | r² **0.562**, Pearson **0.753**, Spearman 0.745 (n=1,000) | DNAGPT paper r² ~0.62 |
| 2 | GUE — promoters & splice sites | **[V] PASS** | MCC prom_core **0.680** / prom_300 **0.897** / splice **0.831** | DNABERT-2 ~0.69/0.87/0.85 |

**Phase A complete: all three tasks pass locally.** Per-example predictions in `results/runs/*_preds.csv`
are the frozen oracle for the encrypted (FHE) path.

See [tasks.md](tasks.md) for methods/commands and [data_provenance.md](data_provenance.md) for sources.

## Environment

Mac (darwin arm64), Python 3.12, torch 2.13 (MPS), float32. 0.1b models run comfortably on MPS
(GSR ~6 ms/seq, mRNA ~67 ms/seq at 10.5 kb, GUE fine-tune ~minutes/epoch).

## Key provenance note

The Xpresso mRNA dataset host (`krishna.gs.washington.edu`) is **globally offline**; the canonical
human input was recovered from the **Internet Archive** and reprocessed with a faithful Python-3 port
of Xpresso's split (18,377 genes → last 1,000 = test). Details in [data_provenance.md](data_provenance.md).
