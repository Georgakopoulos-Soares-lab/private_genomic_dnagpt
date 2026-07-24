# Evidence contract

`results/` is the immutable evidence store. It is the only place performance numbers are considered
real, and it is what the paper cites.

## Rules

- **One run = one file.** Each evaluation writes `runs/<tag>.json` (metrics + config + provenance)
  and, where per-example outputs matter, `runs/<tag>_preds.csv`. These are append-only — never edit
  or overwrite a completed run; a re-run gets a new `<tag>`.
- **Every run has a manifest row** in `manifest.yaml`: task, weight, dataset+source, device, n,
  metrics, reference, and a `[V]`/`[U]` verdict.
- **A number without a run file does not exist.** Do not quote metrics from chat or logs.
- Per-example prediction CSVs are the frozen **oracle** for later encrypted-inference acceptance.

## Current runs

| tag | task | headline |
|---|---|---|
| `gsr_aataaa_human_full` | GSR | acc 0.9124, F1 0.916 (n=22,604) |
| `mrna_xpresso_human_1ktest` | mRNA regression | r² 0.562, Pearson 0.753 (n=1,000) |
| `mrna_smoke_regression_sh` | mRNA pipeline check | 2 author examples within ~0.04 |
| `gue_prom_core_all` | GUE core promoter | MCC 0.680, acc 0.840 |
| `gue_prom_300_all` | GUE promoter 300bp | MCC 0.897, acc 0.948 |
| `gue_splice_reconstructed` | GUE splice (3-class) | MCC 0.831, acc 0.895 |
