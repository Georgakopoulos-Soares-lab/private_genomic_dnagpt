# Roadmap

## Phase A — plaintext baseline (current)

Goal: all three tasks measured locally with full provenance = the FHE acceptance oracles.

- [x] Task 1 GSR — **[V] PASS** (acc 0.9124).
- [x] Task 3 mRNA regression — **[V] PASS** (r² 0.562, Pearson 0.753).
- [x] Task 2 GUE — **[V] PASS** (MCC 0.680 / 0.897 / 0.831 core/300/splice).

**Phase A complete.** All verdicts in `docs/tasks.md`, each backed by an immutable run + manifest row.
The per-example `results/runs/*_preds.csv` are the frozen oracle for Phase B.

## Phase B — strengthen the baseline (optional)

- Verify the exact Xpresso 1,000-gene test identity (close the r² gap to ~0.62) if it matters for
  the oracle tolerance.
- Extend GUE beyond the 3 named datasets toward the full 28 if broader coverage is wanted (GPU/Brev).
- Add `dna_gpt3b_m` comparisons (GPU/Brev).

## Phase C — toward FHE (the reason for the baseline)

- Freeze each plaintext prediction (per-example outputs already in `results/runs/*_preds.csv`) as the
  oracle a future encrypted DNAGPT run must match within a declared tolerance — mirroring evo2's
  oracle→encrypted acceptance contract.
- Pick the first FHE target op/subgraph on the 0.1b backbone; the lightest task (GSR, 600 bp,
  single-token readout) is the natural first encrypted end-to-end candidate.

## Standing constraints

Mac cannot produce GPU-performance evidence; heavy runs move to Brev. Never commit weights/data.
Every new number = new run tag + manifest row.
