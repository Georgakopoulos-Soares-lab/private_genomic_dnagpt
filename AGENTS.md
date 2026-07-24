# AGENTS.md

DNAGPT local performance-baseline project. Full charter: **[CLAUDE.md](CLAUDE.md)**.

## TL;DR

Goal: evaluate whether **DNAGPT** (arXiv 2307.05628) inference can run under **FHE** (encrypted
operations) — the DNAGPT counterpart of `../evo2`. **Phase A (current):** measure DNAGPT locally on 3
tasks (GSR, GUE, mRNA regression) to produce the trustworthy **plaintext oracle** the encrypted path
must later reproduce. **Phase B:** encrypted operators on the 0.1b backbone. Evidence discipline
mirrors `../evo2`.

## Conventions

- Python 3.12 in `.venv`; PyTorch on MPS/CPU, float32. Run from repo root.
- Never commit weights or datasets (`checkpoints/`, `data/` are gitignored). They are re-downloadable.
- Every dataset → provenance in `docs/data_provenance.md`. Every result → immutable
  `results/runs/<tag>.json` + a `results/manifest.yaml` row. Never overwrite an old run.
- Tag load-bearing claims `[V]` measured / `[U]` blocked / `[A]` assumption, with a source.
- One exact reproduction command per reported number lives in `docs/tasks.md`.

## Common commands

```bash
source .venv/bin/activate
python eval/eval_gsr.py  --limit -1 --tag gsr_aataaa_human_full          # Task 1
python eval/build_mrna_testset.py                                        # Task 3 data
python eval/eval_mrna.py --tag mrna_xpresso_human_1ktest                 # Task 3
python eval/finetune_gue.py --data data/gue/GUE/prom/prom_300_all --tag gue_prom_300_all --max_len 64  # Task 2
```

## Where things are

Code `eval/` · upstream `DNAGPT/` · evidence `results/` · docs `docs/` · charter `CLAUDE.md`.
