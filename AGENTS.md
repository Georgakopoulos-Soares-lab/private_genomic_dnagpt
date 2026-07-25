# AGENTS.md

DNAGPT local performance-baseline project. Full charter: **[CLAUDE.md](CLAUDE.md)**.

## TL;DR

Goal: evaluate whether **DNAGPT** (arXiv 2307.05628) inference can run under **FHE** so an untrusted
compute provider never sees plaintext DNA. **Phase A is complete:** three local tasks provide the
plaintext oracle. **Phase B is active:** the encrypted toy block and a full real-weight block pass
under pure non-interactive CKKS (**Scheme A**, now frozen as baseline); chained multi-block
composition hit a root-caused GPU memory wall, so the active path is **Scheme B**, hybrid
client-assisted CKKS — server-side linear algebra stays encrypted on GPU, only the data-owning
client (who already holds the secret key) decrypts at pre-declared nonlinearity boundaries. See
`docs/feasibility/05_architecture_options.md`.

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

Code `eval/` and `fhe/` · upstream `DNAGPT/` · evidence `results/` · docs `docs/` · charter `CLAUDE.md`.
