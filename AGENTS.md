# AGENTS.md

DNAGPT encrypted-inference feasibility project. The full charter and safety rules live in
[CLAUDE.md](CLAUDE.md).

## Read first

1. [CLAUDE.md](CLAUDE.md) — objective, scope, evidence rules, and repository layout.
2. [docs/overview.md](docs/overview.md) — current result and unresolved boundary.
3. [docs/roadmap.md](docs/roadmap.md) — cross-project execution order.
4. [docs/hybrid/roadmap.md](docs/hybrid/roadmap.md) — active client-assisted CKKS optimization plan.

Writing the paper? Go to [paper-docs/AGENTS.md](paper-docs/AGENTS.md) instead — it overrides this
file for anything under `paper-docs/`, and it is where the terminology and claim rules live.

Do not reconstruct current status from filenames or old run tags. The roadmaps own current decisions;
`docs/{pure,hybrid}/tasks.md` and `results/` preserve the detailed experiment history.

## Current state

- `[V]` Three plaintext DNAGPT tasks pass and supply frozen numerical oracles.
- `[V]` Pure non-interactive CKKS closes one real-weight block but is frozen after chained
  composition hit a root-caused A100 memory wall.
- `[V]` Client-assisted CKKS closes one complete real-weight block at the 103-token GSR length with
  eight-token SIMD packing, depth 13, about `4e-9` relative error, and about `9.8 GiB` peak
  device-wide GPU memory used during the run.
- `[V]` Two released blocks compose at two tokens using a declared client refresh.
- `[V]` All 12 released blocks plus the GSR head pass at 103 tokens in one whole-node execution
  with no contention detected in recorded telemetry: `6683 s` (`1.86 h`), correct label, no
  homomorphic bootstrap, and `9839 MiB` peak device-wide GPU memory used during the run.
- `[U]` The complete execution has not been repeated, and only one prompt has been evaluated.
  There is no encrypted task-set accuracy or networked client/server measurement.

The next performance step is **not** another unprofiled complete-model run. First obtain a paired
dedicated one-block baseline and current T=103 CPU/CUDA profile, then evaluate the exact-model changes
ordered in `docs/hybrid/roadmap.md`. Repeat the complete execution only to establish variance or to
validate retained changes; treat the existing run as a single-sample feasibility result.

## Terminology

Use descriptive protocol names in prose:

- **pure non-interactive CKKS** for the frozen baseline;
- **client-assisted CKKS** or **client-assisted hybrid CKKS** for the active path.

`Scheme A` and `Scheme B` are legacy implementation/result namespaces. Use them only when a literal
path, target, or historical result requires the internal name. Narrative and paper documentation must
not depend on tags, hashes, fixture names, host names, or other development bookkeeping.

## Evidence and editing rules

- Python 3.12 in `.venv`; Phase A uses PyTorch MPS/CPU float32. Run commands from the repository root.
- Never commit weights or datasets. `checkpoints/` and `data/` are re-downloadable and gitignored.
- Every dataset change updates `docs/data_provenance.md`.
- Every accepted result is immutable: add `results/runs/<tag>.json` and the appropriate
  `results/{pure,hybrid,shared}/manifest.yaml` row. Never overwrite evidence.
- Put exact Phase-A reproduction commands in `docs/tasks.md`; put architecture-specific commands in
  `docs/pure/tasks.md` or `docs/hybrid/tasks.md`.
- Tag load-bearing claims `[V]` measured/verified, `[U]` unresolved, or `[A]` assumed/derived, and cite
  the owning source.
- Preserve useful negative results, but summarize their mechanism and consequence in canonical docs.
  Do not keep parallel status snapshots or raw handoff ledgers.
- Read before editing, keep diffs reviewable, and do not commit, push, deploy, or start a remote GPU job
  unless asked.

## Common commands

```bash
source .venv/bin/activate
python eval/eval_gsr.py --limit -1 --tag gsr_aataaa_human_full
python eval/build_mrna_testset.py
python eval/eval_mrna.py --tag mrna_xpresso_human_1ktest
python eval/finetune_gue.py --data data/gue/GUE/prom/prom_300_all --tag gue_prom_300_all --max_len 64
PYTHONPATH=. .venv/bin/python -m unittest discover -s fhe/gpu_real_scheme_b -p "test_*.py"
```

GPU build and launch commands are versioned in [docs/hybrid/brev_runbook.md](docs/hybrid/brev_runbook.md)
and the corresponding evidence entries in `docs/hybrid/tasks.md`.
