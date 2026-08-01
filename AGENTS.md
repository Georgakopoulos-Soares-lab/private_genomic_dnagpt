# AGENTS.md

DNAGPT encrypted-inference feasibility project. The full charter and safety rules live in
[CLAUDE.md](CLAUDE.md).

## Read first

1. [CLAUDE.md](CLAUDE.md) — objective, scope, evidence rules, and repository layout.
2. [docs/overview.md](docs/overview.md) — current result and unresolved boundary.
3. [docs/roadmap.md](docs/roadmap.md) — cross-project execution order.
4. [docs/hybrid/roadmap.md](docs/hybrid/roadmap.md) — active client-assisted CKKS optimization plan.

Do not reconstruct current status from filenames or old run tags. The roadmaps own current decisions;
`docs/{pure,hybrid}/tasks.md` and `results/` preserve the detailed experiment history.

## Current state

- `[V]` Three plaintext DNAGPT tasks pass and supply frozen numerical oracles.
- `[V]` Pure non-interactive CKKS closes one real-weight block but is frozen after chained
  composition hit a root-caused A100 memory wall.
- `[V]` Client-assisted CKKS closes one complete real-weight block at the 103-token GSR length with
  eight-token SIMD packing, depth 13, about `4e-9` relative error, and about `9.8 GiB` process peak
  GPU memory.
- `[V]` Two released blocks compose at two tokens using a declared client refresh.
- `[U]` The built 12-block plus GSR-head driver has not run at 103 tokens.
- `[U]` Existing long timings are contaminated by shared-host load. There is no clean end-to-end
  latency result or networked client/server measurement.

The next performance step is **not** an unprofiled multi-day run. First obtain a dedicated one-block
baseline and current T=103 CPU/CUDA profile, then evaluate the exact-model changes ordered in
`docs/hybrid/roadmap.md`. Run the existing 12-block driver immediately only when the goal is arithmetic
correctness closure, and label it a baseline feasibility run.

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
