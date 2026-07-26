# Documentation map

| File | Owns |
|---|---|
| [overview.md](overview.md) | What the project is, current status table |
| [eval_approach.md](eval_approach.md) | Evaluation methodology & design: why these metrics, harness design, oracle contract, limitations |
| [tasks.md](tasks.md) | Per-task exact commands, measured results, verdicts (Phase A) + pointers into `pure/`/`hybrid/` (Phase B) |
| [data_provenance.md](data_provenance.md) | Every dataset's source, retrieval, recovery, license |
| [roadmap.md](roadmap.md) | Shared FHE acceptance contract and rules + pointers into `pure/`/`hybrid/` for scheme-specific plans |

FHE (Phase B) content is split by architecture — see `CLAUDE.md` for why hybrid
client-assisted CKKS (Scheme B) is the default and pure non-interactive CKKS (Scheme A)
is the frozen ablation:

| Folder | Owns |
|---|---|
| [shared/](shared/) | Content that applies to both schemes: backend selection (CKKS vs. Concrete ML/TenSEAL), the Scheme A/B/C architecture comparison and decision, the Phase-B evidence-progression overview |
| [pure/](pure/) | Scheme A (frozen): operator matrix, toy/real-width block measurements, the Brev CUDA runbook, roadmap and task history up to the pivot |
| [hybrid/](hybrid/) | Scheme B (active): the toy prototype and real-weight gates, active roadmap and task evidence |

Raw evidence lives in [`../results/`](../results/) (`pure/`, `hybrid/`, `shared/`
`manifest.yaml` + `runs/`). The charter is [`../CLAUDE.md`](../CLAUDE.md). Do not
duplicate status narratives across files — update the owner.
