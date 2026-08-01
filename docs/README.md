# Documentation map

Each current decision has one owner. Detailed experiment histories remain append-only, but they do
not own current status.

| Concern | Canonical document |
|---|---|
| Project purpose and strongest current result | [overview.md](overview.md) |
| Cross-project execution order and acceptance rules | [roadmap.md](roadmap.md) |
| Plaintext evaluation design | [eval_approach.md](eval_approach.md) |
| Plaintext commands and results | [tasks.md](tasks.md) |
| Dataset sources, recovery, and licenses | [data_provenance.md](data_provenance.md) |
| Active client-assisted CKKS optimization plan | [hybrid/roadmap.md](hybrid/roadmap.md) |
| Detailed client-assisted experiment history | [hybrid/tasks.md](hybrid/tasks.md) |
| Optimization successes, failures, and remaining gaps | [hybrid/optimizations_and_combinations_report.md](hybrid/optimizations_and_combinations_report.md) |
| Pure non-interactive CKKS baseline | [pure/](pure/) |
| Architecture and backend decisions | [shared/](shared/) |
| Paper-oriented narrative without development bookkeeping | [paper/](paper/) |

Raw accepted evidence lives under [`../results/`](../results/). The project charter is
[`../CLAUDE.md`](../CLAUDE.md).

## Documentation rules

- Update `overview.md` when the strongest defensible result changes.
- Update the relevant roadmap when priorities or gates change.
- Append exact commands and result interpretation to the relevant `tasks.md`.
- Keep paper notes free of internal scheme letters, run tags, hashes, fixture names, and host-specific
  orchestration details.
- Preserve useful negative results as mechanisms and consequences; remove superseded status snapshots
  rather than maintaining parallel summaries.
