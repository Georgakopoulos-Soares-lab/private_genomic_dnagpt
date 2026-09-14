# Paper submission TODO

This file records submission work that cannot be completed from the repository alone. The current
manuscript reports the complete execution at its measured scope and keeps the remaining deployment
questions explicit.

## Submission blockers

### Repeat the complete execution

The twelve-block classifier and released task head completed once on a telemetry-confirmed clean
single A100 allocation. Before making a repeatability or throughput claim, commit multiple
independent executions, the declared
summary statistic, target-process telemetry, and complete environment/configuration records. Keep
older shared-host timings out of the manuscript.

### Conditional: experimental backend comparison

The manuscript now describes Concrete ML and TFHE-rs as a documentation review, not a local
performance comparison, and records access dates. If an experimental comparison is added, record
exact releases, configurations, and measured runs before making comparative performance claims.

## Complete-model and deployment evidence

- Add serialized client/server transport and measure payload size, phase dependencies, bandwidth
  sensitivity, and transport latency.
- Resolve private token-index lookup or keep it outside the claimed encrypted service boundary.
- Select and validate a noise-flooding budget for client re-encryption.

## Author and production tasks

- (2026-09-14) Author e-mail address, corresponding-author designation, institutional affiliations,
  funding/support text, and Acknowledgment are now in `00_frontmatter.tex`/`11_declarations.tex`.
  ORCIDs are still outstanding.
- Complete the multi-author consent process and prepare the cover letter required by the submission
  portal.
- Decide and document the repository reuse license.
- Run `paper-docs/.venv-paper/bin/python paper-docs/scripts/check_numbers.py` after every ledger,
  figure, or prose restoration.

## Resolved in the current draft

- The baseline table now uses DNAGPT's own `0.9151` signal-recognition result and states the split
  mismatch. The incorrectly attributed DeepGSR number is no longer used.
- Host-memory observations are consistently converted from the source MiB values to GiB.
- The long-sequence causal-tile count and fixed-circuit projections have been recomputed.
- All nine figures have been regenerated and restored to the manuscript. The timeline now uses
  the measured complete-run trajectory, and the shorter-task projections use the clean full-run
  anchor. Unsupported client-memory and optimization-speedup claims remain excluded.
- The manuscript now reports the measured complete-classifier result rather than a twelve-block
  projection, while repeat variation and networked transport remain explicitly unresolved.
- The IEEE/JBHI manuscript renders successfully and has been visually inspected page by page.
- The 12-page layout has no empty columns on pages 6–8; algorithms are interleaved with explanatory
  prose, and all figure, table, algorithm, and citation references resolve.
- (2026-09-08) `fig_waterfall` and `fig_cost_split` — two decompositions of the same measured
  `full.encrypted_evaluation` interval, by phase and by party — are merged into one two-panel
  `fig_cost_decomposition` figure in §7. Nine figures become eight; `check_numbers.py`'s structure
  check (dangling labels/refs, orphaned generated figures) is clean after the merge.
