# Source-audit closure — 2026-08-14

## Result: pass

All three findings from `2026-08-14-paper-sources-revision.md` are resolved:

- `manuscript/source/sections/03_scenario_threat_model.tex:116-119` now says that per-token
  embeddings remain sensitive because high-fidelity reconstruction has been demonstrated for
  embeddings from *other* DNA foundation models. It no longer claims equivalence to the raw
  sequence or direct evidence for DNAGPT embeddings.
- `manuscript/source/sections/02_background.tex:73-76` now uses the Concrete ML and TFHE-rs
  citations to identify the evaluated software paths, then explicitly attributes the suitability
  conclusion to this project's backend comparison.
- `manuscript/source/refs.bib:281` now records `Lam, Kwok-Yan`, matching the ICLR 2026 MOAI
  record.

The final style pass did not introduce a new unsupported citation relationship. The THOR and MOAI
comparisons remain explicitly uncontrolled; the GSR source-corpus qualification remains intact; and
THE-X, Powerformer, ELLMo, DNA-embedding inversion, and Li--Micciancio retain the scopes verified in
the preceding review.

Mechanical closure:

- 30 cited keys and 30 bibliography entries;
- no missing citation keys;
- no unused bibliography entries;
- `paper-docs/scripts/build.sh lint`: all four checks clean, including citation lint.

No source or citation item remains open. The existing `main.blg` on disk still contains the old
pre-fix MOAI sorting warning, so it is a stale build artifact; `refs.bib` now has a complete author
field, and the next full bibliography rebuild should remove that warning.
