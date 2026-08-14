# Final academic style revision — 2026-08-14

## Outcome

The manuscript received a final style-only pass after the structure, evidence, source, and figure
revisions. The pass improves concision and mechanical consistency without changing the scientific
claim set. It does not add measurements, citations, comparisons, or performance interpretations.

## Edits made

- Shortened the introduction's results synthesis while retaining all reported values and the
  distinction between the evaluated non-interactive and client-assisted configurations.
- Replaced the staged objection-and-rebuttal sequence in the deployment section with a direct
  comparison: local plaintext inference is preferable when model distribution is allowed; the
  evaluated protocol addresses the served-model premise.
- Rewrote the optimization opening to lead with the observed host-memory failure and the retained
  stage-bounded cache, then state the fixed-circuit invariant once.
- Removed redundant transitional language and split long sentences in the results, related-work,
  limitations, and future-work prose.
- Made memory wording mechanically consistent in headline and summary locations: the retained
  sampler reports device-wide GPU memory used, while host values remain target-process RSS.
- Narrowed the complete-model memory synthesis to the evaluated execution and retained the
  whole-node/no-contention-detected qualification.
- Preserved the provider-side timing bucket as residual encrypted-evaluation time that includes GPU
  backend work and host support; no sentence treats it as GPU-kernel time.
- Preserved communication reporting as ciphertext-object counts and dependency phases only. No byte,
  bandwidth, transport-latency, or network-message estimate was added.
- Preserved the distinction between complete-model graph feasibility and one-input numerical
  agreement, including the single-input and single-execution qualifications.
- Preserved homogeneous attention accounting (client calls and ciphertext objects) and retained the
  warning that the logical-instance counter mixes units and cannot represent scalar work or traffic.

## Files edited

Only `paper-docs/manuscript/source/sections/*.tex` was edited. No evidence ledger, bibliography,
figure source or artifact, preamble, README, or repository documentation was changed by this pass.

## Checks

- `git diff --check -- paper-docs/manuscript/source/sections` — clean.
- `paper-docs/scripts/build.sh lint` — banned terms, evidence numbers, citations, and structure all
  clean.
- `paper-docs/scripts/build.sh` — regenerated all seven registered figures, rebuilt `main.pdf`, and
  reran all four lint checks successfully.
- Tectonic emitted only the pre-existing invalid-UTF-8 warnings from `algorithm.sty` and
  `lineno.sty`; PDF generation completed successfully.
