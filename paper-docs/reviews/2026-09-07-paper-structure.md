# Manuscript structural audit — 2026-09-07

Scope: structural rewrite for an IEEE Journal of Biomedical and Health Informatics regular-paper
submission after the complete twelve-block genomic-signal classifier result entered the evidence
ledger. This review follows `paper-docs/AGENTS.md`, the paper-structure agent contract, the supplied
IEEEtran brief, and the saved journal instructions.

## Breaks in the argument addressed

1. **The result had outrun the manuscript status.** Results, Limitations, Conclusion, and Future
   Work still described the complete classifier as unexecuted. They now report the verified
   twelve-block-plus-head execution for one held-out input, its final-label agreement, measured
   elapsed time, and process memory.
2. **Measured latency had been replaced by an obsolete “no timing” verdict.** The paper now reports
   the 6,683-second clean execution as a measurement while keeping repeat-run variation and
   networked transport unresolved. Practicality remains separate from arithmetic feasibility.
3. **The unsupported baseline factor remained structurally tempting.** The Optimization section
   states that the baseline denominator lacks matching evidence quality and therefore makes no
   speedup claim. The complete measured endpoint carries the performance result.
4. **Single-block and complete-model evidence were conflated.** The revised scope and work tables
   distinguish the task-length block, the complete classifier, and excluded private token-index
   lookup. The complete result is explicitly one held-out prediction, not test-set encrypted
   accuracy.
5. **IEEE double-column floats exceeded their columns.** Wide notation, deployment, optimization,
   scope, and operation tables are double-column floats. The packing diagram is likewise a
   double-column figure, and the long query/key/value pseudocode line is split.

## Order and placement

The retained journal argument is: motivation and feasibility question; cryptographic/model
background and related work; deployment and threat model; independently checked plaintext
reference; protocol; fixed-circuit systems work; results; limitations; and conclusion. Within
Results, evidence now precedes verdicts: plaintext fidelity, complete-classifier correctness,
executed work, elapsed-time split, memory, sequence-length boundary, feasibility, and practicality.

Submission declarations were removed from the Conclusion because the dedicated declarations file
already owns them. Empty biography and acknowledgment placeholders no longer create visible
manuscript sections. PDF metadata now matches the rendered title and author list.

## Gaps that remain visible

- The complete execution needs an independent clean repetition before variance can be stated.
- Client/server boundaries remain in-process; serialized transport and production key custody have
  not been measured.
- Private token-index lookup remains outside the encrypted boundary.
- One encrypted final prediction is not encrypted task-set accuracy.
- Institutional author e-mail addresses, corresponding-author designation, and funding/support
  text remain author-supplied submission metadata.

## What is working

The paper now leads with complete arithmetic closure, keeps the strong local-inference objection
before the protocol answer, preserves the two-stage plaintext-to-encrypted verification chain, and
attributes GPU and host memory findings to their distinct mechanisms. The fixed operation schedule
continues to make the systems section auditable without relying on the unsupported baseline factor.

## Focused layout repair and figure restoration

The author identified blank columns on pages 6–8, consecutive isolated algorithms, and missing
figures in the rendered manuscript. This focused structural pass recovers space for figure
restoration while the main editor repairs float placement. It edits only Introduction, Background,
Related Work, and Systems Optimization; the protocol and measured Results remain with the main
editor.

- Removed repeated motivation, contribution summaries, and generic explanations of the
  client-assisted division from the introductory and supporting sections. All existing citation
  keys in those sections remain in use.
- Consolidated repeated descriptions of diagonal preparation, cache indexing, and stage lifetimes.
  The systems section retains the checked operation counts, clone-versus-reuse mechanism,
  level-specific encoding, flush sites, host-memory findings, verification checks, and the
  distinction between measured endpoints and unmeasured speedup factors.
- Restored the single-column host-memory figure next to the bounded-cache measurements, with an
  explicit figure callout. Removed comments that treated missing figures as intentional
  space-saving exclusions.
- Preserved complete-model arithmetic closure, the one-input scope, the non-interactive memory
  boundary, and separate feasibility and practicality verdicts. No numerical result was changed or
  introduced during compression.

The remaining structural check is the main editor's rebuilt PDF: algorithms must flow with the
associated protocol text, restored figures must remain readable, and the final layout must fit the
journal page limit without blank columns caused by forced float barriers.
