# Paper style audit — 2026-08-08

## Scope and verdict

Full-manuscript pre-submission audit of `manuscript/source/sections/*.tex`, using
`.claude/agents/paper-style.md`, `paper-docs/AGENTS.md`,
`context/00_terminology.md`, and the global writing-style policy. This was a diagnostic pass over
the shared manuscript snapshot. No manuscript, evidence, figure, or bibliography file was edited.

**Verdict: substantial style revision is advisable before submission.** The draft is technically
specific, restrained, and mostly free of filler, but it has the exact machine-prose signature that
the project instructions warn about: a large fraction of paragraphs close with a contrast,
qualification, or disclaimer. Cross-section repetition then makes several important limitations
feel routine rather than salient.

## Severity-ranked findings

### High — paragraph endings repeat one rhetorical shape

A lexical screen found **50 qualification/contrast endings among 84 prose blocks**. It also found
30 uses of “rather” and 22 of “therefore.” The screen is deliberately broad, but the concentration
is visible without the count: many paragraphs make a claim and then close with “not X,” “rather
than Y,” “does not establish Z,” or “remains unresolved.”

The densest areas are:

- `08_limitations.tex:10-62`: nearly every paragraph ends by negating a stronger claim. The
  content belongs here, but the section reads as a sequence of identically shaped disclaimers.
  Consolidate the projection qualifications (`:10-21`), keep the model-confidentiality consequence
  (`:35-45`), and vary the remaining paragraph endings so some land on the experiment needed or
  the mechanism exposed.
- `09_related_work.tex:10-85`: each literature group is followed by a local non-claim: “not
  directly comparable” (`:20-21`), “do not claim ... as new” (`:27-32`), “not a general
  impossibility” (`:37-42`), “not priority” (`:54-55`), “rather than a property of CKKS”
  (`:62-65`), and “do not supply drop-in speedups” (`:80-85`). Keep the Safhire and THOR
  concessions, but collect the overall positioning into one closing paragraph. Let the genomics,
  GPU, and matrix-primitives paragraphs end on the computation each body of work enables.
- `06_optimization.tex:16-142`: the fixed-circuit distinction is important, but the prose often
  reaches it through a closing negation (`:25-30`, `:39-48`, `:66-69`, `:84-88`, `:138-142`).
  Recast several positively around the checked schedule, clone lifecycle, and cache topology.
- `07_results.tex:17-167`: paragraph endings repeatedly distinguish a result from a stronger
  reading (`:24-28`, `:90-96`, `:126-144`, `:155-167`). Preserve the feasibility/practicality
  split, but move secondary scope distinctions earlier in their paragraphs.

The strongest “X is A, not B” sentences still earn their place: the feasibility/practicality
verdict and the operation-count/latency distinction. The rest should not imitate their cadence.

### High — limitations and deployment premises recur outside their homes

- The local-inference premise receives two full treatments in `01_introduction.tex:23-32` and
  `03_scenario_threat_model.tex:23-36`. Retain the full argument in the deployment section and
  compress the introduction to one sentence establishing why the served-model case exists.
- The unexecuted complete-model evaluation is restated in `05_protocol.tex:248-254`,
  `07_results.tex:126-129`, `08_limitations.tex:10-21`, and `10_conclusion.tex:7-14`, in addition to
  the projected labels in the front matter. Its required homes are the feasibility verdict and
  limitations; the protocol needs only the composition mechanism, and the conclusion can refer to
  the projection without repeating the entire caveat.
- Private token lookup and adversary exclusions are listed in both
  `03_scenario_threat_model.tex:105-109` and `08_limitations.tex:30-33,59-62`. Choose one complete
  list and cross-reference it from the other section. The threat model is the more natural home for
  exclusions; limitations should explain the operational consequence of private lookup.
- The dataset qualification is already explained at `04_plaintext_baseline.tex:76-81`; the reminder
  at `08_limitations.tex:55-57` adds another disclaimer without adding information.
- Model confidentiality is deliberately stated in both §3 and §8. That repetition is required by
  the manuscript contract and should remain.

### High — several terms sound like development bookkeeping or drift from defined vocabulary

- “passes local contracts” appears at `05_protocol.tex:253-254`, `07_results.tex:126-129`, and
  `08_limitations.tex:10-14`. A reader cannot tell what a local contract checks. Replace all three
  occurrences with the same concrete, evidence-supported description of the pre-execution checks,
  or omit the phrase.
- “plaintext oracle” and “same oracle” at `09_related_work.tex:28-31,83-85` should be **frozen
  plaintext reference**, the term defined for the manuscript.
- “physical client crossings” at `07_results.tex:63` and “physical client crossing” at
  `08_limitations.tex:25-26` should retain the complete term **physical client boundary crossing**.
- “frozen constants” at `06_optimization.tex:20-22` is implementation-flavored. Naming the asserted
  operation schedule and expected boundary counts is clearer.

### High — one practicality sentence outruns the evidence

`07_results.tex:134-137` says the projected costs “can support batch or offline analysis.” The paper
defines no batch-service latency target, throughput target, or deployment measurement. The
interactive verdict is supported; the affirmative batch-use statement is not. Remove it or make an
explicitly conditional claim tied to a named use case and service requirement. This needs an author
decision because changing the sentence changes the practicality claim.

### Medium — headings are conversational rather than consistently technical

The numbered headings use sentence case, but several are rhetorical or promotional:

- `02_background.tex:7`: “Background: what encrypted transformer inference requires”
- `04_plaintext_baseline.tex:7`: “Is the target worth encrypting?”
- `05_protocol.tex:7`: “Two designs, and where each one stops”
- `06_optimization.tex:13`: “Making it fast without changing the arithmetic”
- `06_optimization.tex:32`: “The binding cost was not the GPU”
- `07_results.tex:69`: “Where the time actually goes”

Prefer domain headings that survive skimming and indexing: **Background**, **Plaintext reference
and model fidelity**, **Client-assisted CKKS protocol**, **Systems optimization at fixed circuit**,
**Host-side encoding bottleneck**, and **Resource trace**. Similar tightening would help “Where the
boundary goes,” “The protocol, stated precisely,” and “Headroom that remains.” “Threat model” is
already used correctly and should not be softened.

Unnumbered headings mix title case with sentence case: “Key Points,” “Biographical Note,” “Ethics
Statement,” “Data Availability Statement,” and “Code Availability Statement.” Convert them to
sentence case unless the target journal mandates those exact labels.

### Medium — mechanical consistency needs a copy-edit pass

- The manuscript mixes US and UK English: “behavior” (`02_background.tex:67`,
  `04_plaintext_baseline.tex:20`) with “analysed” (`06_optimization.tex:130`), “labelled”
  (`08_limitations.tex:15`), “licence,” and “Acknowledgements” (`10_conclusion.tex:59,61`). Select
  one dialect; US English is the natural default for the present affiliation.
- “run-time assertions” (`00_frontmatter.tex:90`, `01_introduction.tex:62`) conflicts with
  “runtime” (`08_limitations.tex:20`, `09_related_work.tex:61`). Use **runtime** consistently.
- `06_optimization.tex:148-159` mixes hours and seconds for block latency in one table. Use one unit
  in the Effect column, subject to the evidence ledger’s numeric rule.
- Citation spacing is mixed: the manuscript has 15 nonbreaking `~\cite...` forms and body citations
  with ordinary spaces at `02_background.tex:24-26,67`. Apply one convention. The prose at
  `04_plaintext_baseline.tex:71-72` is also smoother as “the value reported by ...” than “the
  $0.62$ [author] report.”
- `05_protocol.tex:97-98` is grammatically incomplete: “for each $\delta$ any later query group
  needs.” Rewrite as “for each $\delta$ required by a later query group,” without changing the
  algorithm.
- `06_optimization.tex:52-54` uses the awkward “device-unloaded” and says loading and eviction have a
  “lifetime.” State the lifecycle concretely: each multiplication receives a clone that is loaded
  and evicted for that operation.
- `10_conclusion.tex:10-13` says the process “remained within $9.6$ GiB.” “Peaked at $9.6$ GiB” is
  the precise idiom already used elsewhere.

### Medium — two captions include material not represented by the visual

- `07_results.tex:32-35` captions a wall-time split but adds “roughly 14 GB of peak memory.” Memory
  is not part of the cost-split visual, the quantity uses a different unit convention from the
  manuscript’s GiB results, and “roughly” is prohibited when precision is uncertain. Remove the
  memory clause or move an exact ledger-supported client-memory value to the relevant text/figure.
- `07_results.tex:71-74` states the interpretation “not GPU-bound” in the timeline caption, then
  repeats it in `:78-83`. Keep enough in the caption to make the figure self-contained, but avoid
  duplicating the full inference immediately below.

The other captions are history-free and mostly self-contained. The architecture, packing, memory,
waterfall, and scaling captions explain the depicted relationship without referring to the review
process. The waterfall caption appropriately carries the fixed-circuit invariant because that
qualification changes how the figure may be read.

### Medium — a few phrases weaken an otherwise precise academic tone

- “The obvious alternative” (`01_introduction.tex:23`) and “The strongest objection”
  (`03_scenario_threat_model.tex:24`) stage a debate. State the alternative directly.
- “The natural non-interactive design” (`05_protocol.tex:49-50`) labels a design choice as natural;
  “The non-interactive design” is sufficient.
- “Without a lifetime bound, the exchange is fatal” (`06_optimization.tex:71-75`) is dramatic and
  abstract. The following measured process termination is stronger without “fatal.”
- “the hardest measured case” (`07_results.tex:158-160`) should be “the longest measured prompt” or
  another named dimension.
- “the necessary non-interactive counterpoint” (`09_related_work.tex:34`) editorializes. THOR can be
  introduced directly by the distinction it establishes.

## What is already working

- No uses of “Furthermore,” “Moreover,” “Additionally,” “it is worth noting,” exclamation points,
  or the policy’s hype vocabulary were found.
- Technical terms such as threat model, ciphertext--plaintext multiplication, multiplicative depth,
  and client boundary crossing are generally preserved rather than over-glossed.
- The manuscript leads with the successful fixed-circuit optimization and states separate
  feasibility and practicality verdicts.
- Negative experiments are used to explain retained choices rather than collected into a failure
  catalogue.
- Paragraph openings have reasonable variety; the recurring signature is concentrated at paragraph
  endings, not beginnings.
- The central repetitions “correctness is not the barrier,” “memory is a barrier that design moves,”
  and “cost remains the constraint” function as the paper’s deliberate spine and should not be
  removed merely to avoid repetition.

## What changed and why

Only this review report was added. The parent task required all concurrent reviewers to inspect an
identical manuscript snapshot, so no prose, heading, caption, bibliography, evidence, or figure was
changed.

## What was left alone deliberately

- All numeric values, scope words, measured/projected distinctions, and evidence-required
  qualifiers.
- The explicit model-confidentiality limitation in both §3 and §8, because the paper contract
  requires the consequence in both locations.
- Algorithm operations and ordering; the one grammar issue is reported rather than silently edited.
- Figure captions that repeat a qualifier needed for self-contained interpretation, especially the
  fixed-circuit waterfall and projected scaling captions.
- Author metadata and repository licensing, which require author action rather than stylistic
  judgment.

## Author decisions required

1. Decide whether to remove or support the batch/offline practicality claim in
   `07_results.tex:134-137`.
2. Define what “passes local contracts” means in publication-facing terms before replacing its
   three occurrences.
3. Choose US or UK English and confirm whether the target journal mandates title-case unnumbered
   headings.
4. Supply the corresponding-author email, biographical note, acknowledgements, and repository
   licence before submission.

## Checker

The required read-only post-audit run of
`paper-docs/.venv-paper/bin/python paper-docs/scripts/check_numbers.py` reports:

- Banned terms: clean
- Numbers without evidence: clean
- Citations: clean
- Structure: clean

All checks clean.
