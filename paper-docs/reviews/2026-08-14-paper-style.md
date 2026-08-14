# Paper style review — 2026-08-14

## Scope and verdict

This is a review-only academic style audit of the full manuscript in
`manuscript/source/sections/*.tex`. It applies `.claude/agents/paper-style.md`,
`paper-docs/AGENTS.md`, `context/00_terminology.md`, and the global writing-style policy. No
manuscript, figure, evidence, or bibliography file was edited.

**Verdict: the paper has enough technical substance for a solid submission, but one focused prose
revision is still needed.** The strongest contribution is concrete: a released twelve-block
genomic transformer and task head execute at the task-defined prompt under client-assisted CKKS;
the output is checked through a two-stage reference chain; and the paper exposes the depth, memory,
operation schedule, and client/server cost of doing so. The fixed-circuit cache design and the
finding that softmax dominates boundary traffic are also transferable systems results.

The draft's main weakness is that it spends too much space defending what it does not claim. The
same concessions recur in the abstract, key points, introduction, results verdicts, related work,
limitations, and conclusion. This makes the prose sound machine-revised and leaves the paper's
positive value less prominent than its comparison with THOR. A high-return revision would correct
the few over-broad claims below, cut repeated qualifications, and shorten the front matter and
introduction. A sentence-by-sentence beautification pass is not necessary.

## Must fix before submission

### 1. Two “nine orders” statements are numerically wrong

- `00_frontmatter.tex:81-83` says the margin error is “nine orders of magnitude inside” the
  tolerance.
- `07_results.tex:40-47` says the classifier margin remains inside the tolerance “by nine.”

The reported values are $8.56\times10^{-9}$ and $4\times10^{-2}$. Their ratio is approximately
$4.7\times10^6$, or 6.67 orders of magnitude. “Nearly seven orders of magnitude below the
tolerance” is supportable; “nine orders” is not. This semantic error passes the number lint because
both individual numbers exist in the ledger.

### 2. GPU telemetry does not by itself prove that the run is not accelerator-bound

The manuscript makes the causal claim in the abstract (`00_frontmatter.tex:60-64`), key points
(`:95-97`), introduction (`01_introduction.tex:89-92`), timeline caption
(`07_results.tex:102-105`), practicality verdict (`:278-282`), and conclusion
(`10_conclusion.tex:19-25`). A maximum sampled utilization of $51\%$ shows that the GPU was not
saturated in the recorded trace. It does not, without a profile or intervention, establish that
GPU latency, synchronization, kernel granularity, or host--device dependencies do not determine the
wall time.

Use the measured statement in the headline locations: **sampled GPU utilization had a median of
$32\%$ and a maximum of $51\%$**. Treat host encoding and sequential client boundaries as candidate
bottlenecks identified by the implementation and trace, not as a completed causal attribution.

### 3. The affirmative batch/offline-use claim outruns the evaluation

`01_introduction.tex:89-92`, `07_results.tex:251-256`, and `10_conclusion.tex:19-22` say that the
measured cost leaves batch or offline analysis available. The paper defines no batch workload,
throughput target, cost budget, or service requirement, and it has one timing sample. The negative
interactive verdict is supported. An affirmative batch-suitability verdict is not.

The economical fix is: **the measured cost rules out interactive use; suitability for batch or
offline use depends on workload and service constraints not evaluated here.** If the authors want
to retain the positive batch claim, they need a named use case and acceptance threshold.

### 4. Narrow three headline claims to the experiment actually performed

- **Correctness.** “Correctness is not the barrier” appears in the abstract, key points,
  introduction, feasibility verdict, and conclusion. The complete encrypted model was evaluated on
  one input, while Section 8 correctly states that encrypted task accuracy over a test set was not
  measured. Prefer **numerical agreement was achieved for the evaluated complete-model input** in
  headline prose. The broader statement can otherwise be read as circuit correctness over the
  input domain.
- **Memory.** “The memory barrier ... is removed” is defensible only for the barrier encountered by
  the evaluated non-interactive configuration. The body usually carries that scope, but
  `00_frontmatter.tex:84-89`, `07_results.tex:108-114`, and `10_conclusion.tex:12-17` compress it
  into a general result. Prefer **client assistance avoided the $80$~GB composition failure observed
  in the evaluated non-interactive configuration; peak GPU memory did not grow with block index in
  the measured run**.
- **LayerNorm.** `00_frontmatter.tex:45-46` says the client evaluates “exact LayerNorm.” The server
  computes the LayerNorm statistics under approximate CKKS arithmetic; the client evaluates the
  inverse square root. Name the actual boundary function: **the LayerNorm inverse square root,
  causal softmax, and GELU**.

These changes do not weaken the paper. They align its strongest sentences with the experiment and
make the verified result harder to challenge.

## High-value style revision

### 5. Consolidate repeated caveats and comparisons

A rough lexical screen of the prose found 45 uses of “rather,” 30 of “therefore,” and 100 of
“not.” A paragraph-shape screen marked 34 of 103 prose blocks as ending in a contrast or
qualification. The exact counts are heuristic; the visible pattern is the issue. Paragraphs
repeatedly end with “not X,” “rather than Y,” “does not establish Z,” or a warning against a
stronger reading.

The main repetitions are:

- The **single-execution qualification** appears about ten times, including both the cost-figure
  caption and cost-table caption, the results opening, the practicality verdict, limitations, and
  conclusion. Keep it once in the abstract because it changes how the headline latency is read,
  once with the results table, and fully in limitations.
- The **THOR/non-interactive cost concession** occupies the abstract conclusion, key points,
  introduction, practicality verdict, related work, limitations, and conclusion. Keep one concise
  comparison in the introduction, the detailed comparison in related work or the practicality
  verdict, and one consequence in limitations. The present repetition makes THOR feel like the
  paper's subject.
- **Encrypted token-index lookup** is mentioned in the abstract twice, the threat model, the scope
  table, feasibility verdict, limitations, and future work. Retain the abstract boundary statement,
  its formal home in the threat model or limitations, and the future-work action. The other reminders
  add little.
- **Tolerance fixed in advance**, **no homomorphic bootstrap**, **exact nonlinearities**, and the
  **feasibility/practicality split** each recur across most major sections. Preserve each as a
  load-bearing idea, but not in every summary layer.

The model-confidentiality limitation is the exception. Its appearance in both the threat model and
limitations is required by the paper contract and should remain.

### 6. Shorten the front matter and let it end on this paper's result

The structured abstract is approximately 370 words, followed by eight key points totaling roughly
320 words. The key points largely repeat the abstract, including two separate bullets on the
non-interactive comparison. This is substantial reader burden before the introduction begins, and
it weakens the hierarchy of findings.

Unless the target venue prescribes otherwise, aim for a shorter abstract and four or five key
points:

1. the evaluated question and encrypted boundary;
2. complete-model numerical agreement and stable memory;
3. the fixed-circuit systems result and boundary-cost finding; and
4. the measured complete-inference cost and negative interactive verdict.

One sentence can acknowledge that cited non-interactive systems are faster. Ending both the
abstract and key points with a three-part defence of client assistance makes the paper sound more
apologetic than warranted.

### 7. Recenter the introduction on the paper's concrete value

The introduction is about 1,160 words and performs the work of motivation, threat-model defence,
related-work comparison, methods summary, full results summary, and a five-item contribution list.
The following passages are the best compression targets:

- `01_introduction.tex:51-62` gives a detailed THOR/MOAI comparison that related work repeats.
  Reduce it to one scoped sentence and preserve the detailed parameter comparison for Section 9.
- `:75-92` restates most of the results section, including correctness, both memory mechanisms,
  non-interactive scope, latency, batch use, and GPU utilization. Retain the three verdicts, but use
  one compact paragraph.
- Contribution bullet 4 (`:106-110`) carries runtime invariants, cache lifetime, host termination,
  remaining encoding work, and sequential boundaries in one sentence. Split the retained
  contribution from future optimization targets, or omit the latter here.

The introduction already has a defensible novelty sentence at `:47-49`: the contribution is not the
client-assisted division itself, but an auditable application to a released genomic transformer.
Strengthen that concrete positioning with the full-model execution, verified reference chain,
fixed-circuit cache policy, and measured work allocation. Do not add an unsupported “first” claim.

### 8. Remove debate staging and manuscript-process commentary

Several passages address an imagined reviewer instead of stating the scientific point:

- “we state that before reporting our own numbers rather than after”
  (`01_introduction.tex:51-52`);
- “The sharpest objection ... is best stated ... at full strength” and “Every one of those
  statements is correct, and none ... is contested below” (`03_scenario_threat_model.tex:29-38`);
- “The answer is not that the arithmetic is wrong” (`:40-45`);
- “The claim is no longer an argument ...” (`07_results.tex:227-231`);
- “The comparison is therefore generous to the design evaluated here” (`:270-276`).

These sentences are candid, but together they produce a rehearsed, machine-like rebuttal cadence.
State the premise directly: the protocol is relevant when the model is served and not distributed;
local inference is preferable when the client may hold the model. The deployment section can make
that point in one paragraph plus Table 1 rather than three paragraphs of objection and response.

### 9. Reduce reader burden in the protocol and results

The protocol is admirably reproducible, but four displayed algorithms plus packing, attention,
parameters, and composition interrupt the main argument. If the venue permits an appendix or
supplement, keep the top-level block and client-boundary algorithms in the main text and move the
LayerNorm and packed-matrix-product primitives. If they remain, add a short roadmap before the
algorithms telling the reader which two mechanisms matter for the paper's claim: level-$0$
re-encryption resets segment depth, and stage-bounded encoded-plaintext caching controls host
memory.

The results section is about 2,270 words and then ends with feasibility and practicality
subsections that repeat the preceding correctness, memory, and timing subsections. The verdict
subsections are valuable and contractually required; make them short syntheses and remove duplicate
explanation from either the verdicts or the immediately preceding text.

The optimization section should likewise lead with its retained result. The opening at
`06_optimization.tex:16-44` spends three paragraphs explaining what the section is not and why no
speedup is reported. A stronger opening is the measured mechanism: unbounded caching reached
$61{,}863$~MiB and failed; stage-bounded caching reduced redundant encoding while setting peak host
memory by the largest live stage. State the fixed-circuit invariant once after that result.

## Terminology and mechanical consistency

### Publication terminology

- Replace **“plaintext oracle”** at `07_results.tex:34-38` and
  `09_related_work.tex:46-50` with **“frozen plaintext reference.”** “Oracle” is repository
  vocabulary, and the manuscript defines the latter term.
- At first mention in `00_frontmatter.tex:42`, use **“pure non-interactive CKKS configuration,”**
  not “pure non-interactive configuration.”
- `00_frontmatter.tex:47-49` says the model runs “in one encrypted lineage,” although declared
  client decrypt--evaluate--encrypt boundaries and inter-block refreshes create fresh ciphertexts.
  **“Under one key lineage, with declared client refreshes”** is less likely to imply an
  uninterrupted ciphertext lineage.
- “Identical cryptographic parameters” in `00_frontmatter.tex:63-64` and
  `08_limitations.tex:21-23` is broader than the comparison shown. Enumerate the matched parameters
  (ring degree, depth, security level, and accelerator) and call the model shapes comparable.
- “Non-interactive state of the art” at `07_results.tex:264-266` and
  `09_related_work.tex:46-47` is an unbounded literature claim. Unless a current systematic search
  supports it, say **“the cited non-interactive systems.”**
- “What it buys” appears in both the abstract and related work. Prefer a neutral construction such
  as **“The design instead provides ...”**
- “Fatal” in `00_frontmatter.tex:90-93` and `06_optimization.tex:19-24` is dramatic and less precise
  than the measured outcome. Say that the process was terminated or that the unbounded cache was
  not executable within the tested host-memory envelope.

### Mechanics

- The manuscript is predominantly US English but retains **“favour”**
  (`01_introduction.tex:51`, `07_results.tex:259`) and **“analysing”**
  (`07_results.tex:254-256`). Use **“favor”** and **“analyzing.”**
- Citation spacing is almost consistently nonbreaking; `02_background.tex:67` uses an ordinary
  space before `\cite{openfhe}`.
- Unnumbered heading case is mixed: “Key Points” is title case, while “Biographical note,” “Ethics
  statement,” and the availability statements are sentence case. Follow the target venue; absent a
  mandate, sentence case is consistent with the manuscript.
- Several captions are doing too much interpretive work. The timeline caption
  (`07_results.tex:102-105`) combines utilization, causal attribution, host-memory behavior,
  GPU-memory behavior, and block identification. Describe what is plotted and leave the causal
  claim to the results text.
- A few recently expanded sentences exceed 45 words and carry multiple claims, especially
  `03_scenario_threat_model.tex:118-122`, `07_results.tex:251-256`, and
  `09_related_work.tex:84-93`. Split them at the change from observation to implication.

## What is already working

- The title is descriptive and avoids claiming end-to-end private token lookup.
- The manuscript has an appropriately restrained academic tone overall: no hype, empty
  intensifiers, exclamation points, or failure catalogue.
- The threat model states the protected asset and model-confidentiality limitation with unusual
  clarity. Keeping that concession strengthens the paper.
- The two-stage plaintext-reference chain is a genuine methodological strength and is now stated
  clearly in the baseline and results sections.
- Measured operation counts are distinguished from latency, and the $7.92\times$ packing reduction
  is correctly described as structural.
- The non-interactive negative result is scoped to the evaluated backend, packing, parameters, and
  hardware, and its mechanism is explained instead of merely reported.
- The optimization section preserves the fixed-circuit invariant and does not manufacture a
  speedup from a contended baseline.
- The paper reports feasibility and practicality separately and gives the negative practicality
  verdict directly.
- The algorithms, tables, and captions mostly retain domain terminology rather than replacing it
  with generic prose.

These strengths are enough to support the paper's value. The recommended revision should reveal
them more clearly, not enlarge the claim set.

## Minimal revision sequence

For a solid paper without perfectionism:

1. Fix the orders-of-magnitude error, narrow the utilization and batch-use claims, and correct the
   LayerNorm boundary wording.
2. Reduce the abstract and key points, then compress the THOR comparison and full-results recap in
   the introduction.
3. Give each recurring caveat one home and remove the duplicate disclaimer endings elsewhere.
4. Replace the terminology drift and complete the small US-English/citation-spacing copy edit.
5. Re-run the paper checks and stop unless the target venue imposes a stricter word or highlight
   limit.

## What changed and why

Only this review report was added. The manuscript and figures were intentionally left unchanged so
the academic review remains independent and can be reconciled with the evidence, sources,
structure, and figure reviews.

## What was left alone deliberately

- All manuscript numbers, scope words, and evidence-required qualifiers.
- Algorithm operations and order, which are implementation-grounded and should not be altered in a
  style-only pass.
- The repeated model-confidentiality consequence in Sections 3 and 8, because the paper contract
  explicitly requires both.
- Author metadata, biographical material, acknowledgments, and licensing statements, which require
  author decisions.

## Author decisions required

1. Choose the target venue and apply its abstract-length, key-point, appendix, heading, and
   biographical-note requirements.
2. Decide whether to limit the headline correctness claim to the evaluated input or add encrypted
   test-set evidence.
3. Remove the affirmative batch/offline-use claim or define a concrete batch workload and
   acceptance threshold.
4. Decide whether “state of the art” is supported by a current systematic literature search; if
   not, scope the comparison to the cited systems.

## Checker

The required checker was run with the paper environment:

```text
paper-docs/.venv-paper/bin/python paper-docs/scripts/check_numbers.py
Banned terms: clean
Numbers without evidence: clean
Citations: clean
Structure: clean
```

The system `python3` lacks PyYAML, so the repository's paper environment was used. All checks pass;
the orders-of-magnitude issue above is semantic and is not detected by the current lint.
