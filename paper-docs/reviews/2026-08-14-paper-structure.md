# Academic and structural review — 2026-08-14

Scope: read-only review of the complete manuscript as an academic argument. The review considers
novelty, contribution framing, section balance, the claim--evidence chain, likely reviewer
objections, and audience fit. It treats the 2026-08-12 execution of all twelve blocks and the task
head at 103 tokens as current evidence. No manuscript or evidence file was edited.

## Overall verdict

This is a viable applied homomorphic-encryption systems paper after a focused major revision. It
now has a result worth publishing: a released genomic transformer was executed through all twelve
blocks and its released task head at the task-defined prompt length, from encrypted embedded
vectors, and produced the expected class under a predeclared numerical criterion. That is a
materially stronger paper than a one-block feasibility projection.

The paper should not try to win on cryptographic novelty or performance leadership. The
client-assisted pattern is prior art, and the manuscript correctly concedes that a comparable
non-interactive system reports much lower cost. Its defensible value is instead the combination of:

1. a complete, released-weight genomic-transformer execution rather than an operator or toy-model
   demonstration;
2. model-faithful nonlinear evaluation without homomorphic polynomial approximation, and without
   homomorphic bootstrapping;
3. an independently reconstructed numerical reference, an asserted operation schedule, and a
   traceable evidence record; and
4. an unusually candid feasibility/practicality split that identifies where this implementation
   spends time and memory.

The current draft partly obscures that contribution by making the comparison with THOR carry too
much argumentative weight, by presenting reference checkability as something client assistance
uniquely buys, and by making broad claims from one encrypted input and one complete-model timing
sample. These are repairable. The manuscript does not need a new architecture or a larger model to
be worthwhile.

## Highest-return work, in priority order

### Required claim repairs

1. **Repair the numerical-reference chain before relying on “independently checked.”**
   Sections 1, 4, and 7 say that the NumPy float64 reference is asserted against the upstream
   PyTorch computation at $2\times10^{-5}$ (`01_introduction.tex:64-69`,
   `04_plaintext_baseline.tex:24-31`, `07_results.tex:20-25`). The evidence audit shows that this
   gate is against a float64-cast shadow, not the released float32 inference path. The final label
   still agrees across float32, the float64 shadow, the manual reference, and encrypted execution,
   so the central result survives. The paper must state those links separately and report the
   float32-to-float64 drift. Until then, one of the paper's claimed methodological contributions is
   stronger than its evidence.

2. **Scope the correctness verdict to what one input establishes.** The full model has run once on
   one input. This proves that the complete graph is executable and that no numerical failure
   occurred for that prompt. It does not establish input-independent numerical stability or
   encrypted task accuracy. The limitations section says this clearly (`08_limitations.tex:10-15`),
   but the abstract, Key Points, Introduction, Results verdict, and Conclusion repeatedly state
   “correctness is not the barrier” without the same scope (`00_frontmatter.tex:55-60,81-83`,
   `01_introduction.tex:75-78`, `07_results.tex:227-240`, `10_conclusion.tex:7-17`). Either add a
   small multi-input encrypted panel or narrow the headline to “the executed complete-model case
   encountered no correctness barrier.” A full encrypted test set is unnecessary for this paper;
   a few inputs spanning both labels and different plaintext margins would address the actual
   generalization question.

3. **Correct the remaining headroom arithmetic.** “Nine orders of magnitude inside” the
   $4\times10^{-2}$ tolerance is wrong for an $8.56\times10^{-9}$ error; the headroom is about seven
   orders. The error audit has already identified the affected front-matter and Results sentences.
   This is small to fix but damaging if a reviewer finds it first.

4. **Do not infer a bottleneck from utilization alone.** The manuscript repeatedly says that a
   maximum sampled GPU utilization of 51% proves the evaluation is “not accelerator-bound” and that
   cost “sits in host-side encoding and in the sequential client boundary”
   (`00_frontmatter.tex:61-66,95-101`, `01_introduction.tex:89-92`,
   `07_results.tex:278-284`, `10_conclusion.tex:19-25`). The trace establishes that the implementation
   does not keep the accelerator saturated. It does not by itself distinguish host encoding,
   synchronization, memory bandwidth, kernel granularity, launch overhead, or inefficient GPU
   kernels. State the observation as underutilization unless a current CPU/CUDA profile attributes
   the idle periods. This is the largest remaining break between a measured fact and a causal
   systems claim.

5. **Separate protocol properties from evaluation methodology.** The abstract, Introduction,
   Results, Related Work, and Conclusion say that client assistance buys exact nonlinearities, no
   bootstrap, and an output checkable against a frozen plaintext reference. The first two are
   protocol properties. The third is not: a non-interactive encrypted implementation can also be
   tested against a frozen plaintext reference. Reference checkability is a strength of this
   study's validation method, not a compensating benefit of interaction. Keep it as a contribution,
   but remove it from the latency trade-off.

### Highest-value additional evidence

6. **Run one controlled paired single-block timing experiment.** Section 6 is careful not to claim
   a speedup because no dedicated-node pre-optimization baseline exists
   (`06_optimization.tex:37-44`). That honesty is correct, but it leaves the systems section with an
   enabling memory result and an operation-count reduction rather than a measured performance
   result. A same-node, same-input, same-circuit comparison between the documented pre-cache path and
   the retained path is the single experiment most likely to increase the paper's academic value.
   It does not require a second twelve-block baseline. Pair it with the CPU/CUDA profile above.

7. **Report communication complexity even if a networked implementation remains future work.** An
   interactive protocol with 10,299 in-process boundary calls cannot be evaluated academically from
   a crossing count alone. Reviewers will ask for ciphertexts or bytes transferred, the minimum
   number of sequential dependency phases, peak message size, and whether independent tiles could
   be batched into fewer network exchanges. Section 8 correctly says that no network messages were
   measured (`08_limitations.tex:27-32`), but deferring all communication analysis leaves the
   deployment claim incomplete. A derived byte/phase bound from the implemented layout is the
   minimum; a localhost serialization test or one simple network experiment would be better.

8. **Repeat the complete-model run and add a small input panel, not a large benchmark campaign.**
   One exact repeat is enough to stop presenting 6683 seconds as an isolated sample. Two or three
   additional prompts selected by label and plaintext margin would give a defensible numerical
   stability statement. This is a better use of compute than expanding to more plaintext tasks or
   another architecture.

9. **Measure the local/client comparison that the deployment argument uses.** “Well under a second”
   is not in the evidence ledger (`03_scenario_threat_model.tex:29-38`,
   `07_results.tex:251-256`). The same section assumes ordinary client CPU resources, but the
   evaluated client role ran in-process on a 32-thread compute node and standalone client peak memory
   is not reported. A named CPU forward-pass timing and a client-only CPU/RAM profile would turn the
   local-inference objection from plausible rhetoric into evidence. Otherwise remove the numerical
   timing phrase and keep the qualitative deployment distinction.

## Breaks in the argument

### 1. Full-model execution licenses feasibility, but not broad correctness

The manuscript now correctly says that the feasibility claim is no longer a projection from one
block (`07_results.tex:227-231`). That is the central advance. The next paragraph then says that
correctness, depth, and accelerator memory are “all resolved” for a complete model at task length
(`07_results.tex:233-240`). Depth and memory are structural properties observed across all twelve
blocks in the execution. Numerical correctness is more input-sensitive. One accepted prompt cannot
carry the same generality.

**Specific fix:** define two levels of conclusion:

- **graph feasibility:** measured for all twelve blocks and the task head at 103 tokens;
- **numerical generality:** demonstrated for one full-model prompt and therefore preliminary until a
  small input panel passes.

This keeps the full-model result strong without pretending it is a task-set evaluation.

### 2. The paper's differentiator is misclassified

The current contrast with THOR is: slower, but exact, bootstrap-free, and checkable against a frozen
reference (`01_introduction.tex:51-62`, `07_results.tex:258-268`,
`09_related_work.tex:34-53`). Only exact model formulas and the absence of homomorphic bootstrap
belong on the protocol side of that comparison. The frozen reference is experimental rigor shared by
any well-evaluated encrypted system.

**Specific fix:** use two separate paragraphs or table rows:

- **design trade-off:** exact client-side nonlinear formulas and shallow encrypted segments versus
  interaction, exposed client-side intermediates, and no cryptographic model confidentiality;
- **artifact contribution:** released weights, independently checked reference, asserted operation
  schedule, immutable evidence, and complete-run telemetry.

This makes the contribution more credible because it stops asking methodology to compensate for
latency.

### 3. The systems story identifies work, but does not yet measure its payoff

Section 6 is well structured and proportionate. Its best result is not speed: an unbounded encoded
plaintext cache reaches 61,863 MiB and dies, whereas stage-bounded lifetime management makes caching
usable and bounds host memory (`06_optimization.tex:16-27,84-110`). The asserted fixed circuit makes
the executed-work comparison trustworthy (`06_optimization.tex:29-40`). What the section cannot yet
support is a performance-improvement claim, because it intentionally omits a timing baseline.

The Conclusion nevertheless says host-side encoding and the sequential boundary are where cost
“sits” and that both are shown to be addressable (`10_conclusion.tex:23-25`). The boundary share is
measured; the causal share of host encoding is not.

**Specific fix:** either add the paired baseline/profile or keep Section 6 as an execution-enabling
memory and work-reduction result, and change the conclusion to say that the trace and operation
counts identify hypotheses for optimization rather than measured latency causes.

### 4. The served-model scenario does not yet close the input path

The local-inference objection is stated before it is answered, which is excellent. The answer rests
on a client that may not hold the served model (`03_scenario_threat_model.tex:40-52`). Yet the
evaluated encrypted boundary begins after token-index lookup and embedding construction
(`08_limitations.tex:34-37`). The paper does not state how a service-only client obtains or applies
the embedding table without receiving that part of the model. Since the experimental artifact is
public, the run itself is coherent; the hypothetical service deployment needs an explicit choice:
distribute the tokenizer and embedding table, or perform private lookup.

**Specific fix:** add one sentence to the deployment scenario stating which choice the evaluated
protocol assumes. Keep private lookup as the production gap. This is more important than repeating
that token lookup is out of scope.

The deployment table also omits the ordinary served-model baseline: send plaintext genomic input to
the server. Add that third column. The table then shows the actual choice the paper is designed to
change, instead of comparing only local inference with the proposed protocol.

### 5. The interaction cost is counted cryptographically but not characterized as a system

The protocol carefully distinguishes 857 physical crossings per block from 129,162 logical values
(`05_protocol.tex:189-197`). The full model has 10,299 in-process calls
(`07_results.tex:189-195`). That is useful, but a reviewer cannot infer remote feasibility from it.
Some calls may be independent and batchable; others are separated by the global softmax
synchronization point. The paper needs round/dependency structure and communication volume, not just
an implementation-call count.

**Specific fix:** place a compact communication-complexity table immediately after the full-model
cost table, with ciphertext count, total encoded bytes, sequential phases, and current in-process
status. Leave measured network latency in Limitations if it is not run.

### 6. The cross-paper comparison is useful context, not a controlled ablation

The manuscript is right to introduce THOR before readers mistake client assistance for a speed
claim. It is too strong to say that the comparison “decides” the relative-practicality verdict
(`07_results.tex:258-268`). The systems share parameters, accelerator class, depth, and approximate
shape, but they evaluate different model families, circuits, nonlinearity choices, implementations,
and output contracts. Raw wall time is informative, not controlled.

**Specific fix:** let 1.86 hours and 10,299 required boundaries decide the absolute practicality
verdict. Present THOR and MOAI as evidence that this implementation is not performance-leading. In
Related Work, replace the repeated prose comparison with a compact matrix containing model shape,
prompt length, hardware, security/ring/depth, interaction, nonlinear treatment, bootstrapping,
batch/amortization scope, and reported time. Label it explicitly as a cross-paper comparison.

### 7. The genomic motivation is broader than the evaluated input

The opening motivates privacy through whole-genome persistence, familial disclosure, and
re-identification (`01_introduction.tex:11-23`), while the measured task consumes one 600-base-pair
window (`07_results.tex:251-253`). A reviewer may reasonably ask whether the privacy consequences
claimed for a genome transfer unchanged to this short motif-classification input.

**Specific fix:** state that the evaluated prompt is a genomic fragment associated with a data
owner, not a whole-genome upload, and rely on the applicable genetic-data/privacy rationale that can
be supported for that scope. Do not imply that this experiment demonstrates private analysis of an
entire genome. If the service scenario assumes that many such windows are queried from one person's
genome, say so as an assumption rather than a measurement.

### 8. The security claim is adequate for a systems case study, not for a security proof paper

The threat model is unusually candid. It states semi-honest behavior, non-adaptive boundaries, no
model-confidentiality claim, no malicious-server protection, and no noise flooding
(`03_scenario_threat_model.tex:80-122`, `08_limitations.tex:39-61`). The algorithms and invariants
explain the intended transcript, but there is no simulation-based security argument and the
evaluated roles share one process and one node.

**Specific fix:** call this an evaluated protocol under a stated semi-honest model, not a proved
secure two-party system. A formal proof, malicious-client/server extension, and flooding analysis are
venue-dependent future work; they are not all required for an applied systems paper.

## Contribution and novelty framing

The five contribution bullets in the Introduction currently mix contributions, methodology, and
paper organization (`01_introduction.tex:94-112`). “Giving separate verdicts” and “naming the
measurements required” are good scholarly practice, not contributions. Plaintext task reproduction
is support for the encrypted study, not a standalone advance unless released as a reusable benchmark
artifact.

Use three contributions, led by the evidence that is now measured:

1. **Complete-model feasibility.** A released twelve-block genomic transformer and task head execute
   at the full 103-token task prompt from encrypted embedded vectors under client-assisted CKKS,
   without homomorphic bootstrap, and the evaluated prompt reproduces the reference label.
2. **Reproducible, model-faithful evaluation.** The paper supplies the protocol, packing, fixed
   operation schedule, predeclared acceptance rule, and a corrected validation chain from the
   released float32 path through the float64 reference to encrypted execution.
3. **Measured systems anatomy and scoped negative result.** The paper reports full-model timing,
   client/server work, memory across depth, interaction count, and the stage-bounded cache needed to
   make host-side encoding reuse executable; it also records where the evaluated non-interactive
   configuration exceeds its tested memory envelope.

A defensible one-sentence thesis is:

> Client-assisted CKKS can execute a complete released genomic transformer from encrypted embedded
> vectors at its real task prompt, with the released nonlinear formulas evaluated outside the
> homomorphic circuit and no bootstrap; the result is arithmetically feasible and reproducible, but
> one measured execution shows that the current interactive implementation is not yet practical or
> performance-leading.

This thesis claims enough to matter and no more than the evidence supports.

## Order and placement

1. **Keep the current macro-order.** Plaintext reference before protocol is correct: the numerical
   contract must exist before the encrypted circuit that claims to reproduce it. Related Work before
   Limitations is also correct because the limitation section can concede the competitive result
   rather than surprising the reader with it afterward.

2. **Shorten the THOR comparison in the Introduction.** Keep one sentence saying that comparable
   non-interactive systems are faster and that speed is not the claim. Move the parameter-by-parameter
   comparison to a table in Related Work. The current twelve-line comparison
   (`01_introduction.tex:51-62`) interrupts the paper's own research question before its method and
   contributions are stated.

3. **Add an “Evaluation setup and endpoints” subsection at the start of Results.** The current
   hardware, sample count, contention qualification, timing boundary, telemetry, and input-selection
   details are distributed across prose and captions. A systems reviewer needs one place that names
   the CPU and RAM, A100 model and capacity, software configuration, thread count, prompt selection,
   $n=1$ status, timing definition, and contention check. This also keeps caveats out of individual
   result paragraphs.

4. **Keep Section 6 at its current length.** Approximate word counts are: Results 2,260; Protocol
   1,430; host-side systems work 1,390; threat model 1,080; Introduction 1,200; Related Work 840;
   plaintext reference 790; Background 740. Results appropriately dominates, and the systems
   section is not crowded out by background or plaintext validation. Do not contract it into an
   implementation note. Give it a paired timing result if possible; otherwise retain its current
   execution-enabling framing.

5. **Reduce the contribution list and Key Points.** Five contribution bullets and eight Key Points
   cause the front of the paper to repeat the same result in slightly different taxonomies. Three
   contributions and four or five Key Points are enough. Preserve the complete-model result, scope
   from embedded vectors, single-sample practicality verdict, and interaction trade-off.

6. **Consider removing either the plaintext table or its bar figure if a page limit appears.** They
   duplicate the same metrics. The table carries the split/backbone qualifications more honestly and
   is the one to keep. This is a page-economy recommendation, not a scientific defect.

7. **Keep the separate Feasibility and Practicality subsections.** They are the manuscript's best
   organizing device. If a biomedical journal requires a conventional Discussion, move those two
   subsections together into a Discussion section without merging their verdicts.

## Strongest reviewer objections

| Objection | Current answer | What makes the answer solid |
|---|---|---|
| “Why not run the small public model locally?” | Correctly conceded and answered as a service-only deployment distinction. | Measure or remove the sub-second claim; state how the client gets the embedding table; add the plaintext-served baseline to Table 1. |
| “One prompt does not establish correctness.” | The paper admits one encrypted input and no encrypted task accuracy. | Add a small label/margin-stratified input panel or narrow the headline to executed-case feasibility. |
| “This is slower than non-interactive HE.” | Correctly conceded. | Treat THOR/MOAI as cross-paper context, not a controlled verdict; separate design properties from artifact rigor. |
| “Where are communication and round complexity?” | Deferred entirely to a future deployment experiment. | Provide bytes/ciphertexts and sequential-phase analysis now; network timing may remain future work. |
| “Low GPU utilization does not identify the bottleneck.” | The trace shows underutilization. | Add a CPU/CUDA profile or remove the causal attribution. |
| “Client assistance and decrypt--reencrypt are not new.” | The paper explicitly cites the antecedent and does not claim invention. | Lead novelty with the complete released genomic transformer, verification chain, and measured systems anatomy. |
| “What privacy statement applies to a 600-bp fragment?” | The introduction motivates genomic privacy generally. | Scope the motivating asset to the evaluated genomic fragment or justify the multi-window service scenario. |
| “Where is the security proof or malicious-server protection?” | The paper declares a semi-honest, in-process prototype and lists exclusions. | Submit as applied systems/engineering work; do not imply a formal or production security result. |
| “Is 6683 seconds reproducible?” | It is explicitly one uncontended execution. | Repeat it under an explicitly exclusive allocation and report a distribution, even if only two or three runs. |

## Venue and audience fit

Without a named target venue, the manuscript fits best as an **applied homomorphic-encryption,
privacy-engineering, or systems evaluation paper**. That audience will value the released model,
complete execution, exact operation schedule, failure mechanism, memory traces, and negative
practicality result. For that audience, the highest-value additions are the paired baseline/profile,
communication complexity, and a small replication panel.

It is currently a weaker fit for three other audience types:

- **Cryptography or security-theory audience:** the client-assisted primitive is not new; there is
  no formal security proof, malicious-server analysis, or deployed two-party separation. Reaching
  that audience would require more than polishing.
- **Mainstream systems audience:** one run, one input, no network, and no controlled optimization
  timing leave the performance evaluation too thin. The additions above could close most of that
  gap without changing the architecture.
- **Bioinformatics methods audience:** the biological tasks validate the target but the paper adds
  no biological method or biological finding, and encrypted evaluation covers one prompt. Reaching
  that audience would require a broader encrypted task evaluation and a stronger account of why
  the 600-bp service input is sensitive and scientifically useful. Merely adding more plaintext
  metrics would not be enough.

The paper should choose the first audience and write its title, abstract, contribution list, and
evaluation priorities for that reader. It can still motivate the work genomically without claiming a
bioinformatics-method contribution.

## Gaps reviewers will ask about

These are ordered by effect on acceptance, not by ease:

1. communication volume and dependency-round complexity for 10,299 client calls;
2. multi-input full-model numerical behavior and one independent timing repeat;
3. a controlled pre-optimization/retained timing comparison plus a current CPU/CUDA profile;
4. the corrected float32--float64--manual--encrypted reference chain;
5. the client-side embedding-table assumption at the encrypted boundary;
6. representative client CPU time and memory, rather than an in-process 32-thread role;
7. explicit exclusive-node evidence and run-to-run variance for timing;
8. security-parameter attestation and a noise-flooding budget if the target venue expects a stronger
   cryptographic argument.

Private token-index lookup, malicious-server security, and a production key-custody service are real
limitations but do not all need to be implemented for this paper. They must remain explicit scope
boundaries.

## What is working and should not be restructured away

- **The full-model result is direct evidence, not a projection.** The manuscript has correctly
  changed its headline after the 2026-08-12 run.
- **Feasibility and practicality are separate verdicts.** The affirmative arithmetic result and
  negative deployment/performance result can coexist without weakening each other.
- **The local-inference objection is stated at full strength before the answer.** This reads as
  confidence rather than evasion.
- **The non-interactive memory result is scoped to the evaluated implementation.** The paper no
  longer turns one backend's memory wall into an impossibility claim.
- **The operation schedule is a run-time assertion.** That makes fixed-circuit systems changes
  auditable and is stronger than relying only on final-output equality.
- **The stage-bounded cache is a real enabling finding.** It explains why an obvious optimization
  fails and why the retained lifetime policy works.
- **The complete-run memory trace changes the evidence.** It shows that GPU state does not grow with
  block index and that host memory follows repeated stage lifetimes rather than accumulation.
- **The limitations are candid.** One input, one timing run, in-process roles, no encrypted lookup,
  no model confidentiality, and semi-honest security are visible rather than buried.
- **Section balance is already sound.** The paper does not need a wholesale reorganization.

## A non-perfectionist path to a solid submission

If resources are limited, do these and stop:

1. fix the oracle-chain wording, headroom arithmetic, “exact LayerNorm” overreach, and the causal
   interpretation of GPU utilization;
2. reframe the three contributions and stop treating frozen-reference verification as a benefit
   unique to client assistance;
3. add a derived communication/round-complexity table;
4. run one paired pre-cache/retained single-block comparison with a current profile;
5. repeat the complete run once and add two or three prompts spanning labels/margins, or narrow the
   correctness headline if those runs are not available; and
6. add a compact Evaluation Setup subsection and a structured cross-paper comparison table.

Do not spend the next cycle adding more plaintext datasets, testing a third cryptographic scheme,
or polishing every sentence. Those activities will not answer the objections that currently decide
whether the paper's contribution is convincing.
