# Integrated academic review — 2026-08-14

## Executive verdict

There is a real paper here. The strongest result is direct rather than projected: all twelve
released DNAGPT blocks and the released task head execute at the task-defined 103-token prompt from
encrypted embedded vectors, and the evaluated prompt produces the same class as the plaintext model.
The operation schedule, memory behavior, client/server split, and complete-run cost are reported in
unusual detail. The stage-bounded encoded-plaintext cache is also a useful systems finding: it turns
an otherwise fatal host-memory trade-off into an executable one without changing the encrypted
circuit.

The current draft nevertheless needs a focused major revision before submission. Its central run is
sound, but several headline claims are broader than the experiment, the reference chain is described
imprecisely, communication is counted but not characterized, and one utilization trace is asked to
support causal performance conclusions it cannot establish. The paper also remains one input and one
complete-model execution.

The right positioning is an **applied homomorphic-encryption systems and feasibility study**, not a
new cryptographic construction, a performance-leading private-transformer system, or a
bioinformatics-method paper. Under that framing, the contribution is solid and the necessary work is
finite.

**Recommendation today:** major revision. A demanding systems or security reviewer could reasonably
reject the current version, chiefly for the one-input/one-run evaluation and absent communication
analysis. After the bounded revision below, it should be a credible applied-HE or privacy-engineering
submission.

## What the paper adds

The audited literature already contains client-assisted neural inference and non-interactive
homomorphic transformer inference. The paper should therefore avoid treating client assistance,
CKKS transformer arithmetic, or diagonal matrix packing as its novelty.

Its defensible contribution is the combination of:

1. **Complete released-model execution.** The experiment evaluates a real twelve-block genomic
   transformer and task head at a task-derived prompt length, not an isolated operator, toy block, or
   whole-model projection.
2. **A reproducible model-to-ciphertext chain.** The repository records the plaintext paths,
   independent numerical re-derivation, encrypted comparison, fixed operation schedule, memory
   trace, and immutable evidence.
3. **Measured systems anatomy.** The paper reports encrypted operation counts, 10,299 full-model
   client calls, client/server time, depth, GPU and host memory, and the cost of one complete
   execution.
4. **Two transferable negative findings.** The evaluated non-interactive composition configuration
   exceeds its tested accelerator-memory envelope, while unbounded encoded-plaintext caching exceeds
   the host-memory envelope. The paper explains both mechanisms and the retained design response.
5. **An honest feasibility boundary.** Arithmetic feasibility is affirmative for the executed case;
   interactive practicality is negative at the measured cost.

The specific combination appears unclaimed in the literature audit: a released genomic transformer,
all blocks and task head, a task-defined prompt, reference-form nonlinearities at declared client
boundaries, and a decrypted label checked through a documented numerical chain. The manuscript should
say this without an unsupported universal “first” claim.

## Findings that must be fixed

### 1. Separate graph feasibility from numerical generality

One complete prompt establishes that the whole graph can execute, the depth-reset mechanism composes,
and memory does not grow with block index in that run. It does not establish encrypted task accuracy
or input-independent numerical stability.

The limitations section states the one-input scope correctly, but the abstract, Key Points,
Introduction, feasibility verdict, and Conclusion repeatedly say “correctness is not the barrier.”
Use two conclusions instead:

- **graph feasibility:** measured across twelve blocks and the task head at 103 tokens;
- **numerical agreement:** measured for the evaluated prompt, preliminary across the input domain.

A full encrypted test set is unnecessary for this paper. A small panel selected before encrypted
execution across both labels and low/median/high plaintext margins would answer the real concern.
The manuscript must also state how the existing prompt was selected; the repository indicates the
first canonical GSR item, but the paper currently gives no selection rule.

### 2. Repair the numerical-reference chain

The paper says the independent NumPy float64 reference is checked against “the upstream PyTorch
computation” at `2e-5`. The actual gate is against a **float64-cast shadow** of the upstream model.
The released/default Phase-A path is float32 and diverges from the float64 lineage by up to about
`8e-4` in intermediate absolute error after block 6. The final conclusion survives: the float32
path, float64 shadow/manual reference, and encrypted execution all give label `N`; the encrypted
margin differs from the float64 manual margin by `8.56e-9` relatively and from the released float32
margin by about `1.61e-6`.

State the chain as three explicit links:

1. released float32 model versus a float64-cast shadow, including the observed drift and label
   agreement;
2. float64 PyTorch shadow versus independent float64 NumPy reference;
3. encrypted execution versus that NumPy reference.

Also distinguish the task-level float32 reference from the cryptographic float64 reference instead
of calling both “the frozen plaintext reference.” This correction strengthens the methodology
because it makes the precision conversion visible.

### 3. Correct the factual and scope errors

These are small edits but submission-critical:

- `8.56e-9` relative error against a `4e-2` tolerance is **6.67 orders of magnitude**, not nine.
  Use “roughly seven orders below the tolerance” in the manuscript and ledger.
- The client does not evaluate all of LayerNorm. The server computes its statistics under
  encryption; the client evaluates the inverse square root. Replace “exact LayerNorm” with “the
  LayerNorm inverse square root” wherever executor allocation matters, including Figure 3.
- “Well under a second” for plaintext CPU inference is unmeasured. Measure it on a named CPU or
  remove the number.
- The local GSR result uses all 22,604 examples, including data used to train the released head. It
  is release/pipeline-fidelity evidence, not held-out task performance or a canonical test split.
- “Exact nonlinearities” should be defined as the released/reference-form functions evaluated in
  double precision without homomorphic polynomial approximation, not exact real arithmetic.
- “One encrypted lineage” is misleading in a design with decrypt/evaluate/re-encrypt boundaries.
  Prefer “one key lineage with declared client refreshes.”

### 4. Stop inferring the bottleneck from sampled utilization

A median sampled GPU utilization of 32% and maximum of 51% establish underutilization. They do not
prove that the run is not accelerator-bound or that wall time “sits in” host encoding and the
sequential client boundary. Low utilization is compatible with launch overhead, synchronization,
memory bandwidth, small kernels, host-device dependencies, or inefficient GPU kernels.

Report the telemetry descriptively unless a current CPU/CUDA profile attributes the idle periods.
The resource trace remains a strong figure after this wording change.

The paper should likewise avoid saying that batch/offline use remains practical. It defines no batch
workload, cost threshold, or throughput target. The supported verdict is that the measured cost rules
out interactive use; batch suitability remains workload-dependent.

### 5. Add communication complexity

The manuscript counts 10,299 full-model boundary calls but provides no ciphertext count by direction,
serialized bytes, peak message size, sequential dependency phases, or batching analysis. Because the
evaluated roles share one process and one node, a “physical crossing” is not a network message.

At minimum, add a derived table containing:

- ciphertexts sent in each direction by boundary class;
- bytes at the actual levels and parameter set;
- the minimum number of sequential client/server dependency phases;
- which independent tiles can be batched into one transport exchange; and
- the current in-process status.

A simple localhost or LAN transport experiment would improve the paper, but a correct derived
byte/phase bound is the minimum necessary academic treatment. Until real separation exists, describe
the implementation as a single-process emulation of two protocol roles, not an evaluated secure
client/server deployment.

### 6. Make the deployment premise internally complete

The paper motivates a service-only model that the client may not hold, but encrypted evaluation starts
after the client has tokenized the sequence and constructed model embeddings. A service-only client
must therefore either receive the tokenizer and embedding table or use a private lookup protocol.
State which assumption the evaluated design makes; keep private lookup as the unresolved production
path.

The deployment comparison table should also contain the actual ordinary service baseline: send the
plaintext genomic input to the provider. Comparing only local inference with client-assisted CKKS
hides the choice the proposed protocol is meant to change.

The evaluated input is a 600-base-pair genomic fragment, not a complete genome. Scope the privacy
motivation accordingly, or explicitly state that repeated windows from a person's genome are the
assumed service workload.

### 7. Treat THOR and MOAI as cross-paper context

THOR's reported `602.26 s` and this paper's `6683 s` are useful context, not a controlled benchmark.
The studies match on reported ring degree, available levels, nominal security, broad model shape,
and accelerator family. They differ in model semantics, prompt length, nonlinear circuits,
bootstrapping, software libraries, CPU path, and potentially the A100 SKU and modulus details.
MOAI's reported 52.8% reduction is verified for its single-A100 comparison, but it must not be mixed
with its separate 256-input-amortized H200 result.

Use absolute cost and interaction to decide this paper's negative practicality verdict. Use THOR and
MOAI to show that the current implementation is not performance-leading. A compact related-work table
would be clearer than repeating the same comparison in the abstract, Key Points, Introduction,
Results, Related Work, Limitations, and Conclusion.

The literature audit also recommends Powerformer and THE-X for positioning, and Li--Micciancio for
the CKKS decryption/noise-flooding discussion. ELLMo is useful if the venue expects coverage through
mid-2026. Correct the MOAI and DNA-embedding-inversion bibliography metadata before submission.

### 8. Narrow the model-extraction statement

The claim that intermediate activations turn model recovery into “inexpensive per-segment
regression” is uncited and unmeasured. Some affine segments may become directly identifiable given
enough diverse full-rank queries, but attention exposes composite relationships and factorization
ambiguities. No extraction experiment or sample-complexity analysis is presented.

The paper can safely say that the transcript creates a plausible chosen-query extraction surface and
makes at least some affine segments identifiable; model confidentiality is therefore not claimed.
That conclusion is sufficient and does not require a stronger universal recovery claim.

## Highest-return additional evidence

These experiments would materially change reviewer confidence. They are ordered by value, not by
ambition.

| Work | Minimum useful form | What it closes |
|---|---|---|
| Complete-model replication and prompt panel | At least one exact repeat plus two or three prompts fixed in advance across labels and plaintext margins. Use `n>=3` repeats of the same prompt before reporting a mean or variance. | Isolated timing sample; one-input numerical overgeneralization; selection concern. |
| Controlled cache comparison and profile | Same node, input, circuit, and parameters for pre-cache and retained one-block paths; preferably three repetitions each; one current CPU/CUDA profile. | Gives Section 6 a measured payoff and identifies, rather than infers, the bottleneck. |
| Communication accounting | Ciphertexts, bytes, dependency phases, batchable calls; optional localhost/LAN measurement. | Makes the interactive protocol evaluable as a system. |
| Plaintext/client baseline | Named CPU forward-pass time plus client-only CPU and peak RAM. | Grounds the local-inference objection and laptop/client claim. |

If compute is scarce, prioritize a repeat, a small prompt panel, and communication accounting. The
paired cache experiment is the next best use of GPU time. Do not spend the next cycle adding more
plaintext datasets or a third cryptographic architecture.

## Recommended contribution statement

Replace the current five contributions with three:

1. **Complete-model feasibility:** a released twelve-block genomic transformer and task head execute
   at the full 103-token task prompt from encrypted embedded vectors under client-assisted CKKS,
   without homomorphic bootstrap; the evaluated prompt reproduces the reference label.
2. **Reproducible model-faithful evaluation:** the paper supplies the protocol, packing, operation
   schedule, predeclared acceptance rule, corrected float32-to-float64 validation chain, and immutable
   evidence supporting the complete execution.
3. **Measured systems anatomy and scoped negative results:** the paper reports time, client/server
   work, interaction, depth, memory across twelve blocks, the stage-bounded cache needed for
   executable host-side reuse, and the memory boundary reached by the evaluated non-interactive
   configuration.

A defensible thesis sentence is:

> Client-assisted CKKS can execute a complete released genomic transformer from encrypted embedded
> vectors at its real task prompt, with reference-form nonlinearities evaluated at declared client
> boundaries and no homomorphic bootstrap; the executed case is arithmetically feasible and
> reproducible, but the current single-process interactive implementation is neither practical for
> interactive use nor performance-leading.

## Structure, length, and presentation

The macro-order is sound: plaintext reference before protocol, and Related Work before Limitations.
Do not reorganize the paper wholesale.

High-value edits are:

- add an **Evaluation setup and endpoints** subsection naming input selection, hardware, CPU/RAM,
  software versions, thread count, timing boundaries, sample count, and contention qualification;
- shorten the roughly 370-word abstract and eight Key Points to four or five points;
- keep one brief THOR concession in the Introduction and the detailed comparison in Related Work;
- remove repeated single-run, token-lookup, tolerance, bootstrap, and practicality caveats from every
  summary layer, leaving each in one main home plus the abstract when needed;
- remove Figure 2 if space is limited; Table 2 preserves the comparator qualifications better;
- keep the top-level block and boundary algorithms in the main text and move lower-level primitives
  to an appendix if needed.

The compiled PDF has a submission-blocking float failure: Algorithms 1--4 appear on pages 24--25,
after the bibliography, with large blank areas. Figure 3 assigns the second LayerNorm to the wrong
party, Figure 7 clips the client label, and Figure 1 implies that the genome/token lookup enters the
encrypted boundary. These need correction and a complete PDF recheck. Most figure annotations also
land near 6-point type at their placed size.

## Venue fit

The best fit is an **applied HE, privacy-engineering, or systems-evaluation venue**. That audience can
value the released model, complete execution, fixed operation schedule, memory mechanisms, and honest
negative practicality result.

- A cryptography or security-theory venue would expect a stronger novelty claim, formal security
  treatment, malicious behavior analysis, and real party separation.
- A mainstream systems venue would expect repeated experiments, a profile, controlled ablations, and
  communication measurements.
- A bioinformatics-method venue would expect biological novelty or broader encrypted task-level
  evaluation; additional plaintext metrics alone would not supply that.

Select the target audience before the final rewrite, because venue choice determines whether the
remaining security proof, network implementation, or biological validation is required.

## Non-perfectionist revision plan

Do this and stop:

1. Correct the headroom arithmetic, float32/float64 reference chain, LayerNorm allocation, GSR split
   wording, cross-paper comparison, GPU-utilization interpretation, and model-extraction claim.
2. State the prompt-selection and embedding-table assumptions; add the ordinary plaintext-service
   baseline to the deployment table.
3. Add a derived communication table.
4. Repeat the complete run at least once and add a few label/margin-stratified prompts, or narrow the
   correctness headline if compute is unavailable.
5. Run one controlled pre-cache/current single-block comparison with a current profile if the paper
   will retain the systems-work section as a principal contribution.
6. Reframe the contributions, compress repeated concessions, correct the bibliography, fix the
   algorithm/figure layout, and add the missing license and author metadata.

Do not add another plaintext task, another model, another cryptographic scheme, or a large encrypted
benchmark unless the chosen venue explicitly demands it. Those would increase scope without closing
the objections that currently determine acceptance.

## Review basis

This synthesis integrates the independent dated audits of evidence, sources, structure, style, and
figures in `paper-docs/reviews/`. The evidence and source passes verified the central full-model
artifact and current bibliography against primary records. The manuscript and figures were not
edited during this review.
