# Manuscript structural audit — 2026-08-08

Scope: read-only, full-manuscript pre-submission review of the snapshot present on 2026-08-08.
The audit follows `.claude/agents/paper-structure.md`, with the current evidence ledger and
implementation ground truth taking precedence over older planning notes. No manuscript, evidence,
figure, or bibliography file was changed during this pass. The mechanical manuscript checker was
clean for banned terms, numbers, citations, and structure.

## Overall assessment

The paper now has a recognizable and defensible spine: the plaintext target is independently
validated, the protocol follows from the nonlinearity problem, the optimization campaign preserves a
checked circuit, and the Results section separates feasibility from practicality. Section balance is
good. The scenario, protocol, optimization, and results occupy most of the manuscript; background,
plaintext validation, and related work support them rather than competing with them.

Two problems still break the headline argument. First, the summaries merge two distinct memory
failures and thereby attribute the accelerator-memory improvement partly to a host-memory cache.
Second, the central `11.6x` engineering result still depends on a baseline whose wall time is marked
as open provenance in the ledger. Both must be fixed before submission.

## Breaks in the argument

### 1. Critical — the paper collapses two different memory barriers into one mechanism

The evidence establishes two separate findings:

- The non-interactive composition configuration exceeded an 80 GB **GPU** memory envelope while
  loading its deeper context and evaluation keys. Client re-encryption moves this barrier by
  shortening encrypted segments and resetting level consumption.
- The first encode cache reached 61.9 GiB of **host** resident memory. Encoding at the level of use
  and flushing between stages moves this second barrier, leaving a 48.6 GiB host peak.

Section 6 and the Results memory subsection distinguish these mechanisms correctly. The manuscript's
headline summaries do not. The Key Points say that client assistance and a bounded cache together
yield the 9.6 GiB GPU peak (`00_frontmatter.tex:86`); the Introduction similarly says client
re-encryption and the cache moved “that” 80 GB barrier before citing GPU memory
(`01_introduction.tex:68`); and the Conclusion repeats the same causal compression
(`10_conclusion.tex:10`). A host cache is not the mechanism that removed non-interactive context and
evaluation-key residency from the GPU.

**Specific fix:** in the abstract, Key Points, Introduction, and Conclusion, use two clauses with two
resources. Attribute the 80 GB accelerator-memory result to the protocol/depth reset and the 9.6 GiB
GPU footprint to the retained client-assisted circuit. Then state separately that cache bounding
prevented host-side encoding memory from becoming the replacement barrier, with the 48.6 GiB host
peak. The paper's spine can still be “memory is a barrier that design moves,” but the designs and
memories must remain paired correctly.

### 2. Critical, submission-blocking — the engineering-speedup link is not yet auditable

The `11.6x` claim carries the Abstract, Key Points, Introduction, Optimization, Practicality verdict,
and Conclusion. Its optimized endpoint is documented, and the unchanged schedule is strongly
checked. Its denominator is not: `prov.baseline_wall_time` says the 7,560 s baseline is
engineer-attested but absent from the committed walkthrough. The CPU-saturation repeats do not supply
that wall time. Consequently, the argument currently establishes that host encoding was a
bottleneck and that the retained circuit takes 652.545 s, but it does not yet let a reader audit the
factor connecting them.

**Specific fix:** land the recorded dedicated-node baseline with its scope and provenance, or rebase
the headline on a documented measured comparison. If neither is possible, remove the factor and the
26-hour starting projection everywhere; retain the measured endpoint, the paired affinity result,
and the mechanism-backed optimization account. Do not solve this by downgrading the same unsupported
factor to vague prose—the paper explicitly depends on cost responding to engineering.

### 3. High — the evaluated public model and the served-model premise are not reconciled explicitly

The local-inference objection is correctly placed early and stated before the protocol. Its answer,
however, assumes that the data owner cannot receive the model, while the experiment repeatedly and
correctly describes DNAGPT and its weights as released. For this evaluated artifact, local execution
is available in principle. Section 3 says the server-only placement is an operational preference,
but it never states the decisive concession from the defensibility note: public weights were chosen
for reproducibility, and this experiment does not itself require DNAGPT to be served.

That omission makes the motivation look internally inconsistent. It is amplified by the honest
model-confidentiality result: a determined client can recover shallow segments, so the protocol does
not cryptographically enforce the model owner's distribution policy.

**Specific fix:** immediately before `The local-inference objection` in Section 3, state that the
released artifact is an auditable workload surrogate for a served genomic model. Concede that the
evaluated release can be run locally by a client that accepts it; the deployment result applies when
model access is service-only by policy or contract. Then keep the current claim that model secrecy is
not a protocol guarantee. The Introduction's opening scenario should use the same distinction rather
than implying that this released model is unavailable to the client.

### 4. High — the planned complete-model result cannot currently land in one row and one point

The scope table is prepared: its “twelve blocks and task head” row can change status without a table
redesign. The scaling figure is not. It plots **per-block** seconds against task length and only adds
whole-pass projections as text. A measured complete-model pass at the task length is not a comparable
point on that axis. Adding it would require an average-per-block transformation or a new panel, either
of which would weaken or redesign the evidence.

The unresolved status is also repeated in the Abstract, Introduction, Protocol, Results table,
Feasibility verdict, Limitations, Conclusion, and Future Work. A new result would require edits
throughout the paper, contrary to the intended drop-in structure and the rule that a limitation has
one home.

**Specific fix:** design the scaling figure with a separate whole-pass axis/panel that can accept a
measured task-length point while retaining the shorter-task projections. Make the scope table and
Section 7 verdict the authoritative current-status locations. Keep one scope sentence in the Abstract
and the full measurement-boundary statement in Section 8; remove routine restatements from the
Protocol, contribution list, and Future Work. The Conclusion should report the current verdict, but
should not introduce another explanation of the same scope limitation.

### 5. Moderate — the practicality verdict uses a projection before its derivation and exceeds it

Section 7 declares the 17-minute to 2.2-hour practicality range before the Sequence-length scaling
subsection explains how shorter prompts are projected from ciphertext-group and causal-tile counts.
The later ordering asks the verdict to rely on evidence the reader has not yet seen.

The sentence that these costs “can support batch or offline analysis” also outruns the study. No
batch-service target, throughput result, network transport, complete pass, task-head execution, or
encrypted prediction was measured. The evidence licenses “not practical for interactive queries”;
it does not yet establish suitability for an offline service.

**Specific fix:** move Sequence-length scaling before the Feasibility and Practicality verdicts.
Replace the batch/offline assertion with a neutral statement that suitability outside interactive
use depends on application latency and deployment costs not measured here.

## Order and placement

1. **Section 7:** order the subsections as correctness, measured cost/resource trace, memory,
   sequence-length scaling, composition and scope, feasibility, practicality. The two verdicts then
   consume evidence already established in the section.
2. **Section 6:** move Table `tab:optimization` and Figure `fig:waterfall` before `Headroom that
   remains`. The measured campaign should culminate before the section turns to unimplemented work;
   currently the empirical summary arrives after speculative targets.
3. **Section 3:** insert the released-workload/served-deployment distinction between the role
   definition and the local-inference objection. This is the premise on which the objection turns.
4. **Sections 5, 7, and 8:** keep the depth-reset mechanism in Section 5, current evaluation status
   in Section 7, and full scope qualification in Section 8. Avoid repeating complete-model status in
   every one of them.
5. **Related work:** its late placement is acceptable for this systems paper because the Introduction
   already names THOR and Safhire and Section 5 scopes the non-interactive wall before making a
   general claim. No section-level move is needed.

No wholesale section reordering is recommended. The present path—scenario, validated target,
protocol, optimization, results—works.

## Gaps reviewers will ask about

- **What exactly makes the model unavailable locally?** This is an assumed deployment policy, not a
  property of the released DNAGPT artifact or a confidentiality guarantee of the protocol. It needs
  an explicit scope sentence, not a stronger security claim.
- **Can the headline speedup be independently reconstructed?** Not until the baseline wall-time
  provenance is committed or the comparison is changed.
- **What does “practical” mean outside interactive use?** The paper defines the negative interactive
  verdict but supplies no positive latency, throughput, cost, or service-level target for batch use.
- **Will a complete model run fit the existing evidence presentation?** The table will; the scaling
  figure will not until it separates per-block and whole-pass quantities.
- **Does client availability survive real transport?** The manuscript correctly identifies the
  current boundary as in-process and names network transport as future work. This remains an open
  deployment question rather than a hidden omission.
- **Does feasibility extend to a genomic prediction?** The manuscript correctly says no encrypted
  task accuracy or final prediction has been measured. That gap is central and should remain visible
  in the Feasibility verdict until the built driver runs.

## What is working

- The local-inference objection appears before the protocol and is conceded rather than evaded.
- The two-stage verification chain closes the “circuit and reference are wrong together” objection
  before encrypted correctness is reported.
- Section 5 explains why depth is per segment rather than per network and quantifies that softmax is
  effectively the boundary traffic.
- Section 6 treats the operation schedule as a fail-closed invariant, not an assurance. This is the
  strongest support for attributing elapsed-time reduction to systems work.
- The optimization section is large enough to function as a systems contribution, and its retained
  mechanisms are explained rather than listed.
- Feasibility and practicality are separate, labelled verdicts. Complete-model feasibility is not
  silently inferred from one block.
- Model confidentiality is disclaimed with its concrete extraction mechanism in both the threat
  model and limitations.
- Related work now handles the two most dangerous comparisons honestly: Safhire establishes that
  the interaction pattern is prior art, and THOR prevents the evaluated non-interactive memory wall
  from being generalized.
- Section lengths are balanced and within their planned ranges; the central scenario, protocol,
  optimization, and results sections dominate as intended.
