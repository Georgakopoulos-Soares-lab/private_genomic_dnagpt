# Manuscript evidence and defensibility

> **Status supersession, 2026-08-14.** Use the post-run variant. The complete twelve-block model
> plus GSR head executed once at 103 tokens on 2026-08-12, matched the recorded label, and measured
> 6683 s on a whole-node allocation with no contention detected in recorded telemetry. The current
> limits are one prompt, one execution, no network transport, no private lookup, and no
> production-security evaluation.

> **Superseded on timing — read this first.**
>
> This note predates the accepted complete execution. Any latency, wall-clock, or server/client
> timing figure below has been replaced by the complete-run measurements in
> [`../evidence/measurements.yaml`](../evidence/measurements.yaml) and
> [`../evidence/optimizations.yaml`](../evidence/optimizations.yaml): one complete 12-block plus
> task-head execution took **6683 s**. No paired pre-cache/current speedup is reportable.
>
> The older figures here were measured on a contended shared host and **do not go in the
> manuscript in any form** — not as results and not as caveats. What remains valid in this note
> is everything contention cannot affect: the protocol, the threat model, the argument structure,
> the operation counts, the depth, the memory footprint, and the claim discipline.
>
> When this note and the evidence ledger disagree about a number, the ledger wins.

Working document for the paper. It holds three things:

1. the **scope decisions** that are now locked for the manuscript;
2. a **draft of every manuscript section** (Abstract, Introduction, Background, Methods, Security
   Model, Results, Discussion); and
3. **NEXT STEPS X** — the evidence that is decided but not yet produced.

Tagging follows the project convention: `[V]` verified/measured, `[U]` unresolved, `[A]` assumption
or derivation. Every quantitative value below is traceable to `results/runs/` or
`results/hybrid/manifest.yaml`. Re-verify each one against its canonical source immediately before
submission.

---

## 1. Locked scope decisions

| # | Decision | Consequence for the manuscript |
|---|---|---|
| 1 | Separate graph feasibility from numerical generality | Report the measured one-input agreement and state that encrypted task accuracy remains unmeasured |
| 2 | The client-cost objection is answered by a case study, **not** by changing the architecture | New Section 3.1 below is the load-bearing defence; it must appear in Introduction and Discussion |
| 3 | **No interaction ablation.** The client-boundary set is fixed at LayerNorm + causal softmax + GELU | The design is presented as a stated protocol choice with a justification, not as a swept variable |
| 4 | The pure non-interactive CKKS result is a scoped negative baseline | Report the completed block and evaluated 80 GB composition boundary without a universal impossibility or speed comparison |
| 5 | **Sequence structure is in scope** | Report ciphertext groups and causal score tiles, not projected time |
| 6 | The complete run is `n=1` | Per-block values inside that execution are within-run traces, not run-to-run variance |
| 7 | **mRNA is not encrypted.** Encrypted scope is GSR + GUE | mRNA stays in the plaintext-validation table as model-fidelity evidence and is explicitly scoped out of the encrypted claim |
| 8 | `T = 103` is presented as an explicit, task-derived choice covering both encrypted-scope tasks | Justified quantitatively in §3.2 |
| 9 | Historical contended timings remain excluded | The single complete-run timing is reported with its whole-node telemetry qualification |
| 10 | Security is presented as a **threat model** with a semi-honest ("non-determined") adversary at **128-bit classical parameters throughout** | New Security Model section |

---

## 2. Manuscript section drafts

### 2.1 Abstract

Use the measured post-run result:

> We evaluate privacy-preserving inference for the released 0.1-billion-parameter DNAGPT model.
> A pure non-interactive CKKS configuration completes one released-weight block but exceeds the
> tested 80 GB accelerator-memory envelope when configured for composition. A client-assisted
> protocol keeps linear algebra encrypted while the data owner evaluates the LayerNorm inverse
> square root, causal softmax, GELU, and declared refreshes. All twelve released blocks and the task
> head execute at the 103-token GSR prompt without homomorphic bootstrap. For the one selected
> prompt, the decrypted label matches the reference and the margin has `8.56e-9` relative error.
> One in-process execution takes `6683 s`; encrypted task accuracy, repetition, private lookup, and
> networked deployment remain unmeasured.

---

### 2.2 Introduction

**Paragraph 1 — why genomic data is a distinct privacy problem.**
Set up the asymmetry that motivates everything. A genome is re-identifiable from a few tens of
markers, is not revocable, and leaks information about people who are not party to the transaction.
Unlike a password or a payment token, the harm from disclosure does not decay and cannot be
remediated. Statutory regimes reflect this: genetic data is special-category data under GDPR
Article 9 and is separately regulated in the United States. The result is a class of computations
for which the ordinary cloud-inference deployment is not merely risky but unavailable.

**Paragraph 2 — why a genomic foundation model is the right target.**
Genomic language models moved sequence analysis from hand-designed features to pretrained
transformers, which means the useful artifact is now a large model with fine-tuned task heads rather
than a script a laboratory can reimplement. DNAGPT is a concrete, released, reproducible instance:
0.1-billion-parameter backbone, released classification and regression heads, published downstream
results. It is small enough to be tractable under encryption and real enough that a result about it
is not a result about a toy.

**Paragraph 3 — the technical obstruction.**
CKKS is a natural fit for the dense, real-valued, matrix-heavy part of a transformer and a poor fit
for its nonlinearities. LayerNorm's inverse square root, softmax's exponential and reciprocal, and
GELU all require polynomial approximation, and repeated approximation is what drives multiplicative
depth, bootstrap scheduling, and evaluation-key material. In our own measurements it is the nonlinear
schedule, not the model's dense maps, that first exceeds a single accelerator's memory envelope.

**Paragraph 4 — our protocol, stated as a choice.**
We therefore adopt client-assisted CKKS. The server holds public evaluation material and the model
and performs all encrypted linear algebra. At boundaries fixed before inference, the data owner —
who already holds the query and the secret key — decrypts values derived from its own request,
evaluates the declared reference-form functions in double precision, and returns fresh ciphertexts. The
server never receives the secret key, a plaintext activation, or a partial decryption. We present
this boundary set as a design decision, not as a tuned parameter: it is the smallest set that removes
every polynomial approximation from the encrypted lineage.

**Paragraph 5 — the objection, met head-on.**
A reader will immediately ask why the client, which evaluates three nonlinearity classes anyway, does
not simply evaluate the whole model locally. We answer this quantitatively rather than rhetorically.
Local inference is preferable whenever the client may hold the model. The protocol addresses a
different premise: the model is served but not distributed. The measured client side is CPU-only
and accounts for `14.5%` of encrypted evaluation, but no comparable local CPU forward-pass time was
recorded. Section 3.1 states that comparison directly.

**Paragraph 6 — contributions.**

1. A reproduced plaintext DNAGPT baseline on genomic signal recognition and on promoter and
   splice-site classification, frozen per-example as the acceptance oracle for every encrypted run.
2. A client-assisted CKKS protocol for DNAGPT with a general, chunked, exact causal-softmax boundary
   that removes the token-count ceiling of earlier packing schemes.
3. Complete-model execution of all twelve released blocks and the task head at the 103-token prompt,
   with one-input label agreement, depth 13, and no homomorphic bootstrap.
4. A measured provider/client cost decomposition, derived ciphertext-object and dependency-phase
   accounting, stage-bounded host caching, and explicit limits covering private lookup, network
   transport, and production security.

---

### 2.3 Background

**DNAGPT.** Architecture at the level the encrypted implementation depends on: 6-mer tokenization,
`D = 768`, 12 heads, 12 blocks, pre-norm residual structure, 4× MLP expansion with GELU, released
0.1b weights and fine-tuned heads. State the exact forward path that the encrypted target reproduces
and note that the encrypted target begins at embedded numeric token vectors.

**CKKS.** Approximate arithmetic over packed real vectors; slots, rotations, rescaling, levels, and
multiplicative depth. The three facts the paper leans on: (i) SIMD packing makes dense maps
affordable relative to per-element schemes; (ii) every multiplication consumes level budget, so
circuit depth determines parameter size, key material, and memory; (iii) nonlinear functions must be
approximated, and approximation is the dominant consumer of depth in a transformer block.

**Homomorphic dense linear algebra.** Diagonal encoding with baby-step/giant-step scheduling; tiling
when the matrix exceeds the packing region; hoisted rotations. At `D = 768` the implementation uses a
32-by-32 decomposition. This is the standard machinery and is presented as background, not
contribution.

**Prior private transformer inference.** Position the work against (a) polynomial-approximation
approaches that keep one non-interactive ciphertext lineage, (b) secure multi-party computation for
transformers, and (c) trusted-execution approaches. The relevant axis for us is where nonlinearity is
evaluated and what that costs in depth, memory, and interaction. *Related-work table to be completed
at drafting time.*

**Software path.** OpenFHE supplies CKKS semantics, parameter selection, encoding, and rotation;
FIDESlib supplies the CUDA CKKS operations actually measured; NumPy/PyTorch supply independent
plaintext oracles and are never on the encrypted performance path.

---

### 2.4 Methods

**Plaintext oracle.** Each task harness follows the released DNAGPT forward path and records
per-example outputs. For encrypted experiments an independent NumPy implementation reconstructs the
selected boundary from released weights and is cross-checked against the upstream PyTorch graph. All
encrypted correctness is measured against this frozen output.

**Acceptance criterion.** Global and worst-token relative infinity error,
`max(|encrypted − oracle|) / max(|oracle|)`, both finite and `≤ 4e-2`. The tolerance is fixed in
advance and is never tightened after seeing a result.

**Protocol.** The evaluation alternates server and client phases on a schedule fixed before
inference:

```text
client encrypts embedded token vectors
  -> server: encrypted LayerNorm statistics and dense maps
  -> client: exact normalization boundary, re-encrypt
  -> server: encrypted Q/K/V and attention-score algebra
  -> client: exact chunked causal softmax boundary, re-encrypt
  -> server: encrypted value aggregation, output projection, residual
  -> client: exact second normalization and GELU boundaries, re-encrypt
  -> server: encrypted MLP projection, residual, output packing
  -> client: decrypt result, or refresh the hidden state for the next block
```

No boundary location, count, or size depends on a decrypted value.

**Boundary batching.** One client call may carry packed data for multiple independent evaluations
in disjoint ciphertext regions. This changes the physical schedule, not the model mathematics. At
103 tokens the complete block makes **857 in-process client calls**; attention accounts for 818
(95.4%) and for 818 of 896 boundary ciphertext objects (91.3%). The asserted 129,162
logical-instance counter mixes token, token-copy, and head/query/key-scalar units and is not a
scalar-work or traffic total.

**Chunked exact causal softmax.** For query position `i`, softmax is evaluated over keys `0..i` only.
The server computes encrypted scores and packs them for the client in fixed-width chunks; the client
reduces chunks and applies a numerically stable softmax, returning encrypted weights for the server's
value aggregation. Chunking removes the token-count ceiling of the earlier score-packing scheme
rather than raising it, which is what makes the full 103-token prompt representable.

**Eight-token SIMD packing.** Eight tokens share one ciphertext group so that one encrypted matrix
map serves eight tokens. At `T = 103` this gives 13 token groups and reduces dense encrypted products
from 1,236 serial-equivalent products to **156** (`7.92×`). A four-token width was screened out by an
exhaustive local operation count — worse in every category — without spending GPU time.

**Parameters.** Ring dimension `2^16`, multiplicative depth 13, 3 hybrid digits, `HEStd_128_classic`.
Depth 13 is the smallest setting demonstrated to pass at this scale, packing, and digit count. We
retain it because it passes at identical memory and there is no evidence it is worse; we make **no**
speed claim for it (§2.6).

**Multi-block composition.** Between blocks the client decrypts the packed hidden state, unpacks, and
re-encrypts it as the next block's input. This resets level consumption and prevents the per-block
depth and memory accumulation that otherwise forces a deeper context.

**Hardware and measurement.** The accepted complete execution uses one A100-PCIE-40GB and 32 CPU
threads on a whole-node allocation. Host RSS is process-specific; retained GPU memory samples are
device-wide. Timing and contention qualifications are stated with the complete-run result.

---

### 2.5 Security Model

**Parties.**

- **Data owner (client).** Holds the genomic sequence, the CKKS secret key, and the decrypted values
  returned at declared boundaries. Assumed honest. It learns nothing it did not already possess about
  its own input.
- **Compute provider (server).** Holds public and evaluation key material and the model weights, and
  performs all encrypted arithmetic. It is the party the protocol protects against.

**Adversary.** We assume a **semi-honest, non-determined adversary**: a compute provider that
executes the specified protocol and may inspect anything it legitimately receives — ciphertexts,
public and evaluation keys, message sizes, and timing — but does not deviate from the protocol,
tamper with ciphertexts to probe the client, or mount dedicated cryptanalytic or physical attacks.
This is the standard opportunistic-insider and curious-operator model. It is *not* a malicious-server
model, and we say so explicitly rather than implying broader coverage.

**What is protected.** Under this adversary, the server obtains no plaintext genomic sequence, no
plaintext activation, no partial decryption, and no secret-key material at any point in the
evaluation. Every value it holds is either public model structure or a CKKS ciphertext under a key it
does not possess. All gates are configured at `HEStd_128_classic` — **128-bit classical security
throughout**, with no parameter set in the paper falling below it. The depth reduction to 13 was
taken inside that constraint, not against it.

**What the server does observe.** The sequence length, the model identity and structure, the fixed
boundary schedule, ciphertext sizes, and wall-clock timing. Because the boundary schedule is fixed
before inference and never selected from decrypted content, this view is a function of `T` and the
public model alone, and is identical for any two inputs of the same length. Empirical confirmation of
that invariance is NEXT STEPS X-4.

**Approximate decryption.** CKKS decryption is approximate, and protocols that expose decryption
results to an adversary are known to be attackable. Our structure differs in the way that matters:
the client never returns a decryption to the server. It returns a **freshly encrypted** value, so the
server observes ciphertexts throughout and no approximate plaintext at any boundary. The residual
concern is that the server selects which ciphertext the client decrypts, and does so many times per
inference. Under the semi-honest assumption above this is not exploitable, because the server does
not deviate from the schedule. To reduce reliance on that assumption we identify noise flooding at
re-encryption as the standard mitigation, and note that our measured error (`≈4e-9`) sits roughly
seven orders of magnitude inside the `4e-2` acceptance criterion, leaving substantial headroom to
absorb it. Quantifying the flooding budget is NEXT STEPS X-5.

**Model confidentiality is not claimed.** The client never receives weights explicitly, but the
boundary transcript is linearly sufficient to recover them: at the GELU boundary the client knows the
normalized input it re-encrypted and observes the corresponding pre-activation, giving `T` equations
per output dimension per query, so roughly `⌈D / T⌉ ≈ 8` queries determine the MLP up-projection, and
the attention boundary similarly exposes `W_q W_kᵀ` over many queries `[A]`. Model confidentiality in
this architecture is therefore an operational property enforced by query budgeting and contract, not
a cryptographic one. We state this as a limitation. It does not weaken the data-owner guarantee,
which is the guarantee the deployment premise requires.

**Explicitly out of scope.** Malicious-server behaviour; authenticated transport and key custody;
traffic analysis and access-pattern privacy; timing and physical side channels; compromised clients;
private token-index lookup, which occurs before the encrypted boundary; and privacy from the data
owner, which is not a goal — the client is the subject of the data.

---

### 2.6 Results

**Plaintext baseline (the oracle).** `[V]`

| Task | Local result | Reference | Reading |
|---|---:|---:|---|
| Human AATAAA signal recognition | acc `0.9124`, F1 `0.916`, `n = 22,604` | DeepGSR acc ≈ `0.916` | Released classification head reproduces the reference on the full canonical set |
| GUE core promoter | MCC `0.680`, acc `0.840` | DNABERT-2 MCC ≈ `0.69` | Locally fine-tuned backbone near reference |
| GUE 300-bp promoter | MCC `0.897`, acc `0.948` | DNABERT-2 MCC ≈ `0.87` | Exceeds reference |
| GUE reconstructed splice sites | MCC `0.831`, acc `0.895` | DNABERT-2 MCC ≈ `0.85` | Close to reference |
| Human mRNA abundance *(model fidelity only)* | r² `0.562`, Pearson `0.753`, `n = 1,000` | DNAGPT r² ≈ `0.62` | Retained as fidelity evidence; **not** in encrypted scope |

Required qualifiers, carried in the caption: GUE uses a locally fine-tuned head because no head was
released, and covers three of 28 datasets; the recovered mRNA split is not verified gene-for-gene
against the historical split.

**Encrypted scope.** The encrypted claim covers the two classification tasks — GSR and GUE. `T = 103`
is the maximum tokenized prompt length across both, so a circuit validated at `T = 103` covers every
prompt in the encrypted scope (§3.2). mRNA is deliberately excluded.

**Complete task-length block.** `[V]`

| Property | Measurement |
|---|---:|
| Boundary | released block 0, complete graph |
| Sequence length | 103 tokens (complete GSR prompt) |
| Width / heads / token lanes | 768 / 12 / 8 |
| Dense encrypted products | 156 (vs 1,236 serial-equivalent; `7.92×`) |
| Ciphertext–plaintext multiplications | 177,734 |
| Ciphertext–ciphertext multiplications | 1,506 |
| Explicit rotations / reduction calls | 8,173 / 8,776 |
| Minimum demonstrated multiplicative depth | 13 |
| Physical client crossings | 857 |
| Attention-related client calls | 818 of 857 (`95.4%`) |
| Boundary ciphertext objects | 896 (`91.3%` attention-related) |
| Relative error (global, across depth-13 samples) | `4.1e-9` – `5.0e-9` |
| Acceptance criterion | `4e-2` |
| Peak device-wide GPU memory used during the run | 9,834 MiB (≈ `9.6 GiB`) |

Accuracy is not the constraint at single-block scale: the measured error is roughly seven orders of
magnitude inside the predeclared criterion. The constraint is dense encrypted linear algebra.

**Provider/client cost decomposition.** `[V]` In the single accepted complete execution, encrypted
evaluation takes `6594.748 s`. The residual provider-side bucket is `5638.561 s` (`85.5%`) and
includes GPU-backend work plus host support. Declared client boundaries and refreshes take
`956.187 s` (`14.5%`) on CPU. These are one execution's values, not a distribution or a causal
profile.

**Composition.** `[V]` Released blocks 0 and 1 first composed at two tokens through a
full-hidden-state client refresh, at `8.48e-11` relative error after block 1. The same mechanism then
composed all twelve blocks and the task head at 103 tokens. Device-wide GPU memory used reached
9839 MiB after about 400 s and did not grow with block index.

**Retained optimizations.** `[V]` Boundary batching; general and chunked exact causal softmax;
eight-token SIMD packing; depth 13; a CPU-side diagonal-vector cache (>99.99% hit rate, no added GPU
allocation, correct output, timing benefit unresolved); cross-process context/key reuse; a
two-GPU Q/K/V split (both workers matched the oracle; one clean unrepeated sample showed ≈`1.65×` on
that substage).

**Negative results, reported as such.** `[V]` Reusing GPU-resident plaintext objects crashed
reproducibly six times on six devices at an identical code path, with same-day byte-identical
controls passing — abandoned rather than patched. A library-native batched linear-transform primitive
was correct but `1.6–1.9×` slower than its own hand-written sibling call sites *in the same run*, so
it was not propagated. A warm-up-prelude hypothesis is genuinely unresolved rather than disproven; no
low-contention window was available. A two-process MLP partial-sum split had exact merge mathematics
but could not be built, because the pinned library provides no mechanism to transport a ciphertext
between processes.

**Timing qualification.** `[V]` Historical block timings from a shared host remain excluded. The
complete execution is one `6683 s` sample from a whole-node allocation with no contention detected
in recorded telemetry; scheduler exclusivity was not requested. It supports neither a mean nor a
service rate, and no paired pre-cache/current speedup is reportable.

**Twelve-block closure.** `[V]` All twelve released blocks and the GSR head execute at 103 tokens for
one selected prompt. `[U]` Repetition and multi-input encrypted evaluation remain open.

---

### 2.7 Discussion

**What the result is.** Complete-model graph feasibility is measured across all twelve released
blocks and the task head at 103 tokens. Numerical agreement is measured for one selected prompt,
whose label matches the reference and whose margin has `8.56e-9` relative error. The current
implementation's `6683 s` cost rules out interactive use in the evaluated form.

**On interaction.** The protocol is interactive by construction. Client boundaries and refreshes
account for `956.187 s`, or `14.5%` of encrypted evaluation. Attention contributes 818 of 857 client
calls and 818 of 896 boundary ciphertext objects per block. The evaluated roles share one process,
so these are not network messages and serialized byte volume remains unknown.

**On the local-execution objection.** Developed as a case study in §3.1 and summarized in the
Introduction. In brief: the local baseline is faster in arithmetic and unavailable in deployment,
because it requires the client to hold weights the premise forbids; the architecture offloads
accelerator capital and the model artifact rather than FLOPs; and the client's share is structurally
decreasing in model width `[A]`.

**Limitations, stated without hedging.** The complete-model result covers one prompt and one
execution. Private token-index lookup remains outside the encrypted boundary. The protocol requires
the key-holding client at every declared boundary and has no network transport. The threat model is
semi-honest at 128-bit classical parameters, not malicious-server, side-channel, or traffic-analysis
security. Model confidentiality against the client is operational, not cryptographic. GUE uses a
locally fine-tuned head over three of 28 datasets. Nothing here is a clinical or biological claim.

**Outlook.** Named as directions, not as results: sequence-length scaling behaviour at a fixed
circuit; distinct dense transforms across the four packed copies, derived to cut dense products from
156 to roughly 52 `[A]`; completing LayerNorm at its existing client boundary and fusing inter-block
refresh with the following normalization; segmented head reductions and compact matrix layouts; safe
encoded-weight reuse; and a genuine networked transport experiment, since the current crossings are
in-process cryptographic boundaries rather than RPCs.

---

## 3. Defensibility dossier

### 3.1 The client-cost case study

Local plaintext inference is the preferred architecture whenever the client may hold and run the
model. It avoids all cryptographic work. This study did not measure a comparable local CPU forward
pass, so it makes no numeric local-overhead claim.

The evaluated protocol addresses a served-model premise in which the provider distributes the
tokenizer and embedding table but not the full model weights. Under that premise, the choices are
to disclose the genomic fragment, decline the inference, or use a privacy-preserving protocol. The
public DNAGPT artifact is an auditable surrogate for that deployment; its public availability does
not itself create a need for the service arrangement.

In the measured complete execution, declared client boundaries and refreshes consume `956.187 s`
(`14.5%` of encrypted evaluation) on CPU. The residual provider-side bucket consumes `5638.561 s`
and includes GPU-backend work plus host preparation, synchronization, and evaluator overhead; it is
not GPU-kernel time. The data owner must remain online throughout.

Model confidentiality from the data owner is not claimed. Intermediate activations create a
chosen-query extraction surface, and some affine segments may be identifiable from sufficiently
diverse queries. The protocol's claim is genomic-input confidentiality against the semi-honest
compute provider under the stated boundary.

### 3.2 Why `T = 103`

`T = 103` is not a truncation and not an arbitrary convenience. It is the maximum tokenized prompt
length over the two downstream tasks the paper encrypts, so a circuit validated there covers both.

| Encrypted-scope task | Raw input | 6-mer tokenization | Harness `max_len` |
|---|---:|---:|---:|
| GSR — human AATAAA signal recognition | 600 bp | 100 tokens + specials = **103** | 512 (padding headroom) |
| GUE — core promoter | ~70 bp | ~12 tokens | 32 |
| GUE — 300-bp promoter | 300 bp | 50 tokens | 64 |
| GUE — reconstructed splice sites | 400 bp | ~67 tokens | 96 |

Every GUE prompt is strictly shorter than the GSR prompt. Choosing `T = 103` therefore covers the
encrypted scope in one configuration, and the choice is stated in Methods as deliberate. The
supporting claim in the manuscript is: *"103 tokens is the complete tokenized prompt for the GSR task
and an upper bound on every prompt in the GUE subset we encrypt; a circuit validated at 103 tokens
covers both encrypted-scope tasks."* mRNA is excluded from encrypted scope and is not covered by this
statement.

### 3.3 Editorial flags

**Flag 1 — removing pure non-interactive CKKS leaves the architecture unmotivated.** Decision 4
removes Scheme A from the paper. That correctly removes an unsupportable comparison: the pure path
never composed two blocks inside one accelerator's memory, so it cannot serve as a baseline. But the
*reason* the client-assisted protocol exists is exactly that measured memory wall, and the
Introduction's Paragraph 3 currently asserts "the nonlinear schedule is what exceeds a single
accelerator's memory envelope" without evidence if Scheme A is deleted entirely.

Recommended resolution, reflected in the drafts above: retain **one sentence** of design rationale in
Methods — that a non-interactive configuration of the same graph exceeded the tested 80 GB envelope
during context and evaluation-key loading at two-block composition, which is why nonlinearity was
moved to the client — with **no** Scheme A results table, no timing comparison, and no ablation. This
keeps the architecture motivated by a measured fact while honoring the decision to drop Scheme A as a
result. If you prefer total removal, Paragraph 3 must be softened to a design argument about
approximation depth rather than a claim about memory. Your call; the drafts currently assume the
one-sentence version.

**Flag 2 — mRNA in the paper but not in the encrypted claim.** A reviewer will ask why three tasks
are validated and two are encrypted. The answer in §2.6 is that mRNA is a regression head over a
different, longer input and is retained as model-fidelity evidence only. That is honest and
sufficient, provided the encrypted-scope sentence appears *before* the mRNA row rather than after it.

**Flag 3 — historical contended timings stay out.** The complete-run timing is reported once with
its whole-node telemetry qualification. Shared-host block timings remain repository history and do
not enter the manuscript or a speedup calculation.

---

## 4. NEXT STEPS X

Decided, not yet produced. Ordered by effect on defensibility.

### X-1 — Replication and a small predeclared prompt panel
Status `[U]`. Repeat the accepted prompt enough times to estimate variability, then evaluate a small
panel fixed in advance across both labels and low/median/high plaintext margins. Report label
agreement and margin/activation error per example. This would strengthen numerical generality and
timing confidence; complete-graph feasibility is already closed for the accepted prompt.

### X-2 — Client-cost case study measurements
Status `[U]`. **No GPU time required.** Two missing measurements complete §3.1:

1. Plaintext DNAGPT-0.1b inference wall time at `T = 103` on representative client CPU hardware,
   measured rather than estimated from FLOPs, including model load.
2. Client-side resource profile at the boundary: peak RAM, confirmation that no GPU is used, and
   a breakdown of decrypt, reference-form function evaluation, and re-encryption.

Also record the model artifact size on disk. These turn the §3.1.3 table from partly derived into
fully measured, which is what makes the argument land.

### X-3 — Sequence-length scaling at a fixed circuit
Status `[U]`. The existing `T = 2/3/8/32/103` points each used a *different* attention circuit and do
not form a curve. Run `T ∈ {8, 16, 32, 64, 103}` on the **current** chunked-softmax plus eight-token
SIMD circuit, one variable changed, same fixture, same parameters, same placement.

Report per `T`: encrypted evaluation time with server/client split, dense product count,
ciphertext–plaintext and ciphertext–ciphertext multiplications, rotations, physical crossings,
client calls, boundary ciphertext objects, peak device-wide GPU memory used, and relative error. Expected figure: cost versus
`T` with the quadratic attention term visible against the linear projection term. Note in the caption
that absolute times inherit the shared-host qualification; the *shape* is the result.

### X-4 — Boundary-schedule input-independence check
Status `[U]`. **No GPU time required beyond two short runs.** Execute two different inputs of the same
length and verify that crossing count, crossing order, ciphertext sizes, and the boundary schedule are
byte-identical. This is the evidence for the Security Model claim that the server's view is a function
of `T` and the public model alone. Currently that claim is asserted by construction and unverified.

### X-5 — Noise-flooding budget at re-encryption
Status `[U]`. Determine how many bits of flooding can be added at the client's re-encryption before
error approaches the `4e-2` criterion, and what it costs. Measured error is `≈4e-9`, leaving roughly
seven orders of magnitude of headroom, so the expected answer is "substantial flooding is
approximately free." If so it is a cheap, strong addition to the Security Model and reduces the
protocol's reliance on the semi-honest assumption.

### X-6 — Security-parameter attestation
Status `[U]`. **No GPU time required.** Record the OpenFHE/lattice-estimator output for the exact
final parameter set — ring `2^16`, depth 13, 3 hybrid digits, `HEStd_128_classic` — and include it as
a Methods table or appendix entry. The manuscript claims 128-bit classical security throughout; since
depth was deliberately reduced to 13, the attestation should be explicit rather than inherited from
the library default.

### X-7 — Private token-index lookup cost estimate
Status `[U]`. The largest declared scope hole. Produce a derived cost for encrypted embedding lookup
(one-hot times embedding table under CKKS at DNAGPT's 6-mer vocabulary): ciphertext count, dense
products, depth, and projected memory. This converts an unbounded omission into a bounded `[A]` with
a number attached, which is materially harder to attack. A micro-gate is optional; the derivation is
the deliverable.

---

## 5. Provenance

Every number in this document should be re-checked against its canonical source before submission.

| Claim family | Source |
|---|---|
| Plaintext task metrics | `docs/tasks.md`, `results/shared/manifest.yaml`, `results/runs/gsr_*.json`, `gue_*.json`, `mrna_*.json` |
| Complete `T = 103` block: counts, depth, memory, error | `docs/hybrid/roadmap.md` "Current evidence"; `results/hybrid/manifest.yaml`; `results/runs/fhe_fides_real_d768_t103_block0_simd_full_*.json` |
| Client/server split, six-run statistics | the six `fhe_fides_real_d768_t103_block0_simd_full_*` run JSONs (`timings_seconds.client_boundary_seconds_total`, `.server_linear_algebra_seconds`) |
| Two-block composition | `fhe_fides_real_d768_t2_blocks0_1_scheme_b_two_block_refresh_a100_20260729_v2.json` |
| Retained and rejected optimizations | `docs/hybrid/optimizations_and_combinations_report.md` |
| Timing contamination | `04_results_and_limits.md` "Timing contamination" |
| Claim wording discipline | `05_claims_and_qualifiers.md` |
| Tokenization and prompt lengths | `DNAGPT/test.py` (6-mer tokenizer), `docs/tasks.md`, `docs/data_provenance.md` |
