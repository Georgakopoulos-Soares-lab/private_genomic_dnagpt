# The deployment scenario

This is the spine of the paper. Every design decision, every measurement, and every limitation is
read against it. Write §3 of the manuscript from this note, and make sure §1 sets it up.

## The scenario

**The data owner** holds a human genomic sequence. It will not transmit that sequence in the
clear: it is special-category personal data under GDPR Article 9 and its equivalents, it is not
revocable once disclosed, and re-identification from genomic data is a live rather than
theoretical risk. It has ordinary CPU compute — a capable laptop, a workstation, or an allocation
on a shared cluster — and **no GPU**.

**The model owner** runs a small but capable domain model on GPU infrastructure and serves it
rather than shipping it. Keeping the weights on the server is an operational preference, and the
paper must say only that.

The asymmetry that makes this a cryptographic problem is one-sided and should be stated that way:
**the genome is protected; the model is not.**

## Do not claim model confidentiality

This is the most important correction in this note, and it must survive into the manuscript.

An earlier framing justified the architecture by the model owner's intellectual property. That
framing is not formally sound, and a competent reviewer will say so, so the paper says it first.

The mechanism is specific and worth stating precisely, because it is stronger than the usual
hand-wave about black-box extraction:

> The data owner observes the full intermediate activations at every nonlinearity. That decouples
> the network into shallow segments whose nonlinearities are already known. The hard global
> inversion problem collapses into cheap per-segment regression.

So an honest statement of what the protocol gives the model owner is: **basic hiding of the
weights that raises the cost of casual copying and does not stop a determined adversary.** Weight
extraction is a per-segment fitting problem, not a research problem. Say that in §3 and repeat the
consequence in the limitations, and the paper is stronger for having conceded it.

What survives is the reason that actually holds: the data owner has no GPU, will not disclose the
genome, and the model owner has the hardware. That is a sufficient and defensible motivation.

## Why the obvious objection does not apply

The strongest reviewer objection is easy to state, and the paper must state it first, at full
strength, before answering it.

> DNAGPT-0.1b at 103 tokens is roughly 2 × 10¹⁰ floating-point operations. That is well under a
> second of ordinary CPU inference. The protocol spends about 120 seconds of client CPU per block
> and roughly 27 minutes across twelve blocks. The client is therefore doing on the order of a
> thousand times more work inside the protocol than it would need to just run the model itself.

Every number in that objection is correct, and the paper should concede it in those words.

The answer is not that the arithmetic is wrong. It is that "just run the model yourself" assumes
the client is handed the model, and the deployment being studied serves the model rather than
distributing it. Against the alternatives that actually exist for such a client — send the genome
in the clear, or do not run the model at all — the comparison is:

| | Local inference | Send the genome | Client-assisted CKKS |
|---|---|---|---|
| Client needs the model | yes | no | no |
| Client needs a GPU | yes | no | **no** |
| Server sees the genome | not applicable | **yes** | no |
| Client compute per block | sub-second | none | ~121 s, CPU |
| Client peak memory | model-sized | none | ~14 GB |
| Server compute per block | not applicable | sub-second | ~532 s, one A100 |

The middle column is what the protocol competes with in practice, and the row that decides it is
the third.

The measured client side is the deployment claim, and it is modest: roughly 14 GB at peak, no GPU
at any point, under a fifth of the wall clock. That runs on a laptop.

## The direction the cost moves

The client's share is not a fixed tax. It falls as the model gets wider.

The client's work at each boundary is proportional to the number of activation values crossing
it, which scales with the hidden width `D` and the sequence length `T`. The server's work is
dominated by dense transforms, which scale with `D²`. The client's share is therefore
`O(1/D)`.

DNAGPT-0.1b has `D = 768`. It is the *narrowest* model in the family, so the measured 18 percent
client share is the least favourable case, not a representative one. On a wider model the same
protocol pushes proportionally more of the work onto the encrypted server side, which is the
direction the architecture wants to move. This is a derivation, not a measurement — tag it `[A]`
and present it as a scaling argument.

## What the resource trace adds

The measured GPU utilization never exceeds 40 percent, and collapses to 1–2 percent for about 90
seconds while the client evaluates attention weights. That is not a defect to apologize for; it
is the empirical shape of the architecture. The encrypted linear algebra is not the saturating
cost at this scale, and the sequential client boundary is a real, identified, addressable
bottleneck rather than an inherent one — parallelizing it was blocked by a backend limitation,
not by the design.

## What this scenario does not cover

State these in §3, once, plainly:

- The compute server is assumed **semi-honest**: it follows the protocol and tries to learn from
  what it sees. A server that deviates arbitrarily is out of scope.
- **Model confidentiality is not claimed** — see the section above. The protocol raises the cost
  of casually copying the weights and does not stop a determined adversary.
- The client learns its own intermediates. It holds the key and the query; this is not a leak.
- Authenticated transport, key custody, traffic analysis, timing and physical side channels,
  compromised clients, and private token-index lookup are all outside the evaluated boundary.

## One-paragraph version, for the introduction

> Genomic sequence is the hardest case for cloud inference: it identifies a person, it cannot be
> revoked once disclosed, and it is exactly the data that domain foundation models are most useful
> on. The obvious answer, run the model locally, assumes the client is handed the model and has
> the hardware to run it; a served model and a client with no GPU satisfy neither assumption. What
> is left is a choice between sending the sequence in the clear and computing on it encrypted.
> This paper measures what the second option costs for a real genomic transformer, and finds it is
> an engineering cost rather than a cryptographic one: a systems campaign that changes no
> arithmetic makes one complete transformer block at full task prompt length roughly twelve times
> faster, while the data owner's side stays on a laptop.
