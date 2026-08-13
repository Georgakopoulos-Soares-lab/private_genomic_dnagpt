# Terminology and prose rules

Read this before writing a single sentence of the manuscript. It is enforced mechanically by
`paper-docs/scripts/check_numbers.py`, which fails on any banned term found under
`paper-docs/manuscript/`.

The project accumulated internal shorthand over a year of development. None of it means anything
to a reader, and several terms actively mislead. A reviewer who reads "Scheme B" learns nothing;
a reviewer who reads "client-assisted CKKS" learns the architecture from the name.

## Banned terms and their replacements

| Never write | Write instead |
|---|---|
| Scheme A | non-interactive CKKS, or the non-interactive baseline |
| Scheme B | client-assisted CKKS |
| hybrid scheme, Scheme C | scheme switching between CKKS and a Boolean scheme |
| T123, T123 v2/v3/v4 | the named optimization (encode caching, host-memory bounding, …) |
| v0, v1, v2, v3 | the configuration being compared, named by what it does |
| config1, config2, config3 | packed evaluation, process sharding, the combined configuration |
| Brev, TACC, Lonestar6, ls6 | one NVIDIA A100 GPU node with 32 CPU cores |
| SLURM, sbatch, job 3341635 | (omit entirely — scheduling is not a result) |
| fixture, oracle hash, contract SHA | the frozen plaintext reference |
| gate, gate token, manifest row, run tag | (omit entirely — repository bookkeeping) |
| cpudiagcache, simd_full_t103, driver name | the evaluated implementation |
| FIDESlib commit, patch name | the evaluated GPU backend |
| Kimon, Christos, engineer names | (authors go on the title page, nowhere else) |

Host names, scheduler identifiers, file hashes, and run tags never appear in the manuscript. They
exist so the repository can prove a result to itself. They do not survive into a paper.

## Terms of art, used precisely

**Client-assisted CKKS.** The compute server evaluates every linear operation on ciphertexts and
never holds the secret key or any plaintext. At three pre-declared points — LayerNorm, the causal
attention nonlinearity, and GELU — the data owner, who already holds both the query and the key,
decrypts values derived from its own input, evaluates the function exactly, and returns a freshly
encrypted result.

**Client boundary crossing.** One physical decrypt-evaluate-encrypt operation. Distinct from a
**logical nonlinear instance**, which is one individual value evaluated. One crossing batches many
instances: the measured block has 857 crossings and 129,162 instances. Never quote one number as
the other, and never write "round trip" — it implies a network operation, and the boundary in the
evaluated implementation is in-process.

**Task-length block.** One transformer block evaluated at the full 103-token prompt. This is the
paper's measured unit. It is not the twelve-block classifier, and a sentence that presents it *as*
a measurement of the whole model is wrong. A labelled projection from it to the whole model is
fine, and expected — see the projection rule below.

**Frozen plaintext reference.** The per-example plaintext prediction, computed once before any
encrypted work, that the encrypted output must reproduce. Not "ground truth" — it is the model's
own answer, not the biological one.

**Non-interactive design.** The configuration that approximates every nonlinearity polynomially
and never decrypts an intermediate. Since the paper asks where encrypted inference stops, this is a
*finding* and gets its own subsection: it completed one real-weight block, and exceeded the tested
80 GB GPU memory envelope when configured for multi-block composition, because repeated polynomial
approximation forces a deeper multiplicative budget and a bootstrapping schedule whose key material
must stay resident. Report the mechanism, which is the transferable part, and scope the result to
the evaluated library, parameters, packing, and hardware. No timing table — none was measured on a
dedicated host, and a correctness-and-memory finding does not need one.

## Claim discipline

These rules exist because each corresponds to a specific way this work could be overstated.

**Operation counts are not latency.** Eight-token packing reduces dense products from 1,236 to
156. That is a 7.92× *structural reduction*. It is not a 7.92× speedup and must never be written
as one.

**Projections are legitimate; unlabelled ones are not.** Twelve times a measured block time is a
projection at fixed circuit. Tag it `[A]`, call it a projection in the sentence where it appears,
and then use it freely — including in the abstract, because the reader's question is what a whole
inference costs. What is forbidden is presenting it as a measurement, or restating the caveat so
often that the result disappears behind it. The measured scope is stated once, in the limitations.

**Depth 13 is minimal, not fast.** It is the smallest demonstrated passing depth at the retained
scale, packing, and digit count. There is no measured speed advantage over other depths, and an
earlier claim to that effect was tested by repetition and withdrawn.

**The server learns nothing is too strong.** Write: the compute server receives no plaintext
activation and no secret key under the stated boundary. Network metadata, traffic analysis, and
side channels were not analyzed.

**Model confidentiality is not claimed.** The data owner observes full intermediate activations
at every nonlinearity, which decouples the network into shallow segments whose nonlinearities are
already known — turning a hard global inversion into cheap per-segment regression. What the
protocol offers the model owner is basic hiding of the weights that raises the cost of casual
copying; it does not stop a determined adversary. State this in the threat model and repeat the
consequence in the limitations. Conceding it makes the paper stronger, not weaker.

**Encrypted scope starts at embedded vectors.** Token-index lookup into the embedding table
happens before the encrypted boundary. Never write "end-to-end encrypted DNAGPT inference."

**The client is not protected from itself.** The data owner sees its own intermediates. That is
not a leak; it holds the key and the query. The privacy objective is confidentiality against the
compute provider.

**Contended timing does not enter the manuscript.** Every reported time comes from the
dedicated-node runs. Measurements taken while other tenants were competing for the same host are
not results, and writing them up as caveated results would be worse than omitting them. They stay
in the repository, and they may still be cited for quantities contention cannot touch — relative
error, operation counts, memory footprint, multiplicative depth.

## Prose style

Follow `~/.agents/policies/writing-style.md`. Three additions specific to this paper, distilled
from the review of the group's previous submission:

**State a caveat once.** The previous paper's review found the same qualifier restated in the
graphical abstract, the abstract, the key points, the introduction, two body sections, the
limitations, and the conclusion. When everything is hedged, the two caveats that actually matter
stop standing out. Each limitation gets one home — normally §3 for scope or §8 for limits — plus
at most one sentence in the abstract if it changes how a headline number may be read.

**Vary paragraph shape.** The same review counted roughly thirty paragraphs ending in an
identically shaped disclaimer of the form "X is A, not B." Two or three such sentences are
emphasis. Thirty is a tic, and it is the single most recognizable signature of machine-revised
prose.

**Keep the domain vocabulary.** Use the terms the cited literature uses. Do not substitute
"calculation" for "computation," do not rename the threat model to "attacker considered," do not
gloss "ciphertext-plaintext multiplication" into something friendlier. Define a term once on first
use and then use it.
