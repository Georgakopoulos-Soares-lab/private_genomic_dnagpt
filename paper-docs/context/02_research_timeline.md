# Research timeline

> **Status supersession, 2026-08-14.** All twelve released blocks plus the GSR head have now run
> once at 103 tokens on a whole-node allocation with no contention detected in recorded telemetry,
> matched the recorded label, and completed in 6683 s. Statements below that call this execution or
> its final label unmeasured are historical.
> Independent repetition, encrypted task-set accuracy, profiling, private lookup, and network
> transport remain open.

> **Superseded on timing — read this first.**
>
> This note predates the accepted complete execution. Any latency, wall-clock, or server/client
> timing figure below has been replaced by the complete-run measurements in
> [`../evidence/measurements.yaml`](../evidence/measurements.yaml) and
> [`../evidence/optimizations.yaml`](../evidence/optimizations.yaml): all twelve blocks and the
> task head took **6683 s** in one execution. No paired pre-cache/current speedup is reportable.
>
> The older figures here were measured on a contended shared host and **do not go in the
> manuscript in any form** — not as results and not as caveats. What remains valid in this note
> is everything contention cannot affect: the protocol, the threat model, the argument structure,
> the operation counts, the depth, the memory footprint, and the claim discipline.
>
> When this note and the evidence ledger disagree about a number, the ledger wins.

The project did not begin with a claim that DNAGPT could run privately. It began with two separate
questions: whether the released model was worth preserving, and whether its arithmetic could be
evaluated without exposing genomic inputs to an untrusted compute provider. The work answered these
questions in that order.

This note follows the scientific progression. It retains failures only when they changed the method
or the strength of the conclusion.

## 1. Establish a meaningful plaintext target

Encrypted agreement is not useful if the underlying model or evaluation harness is wrong. The first
stage therefore reproduced DNAGPT on three genomic task families using the released
0.1-billion-parameter models.

- Human AATAAA polyadenylation-signal recognition used the released classification head on the full
  balanced DeepGSR set.
- Human mRNA abundance used the released regression head and a reconstructed Xpresso test split.
- Human promoter and splice-site evaluation used three GUE datasets. DNAGPT did not release a GUE
  head, so the backbone and a linear classification head were fine-tuned locally.

All three task families produced useful, reference-near results. More importantly for encrypted
evaluation, the harnesses froze per-example plaintext predictions. Those predictions—not a rounded
paper metric—became the acceptance oracle for later arithmetic.

This stage also exposed limitations that remain part of the paper story. The Xpresso source server
was unavailable and its data had to be recovered from the Internet Archive. The preprocessing method
was reproduced, but the identity of the historical 1,000-gene test split is not proven across pandas
versions. GUE covers only three of the benchmark's 28 datasets, and its comparison values are
approximate literature references rather than a new controlled head-to-head study.

## 2. Select an arithmetic and software path

DNAGPT blocks combine large linear maps with LayerNorm, causal softmax attention, GELU, and residual
connections. This makes the choice of encrypted arithmetic consequential.

CKKS was selected because it directly represents approximate real and complex arithmetic, supports
SIMD packing and rotations, and maps naturally to the transformer's dominant dense linear algebra.
OpenFHE provided the correctness path: explicit depth management, polynomial evaluation, rotations,
and approximate bootstrapping in one context. FIDESlib supplied a CUDA implementation and OpenFHE
interoperability for the performance path.

Other evaluated software did not offer a better route for this graph. TenSEAL lacked the exposed
bootstrap and general nonlinear-function support needed for repeated blocks. The evaluated Concrete
ML path used a client/server split similar to the eventual protocol, but its TFHE-rs arithmetic and
reported GPU/ciphertext characteristics were less suitable for this real-valued, matrix-heavy target.
This was a backend decision, not a rejection of client assistance itself. CKKS-to-FHEW switching
remained a plausible non-interactive research direction, but no integrated GPU implementation was
available to extend.

## 3. Prove the operation vocabulary

Reduced OpenFHE experiments established that the required operation classes were expressible before
introducing real width and weights.

- Baby-step/giant-step diagonal matrix multiplication passed and reduced the number of rotations.
- Polynomial LayerNorm, causal softmax, and GELU passed a fixed relative-error gate.
- Approximate bootstrapping refreshed an exhausted ciphertext and enabled further multiplication.
- A complete two-head toy transformer block passed in one ciphertext lineage with no intermediate
  decryption.
- The same toy graph passed through the CUDA implementation.

These results proved arithmetic closure at a reduced shape. They did not establish realistic latency,
model-width memory, sequence-length scaling, or a private embedding lookup.

## 4. Close one real-weight block without client assistance

The next stage used released block-0 weights at hidden width 768 and 12 attention heads. LayerNorm and
attention subgraphs passed first. An initial full-block schedule then exhausted its available depth
before the MLP completed.

At two tokens, two-way softmax can be written as a sigmoid of the score difference. Substituting that
exact identity removed one expensive reciprocal approximation and allowed the complete block to pass
under pure non-interactive CKKS. The result established that one real-weight DNAGPT block could remain
in one encrypted lineage through LayerNorm, attention, GELU, the MLP, residuals, and final packing.

This was a deliberately small causal-attention graph. Two tokens are enough to test a nontrivial
causal branch, but they are not a valid GSR input and cannot support a task prediction.

## 5. Encounter the pure-CKKS composition wall

The pure path required polynomial approximations for inverse square root, sigmoid or softmax, and
GELU. Repeating those approximations across blocks consumed most of the multiplicative depth. A
bootstrap after the first block therefore had to restore enough depth for the next complete block,
which enlarged the cryptographic context, evaluation keys, and bootstrap data.

Chained-composition attempts spanning multiplicative depths 50, 58, and 64 failed during GPU context
loading. The common failure path was allocation of rotation-key or bootstrap-precomputation data; the
required context did not fit one 80 GB A100. A native two-GPU attempt failed inside a less-tested
multi-device setup path rather than completing the load.

This is a scoped systems result, not a proof that pure CKKS can never evaluate DNAGPT. It did show
that the chosen implementation and packing could not compose blocks on the available hardware. More
importantly, it identified the mechanism: encrypted nonlinear approximations drove depth, and depth
drove the memory footprint.

The pure path was frozen as a non-interactive baseline. Its successful block and failed composition
remain useful because they explain why the main protocol became interactive.

## 6. Move exact nonlinearities to the data owner

The new protocol kept all dense linear algebra, rotations, multiplications, and residual additions
under CKKS on the GPU server. At fixed LayerNorm, attention-softmax, and GELU boundaries, the
ciphertext returned to the data owner's client. The client decrypted its own intermediate, evaluated
the exact function, and re-encrypted the result under the same context and key lineage.

This removed polynomial approximation domains and most nonlinear depth pressure. The server still
never received plaintext or the secret key, but the protocol now required an online client and could
no longer be described as non-interactive.

A complete real-weight two-token block passed with error near machine precision relative to the
plaintext oracle. The smaller depth requirement also reduced the ring dimension relative to the pure
baseline. Batching many logical nonlinearities into one ciphertext reduced the first implementation's
24 physical client crossings to seven without changing the functions being evaluated.

## 7. Generalize attention beyond two tokens

The two-token sigmoid identity was not a general attention algorithm. The attention path was
rewritten to compute exact causal softmax by row at the client boundary.

The general circuit first passed at three tokens, then at eight. The eight-token run exposed a
hard-coded output-repacking assumption that corrupted memory once the token count exceeded four. The
packing step was repaired for arbitrary groups rather than patched for a single length.

A second ceiling came from fitting every head's causal scores into one ciphertext layout. Spreading
heads across four copies raised the ceiling and enabled a complete 32-token block. Reaching the full
GSR prompt required a more general solution: a two-phase chunked softmax reduced each long causal row
in fixed-width chunks, then reused the client's row statistics to emit the encrypted weighting data.
This removed the fixed sequence-length ceiling and passed at 103 tokens.

## 8. Pack several tokens into each ciphertext

The first general-attention implementation still repeated the same dense map once per token. A
token-SIMD layout packed eight logical tokens into disjoint lanes of each ciphertext. The layout
reduced one 103-token linear micro-gate from 103 matrix products to 13 and preserved the plaintext
result.

The same layout was then carried through LayerNorm, Q/K/V, chunked causal attention, attention
projection, the second normalization, GELU, the full MLP, residuals, and output packing. One complete
real-weight block passed at the entire 103-token GSR prompt length. It required 156 dense products,
compared with 1,236 products for the serial-equivalent schedule.

The measured event ratio for the linear micro-gate was close to the structural reduction, but both
runs occurred on a shared host under different co-tenancy. The exact operation-count reduction is a
defensible result; the observed wall-clock ratio is directional, not a clean speedup benchmark.

## 9. Prove short composition and reduce depth

A two-block experiment at two tokens added an explicit block-boundary client refresh. The client
decrypted and unpacked the first block's hidden state, then re-encrypted fresh level-zero token states
for the next block. Both released blocks passed, and device-wide GPU memory used during the run did not increase in the
second block. This established the composition mechanism, but only at the shortest nontrivial length.

The 103-token complete block initially used a conservative multiplicative depth of 16. A controlled
search found that depth 13 was the minimum passing setting for the current graph, packing, and scale.
Smaller settings either failed context construction, ran out of precision at the deepest attention
tile, or stopped at an explicit MLP depth guard.

The first depth-13 sample appeared faster than depth 16. Two repeats produced the opposite result.
Target-process memory remained stable, while wall time tracked extreme host-wide CPU contention. The
speed claim was retracted. Depth 13 remains the smallest demonstrated parameter choice, but it does
not currently carry a validated speed advantage.

## 10. Test optimizations and multi-GPU boundaries

Several optimizations produced useful positive or negative results.

- A CPU-side cache of reusable diagonal vectors passed, reused more than 99.99% of lookups, and added
  no GPU allocations. Its timing benefit remains unresolved because cached and uncached
  samples ran under different host load.
- Reusing GPU-resident plaintext objects was abandoned after six reproducible crashes, bracketed by
  passing controls. The safe CPU-vector cache replaced it.
- FIDESlib's native batched linear-transform primitive passed correctness at one call site but ran
  roughly 1.6–1.9 times slower than equivalent manual call sites in the same run, so it was not
  propagated through the block.
- A warm-up prelude preserved correctness, but shared-host contention made the timing hypothesis
  inconclusive.
- Splitting Q/K/V projections across two physical GPUs passed. One clean, unrepeated sample showed
  about 1.65 times lower wall time for that projection substage.
- Combining the CPU-side cache with the two-GPU Q/K/V split also passed correctness. Its wall time was
  explicitly treated as informational because the host was busy.
- Splitting MLP partial sums across processes could not be built with the pinned library because it
  cannot serialize ciphertexts between processes. The merge mathematics passed offline; transport,
  not algebra, was the blocker.

## 11. Reach the current boundary

On 2026-08-12 the retained driver executed all twelve released blocks and the released GSR head at
103 tokens. The selected prompt reproduced the recorded label, and the decrypted classification
margin differed from the NumPy float64 reference by `8.56e-9` relatively. The run took `6683 s`
(`1.86 h`) on a whole-node allocation with no contention detected in recorded telemetry. It used
one key lineage with declared client refreshes and no homomorphic bootstrap.

This closes complete-model graph feasibility and one-input numerical agreement. It does not close
encrypted task accuracy, input-domain numerical generality, run-to-run variance, private token
lookup, serialized client/server transport, or production security. The task-length block also lacks
a current causal CPU/CUDA profile and a controlled pre-cache/current comparison.

The paper must leave four boundaries visible:

1. the numerical result covers one selected prompt and one complete execution;
2. token-index embedding lookup remains outside the encrypted graph;
3. the two parties are roles in one process, so serialized byte volume and network latency are
   unknown; and
4. the measured cost rules out interactive use for the evaluated implementation, while throughput
   and other workload targets remain unmeasured.
