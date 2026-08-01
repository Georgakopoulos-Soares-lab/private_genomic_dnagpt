# Client-assisted CKKS optimization history and remaining opportunities

This document summarizes the engineering history of the client-assisted CKKS
architecture developed to run a genomics transformer model
under fully homomorphic encryption (FHE) — a form of encryption that allows a compute
provider to run a computation directly on encrypted data without ever seeing the
underlying plaintext. In this architecture, the untrusted server performs all linear
algebra (matrix multiplications, additions) on ciphertexts using a GPU-accelerated
encrypted-arithmetic library, while the data owner (the only party holding the
decryption key) is asked, at a small number of pre-declared points in the computation
where an exact nonlinear function is needed (such as the normalization step or the
activation function between neural-network layers), to decrypt only that quantity,
compute the exact function on its own data in plaintext, and re-encrypt the result
before returning it to the server. Each such exchange is called a client boundary
crossing, or round trip. The server never observes plaintext, a partial decryption,
or the secret key at any point.

All work below was measured against a released, pretrained genomics transformer's
real weights and a real held-out example, on real GPU hardware (a data-center-class
GPU with roughly 80 gigabytes of memory), unless explicitly marked otherwise as a
local, GPU-free contract check. "Passing" means the encrypted computation's final
output matched the unencrypted (plaintext) reference computation to within an
explicit, unchanged error tolerance — in every case reported below, the actual
measured discrepancy was several orders of magnitude tighter than that tolerance,
typically differing from the reference by less than one part in a hundred million to
one part in a billion. Where a timing number is reported, its trustworthiness is
qualified explicitly, because nearly all of this work was performed on a shared
cloud host whose other tenants' activity fluctuated dramatically and, at times,
dominated the measured wall-clock time. A number of engineering configurations use
a compact packing scheme in which several encrypted values are placed in disjoint
regions of a single unit of ciphertext ("packing"), and a technique referred to
below as diagonal-batched matrix multiplication, a standard method for multiplying
an encrypted vector by a plaintext matrix one diagonal at a time; both are treated
as background machinery rather than re-explained at each mention.

## 1. What was tried, and what worked

### Optimizations that passed correctness and were kept

The foundational architectural choice — replacing calibrated polynomial
approximations of the normalization, attention, and activation nonlinearities with
exact client-side computation at a decrypt/compute/re-encrypt boundary — was the
first optimization tried and is the one that made the rest of this program possible.
It was measured correct on a complete, real-weight transformer block on real GPU
hardware, matching the reference output to roughly one part in a billion, and
running several times faster than the earlier all-encrypted design it replaced
while using half the encryption parameter size. It was kept as the permanent
foundation of the architecture; every later optimization below was built on top of
it.

A parameter-size reduction followed directly from the architectural change: because
each nonlinearity now resets the ciphertext to its lowest computational "level"
rather than consuming encryption budget for the rest of the block, the encryption
parameters could be shrunk substantially relative to the earlier, fully encrypted
design. A further, more deliberate search later re-derived the minimum parameter
setting that still passed correctness, finding a smaller value than the original
conservative choice. Both reductions passed correctness at the same very tight
precision as every other run and were kept as the operating parameters going
forward, though see the "abandoned" section below for why the associated speed
claim for the second reduction did not hold up.

Batching multiple client-boundary crossings into a single decrypt/compute/re-encrypt
call was tried and passed: several logically distinct nonlinear evaluations (for
example, many attention heads' worth of a nonlinear function, or several chunks of
an activation function) were packed into disjoint regions of one ciphertext so that
one physical round trip could carry many logical crossings at once. This reduced the
number of physical round trips in a complete block by roughly a factor of three,
with no change in accuracy and no change to the exact mathematical function being
computed at each boundary. It was kept as a standing technique used by every later
gate.

The single most consequential functional gap closed during this program was
generalizing the encrypted attention mechanism beyond a toy two-token special case
to an arbitrary number of tokens, using the real, numerically exact causal softmax
function evaluated at the client boundary rather than any closed-form shortcut. This
general mechanism was designed, proven first in an offline numerical check against
the real reference computation, and then verified correct on real GPU hardware at
successively larger token counts, each passing at the same tight precision band.
Along the way, two real, previously undiscovered defects were found and fixed as
part of keeping this optimization, not as separate initiatives: a fixed assumption
in the final output-repacking step that silently corrupted memory once the token
count exceeded a hard-coded limit (fixed generally, not patched for one token
count), and a fixed-capacity ceiling in how many attention scores could be packed
into one ciphertext simultaneously, which was resolved by spreading scores for
different attention heads across separate physical copies of the packing scheme,
raising the workable token count substantially. Reaching the full length actually
used by the downstream classification task required a further redesign: a two-phase
chunked version of the causal softmax that reduces a long attention row's scores in
fixed-size chunks before combining them at the client, removing the token-count
ceiling entirely rather than just raising it. This chunked design was proven
correct offline for row lengths far beyond what the previous scheme could even
represent, and then passed on real GPU hardware at the full token length used by the
downstream task. All of this — general causal attention, the output-repacking fix,
the spread-across-copies fix, and the final chunked-softmax design — was kept and
now constitutes the attention mechanism used by every subsequent, larger-scale gate.

A token-batching scheme that packs several tokens' worth of a matrix multiplication
into one ciphertext, so that one encrypted matrix-multiply call does the work of
several, was tried and passed. Measured on a matched pair of runs — one using the
packed scheme, one a same-source unpacked control computing the identical
mathematical result token by token — the packed version needed roughly one-eighth as
many encrypted matrix-multiply calls, and its measured wall-clock time (under
whatever ambient host load happened to be present in each of the two runs) was about
seven times shorter. Both passed correctness at the same tight precision band and
used substantially less peak memory on the process actually doing the work. This
batching width was chosen after a smaller batching width was evaluated by (see
below) a purely analytical comparison and rejected. The wider batching scheme was
kept as the standard configuration for all subsequent large-token-count work,
including the complete-block correctness gate at the downstream task's full token
length — which, combining every optimization above, passed with the same
high-precision match to the plaintext reference and was confirmed, via direct
process memory measurement, to fit comfortably within a single data-center GPU's
memory budget.

Reusing the one-time cryptographic setup cost (key generation and context loading)
across multiple encrypted evaluations, rather than repeating it for every single
run, was tried and passed, both within one running process and — via a
serialize-to-disk-and-reload mechanism — across two entirely separate operating
system processes. Both mechanisms produced results that passed correctness
identically to a freshly-set-up run, and both were kept as valid, safe options,
though (see below) neither delivered a large fraction of total run time, since the
one-time setup cost this optimization avoids is small relative to the encrypted
computation itself.

A CPU-side cache of the reusable pieces of a diagonal-batched matrix multiplication
— caching only ordinary host-memory numeric vectors, and deliberately never
caching or reusing any GPU-resident encrypted object — was tried after an earlier,
riskier version of the same idea failed (see below), and this safer version passed:
it produced the same correct output, hit its cache on the overwhelming majority of
lookups (confirming that the underlying computation was indeed being redundantly
repeated as suspected), and added no additional GPU memory cost relative to the
uncached version. It was kept and later became one of the two levers combined in
the most advanced experiment run to date (Section 2).

Composing more than one transformer block in sequence — decrypting the packed
output of one block, unpacking and re-encrypting it as the fresh input to the
next — was tried and passed for a two-block sequence, with direct process-memory
measurement confirming that the memory footprint used by the first block was not
carried forward or accumulated into the second. This was kept as the mechanism by
which multiple blocks will eventually be chained together, though it has so far
only been exercised at the smallest possible token count and for two of the model's
many blocks (see Section 3).

Splitting one block's encrypted work across two physical GPUs, each running as an
independent process against a shared, deserialized cryptographic context, was tried
for the least entangled part of a transformer block — the three initial linear
projections that do not depend on one another — and passed: both physical
processes, running concurrently on two separate graphics processors, produced
output matching the plaintext reference at the same tight precision band as every
other gate, with no cross-process leakage of state. In the one sample measured
where both processors happened to be genuinely free of other work, splitting this
piece of work across two processors did complete faster in wall-clock terms than
computing the same three projections serially on one processor, which is the first
directly measured (rather than only designed) concurrency benefit found in this
program. This mechanism was kept and is one of the two levers in the most advanced
combined experiment (Section 2).

### Optimizations that were tried and abandoned

An earlier, more aggressive version of the diagonal-batched-matrix-multiplication
cache attempted to reuse the same GPU-resident encrypted-plaintext object across
two separate multiplication calls (rather than only caching an ordinary host-memory
vector, as the safer version above does). This was implemented and then tried
repeatedly on real GPU hardware; every single attempt — six in total, on six
different physical graphics processors — crashed with an out-of-memory error deep
inside the encryption library's internal object-growth logic, always at the
identical code path. A same-day, byte-for-byte-identical control run of the
unmodified program, run twice, passed cleanly both times, ruling out simple
external contention as the explanation. The underlying mechanism was narrowed — it
is specifically the act of reusing one GPU-resident encrypted-plaintext object for
a second multiplication, not simply having many such objects in memory over time,
that triggers the crash — but the precise low-level reason inside the encryption
library's memory-growth code was not further diagnosed, since doing so would have
required reading deeper into that library's internal allocation logic than was
judged worthwhile given a safe alternative (the CPU-side-only cache above) already
existed. This approach is reported as an honest negative result and was abandoned,
not patched.

A library-native, pre-built batched primitive for exactly the diagonal-batched
matrix multiplication this program's own hand-written code performs was identified,
carefully scoped against this project's exact packing layout (three concrete
adaptations were required; it was not a drop-in replacement), implemented for one
representative call site, and measured correct on real GPU hardware. However, in
the same run, and therefore under identical ambient host conditions, the converted
call site measured measurably and consistently slower — by roughly 60 to 90 percent
— than its unconverted sibling calls computing an equivalent quantity by the
original hand-written method. Because nothing about the one converted call site was
structurally unusual relative to the others, converting the rest of the block's
calls to the same primitive was not attempted, since doing so was judged very
likely to reproduce the same regression without a clear hypothesis for why the
supposedly optimized library primitive underperforms the hand-written approach
here. This is reported as a valid, measured negative result: the library-native
primitive was tried and abandoned for this exact library build, not because it was
unsafe, but because it measured slower.

A hypothesis that some of the unexplained call-to-call timing variance seen in
early profiling was caused by one-time GPU allocator or kernel warm-up costs (the
first use of a given memory size or a given GPU kernel being slower than later,
repeated uses of the same kernel) was tested by inserting a deliberately untimed
warm-up pass before the real, timed computation. The correctness of the underlying
computation was reconfirmed unaffected. However, the timing comparison meant to
test the warm-up hypothesis itself was run under some of the heaviest sustained
host contention observed anywhere in this program, and the resulting per-stage
timing pattern was fully explained by that contention worsening over the course of
the run rather than by any warm-up effect. No clean, low-contention window was ever
available to repeat the comparison. This hypothesis was left genuinely
unresolved — neither confirmed nor refuted — rather than actively disproven, and
was not pursued further.

Splitting the second, more entangled stage of a transformer block's linear
work — the down-projection portion of the feed-forward layer, whose four chunks
must be summed together after independent computation — across two physical
graphics processors was designed and its underlying merge arithmetic was proven
exact in an offline numerical check. It could not be built, however: the pinned
version of the GPU-accelerated encryption library used throughout this project
provides no mechanism to transport an encrypted value (as opposed to
cryptographic keys or context, which it can transport) from one operating-system
process to another. Two source files implementing this split failed to compile at
exactly the point where an encrypted value would need to be written to, or read
from, disk for this transport. Everything else in both files compiled without
issue. This was therefore dropped as infeasible on the current library, not because
the idea was wrong — the merge mathematics themselves are exact and were
confirmed offline — but because the underlying library used in this project
currently has no way to move an encrypted intermediate value between two separate
processes. The equivalent first-stage split described above (kept, in the previous
subsection) was unaffected because it only ever needs to transport cryptographic
context and keys, never an encrypted intermediate value.

Finally, the re-derived, smaller cryptographic-depth parameter mentioned above
initially appeared, in a single measurement, to run noticeably faster than the
original, more conservative parameter choice. Two independent repeat measurements
of the smaller-depth configuration, run specifically to test whether that speed
advantage was real, both came back two to three times slower than the original
single sample and slower than the larger-depth baseline as well — the opposite
direction from the initial result. Careful comparison of per-process memory
telemetry across all repeats showed the smaller-depth configuration's own memory
footprint was identical in every repeat regardless of speed, while the measured
wall-clock time correlated instead with an independently observed measure of
whole-host processor contention that varied by roughly a factor of four across the
different attempts. The conclusion drawn is that this specific speed claim does not
hold up under repetition on this shared hardware, and it was retracted: the smaller
depth parameter itself remains in use going forward (it still passes correctness,
uses the same memory, and there is no evidence it is worse), but no speed advantage
from it is claimed.

### Optimizations rejected analytically, without a GPU run

A narrower token-batching width — packing half as many tokens into each ciphertext
as the batching width that was kept — was evaluated entirely through a local,
GPU-free count of every operation category the encrypted computation would need
under each width: the number of matrix-multiply calls, the number of
ciphertext-to-ciphertext and ciphertext-to-plaintext multiplications, the number of
rotations, the number of client round trips, and, most steeply, the number of
attention score decryptions. The narrower width came out worse on every single one
of these measures, using the identical underlying arithmetic machinery already
validated as accurate on real GPU hardware for the wider width. Because there was
no structural reason a live GPU measurement could reverse a result that held
uniformly across every operation category using already-validated cost
assumptions, no GPU time was spent confirming it, and the narrower width was
rejected without ever being run.

## 2. What has been combined

Several of the optimizations above were not developed as one-off, isolated
experiments but were deliberately layered onto one another over the course of this
program, and by the point of the single largest real-hardware gate run to date, a
substantial number of them were already combined in one experiment: the general,
chunked causal-attention mechanism, the client-boundary batching scheme, the
wider token-batching width, and the smaller re-derived cryptographic depth were
all present simultaneously in the complete-block correctness gate run at the
downstream task's full token length, and that combined configuration passed at the
same very tight precision as every individual piece measured in isolation, while
fitting within a single graphics processor's memory budget.

Two further, more targeted combinations were then attempted, each layering exactly
one additional lever onto that same already-combined baseline. First, the CPU-side
diagonal cache was added to the full combined single-process configuration above
and passed correctness on real GPU hardware, with the expected very high cache-hit
rate and unchanged memory use; a repeat of this exact combination the following day
also passed and happened to be the fastest sample recorded of any variant of this
configuration to date, though — because the two passing runs of this exact
combination used identical cache behavior and differed only in how busy the shared
host happened to be at the time — that speed difference cannot be cleanly
attributed to the cache rather than to the host simply being quieter that day, and
no clean, matched-conditions speedup claim is made for this pairing. Second, and
separately, the two-physical-GPU split of the first-stage linear projections was
added to that same baseline (without the CPU cache) and passed correctness with
both processors running concurrently, again reproducing the one directly measured
concurrency benefit described earlier, under the same single-sample caveat.

The most advanced combination attempted to date brings both of those levers
together at once: the CPU-side diagonal cache and the two-physical-GPU split of
the first-stage linear projections, run together in the same experiment for the
first time, on top of the already-combined chunked-attention, batched, and
reduced-depth baseline. This experiment has in fact already been run and has
already passed, as of the writing of this document. Both concurrent processes —
one computing two of the three initial projections with caching enabled, the other
computing the third with caching enabled — matched the plaintext reference at the
same very tight precision band as every other gate in this program, and the
internal bookkeeping that tracks how often the cache was actually reused matched an
independently derived arithmetic prediction exactly, with no leakage of cached
values between the two separate operating-system processes. A first attempt at
this exact combined experiment failed before producing any usable measurement, but
for a narrow bookkeeping reason unrelated to either lever's own correctness: an
internal consistency check that was newly written for this experiment miscounted
how many cache lookups to expect, and threw an error after both processes had
already finished their real encrypted computation correctly. That check was
corrected against an independent, from-scratch numerical recomputation of the
expected count, and the experiment was re-run successfully. As with the individual
pairings above, this combined result is explicitly a correctness proof rather than
a speed measurement: the host was, at the time of this run, concurrently busy with
unrelated work from another experiment running on the same shared machine, so no
speed claim is drawn from it.

Some individually-proven optimizations have never been tested together and this is
worth stating plainly. Most notably, the two-physical-GPU split has, in every form
tested so far, only ever covered the three initial linear projections of one
transformer block — it has never been combined with the multi-block composition
mechanism (which itself has only ever been exercised for two blocks, at the
shortest possible token length, well short of the downstream task's actual token
length). Likewise, the general causal-attention mechanism at the task's full token
length has never itself been split across two physical processors — only the
simpler initial linear projections have been sharded this way to date. The
second-stage feed-forward split was never combined with anything, because, as
described above, it could not be built at all on the encryption library version
used throughout this project. And the one-time cryptographic-setup-reuse
mechanisms, while proven independently both within one process and across two
separate processes, have not been folded into the same single experiment as the
CPU-side cache and the two-GPU split described just above, although the two-GPU
split's own underlying mechanism does itself rely on the cross-process
setup-reuse technique to share one cryptographic context between its two
concurrent processes.

## 3. The best combined result so far

The most-optimized configuration that has actually passed correctness in a single
run to date is the complete, single transformer block, evaluated at the downstream
task's full real token length, on one physical graphics processor within one
operating-system process, using every kept optimization from Section 1 layered
together: the general chunked causal-attention mechanism, client-boundary batching
of multiple nonlinearity crossings into single round trips, the wider
token-batching scheme, the smaller re-derived cryptographic depth, and the
CPU-side diagonal-multiplication cache. This configuration matched the plaintext
reference to within roughly four parts in a billion, well inside the accepted
tolerance, used no additional graphics-processor memory relative to the same
configuration without the cache, and was directly confirmed, through per-process
memory telemetry, to fit within a single data-center graphics processor's memory
budget. To date, no clean, uncontaminated timing measurement of this exact
combined configuration exists — every sample measured so far shared the machine
with unpredictable and sometimes very heavy activity from other tenants, so while
the correctness of this configuration is solid, no confirmed speed figure can yet
be attached to it.

The 12-block task-length driver and final task head still need to close for arithmetic correctness.
That run alone would not make the implementation optimization-complete. The audit below found exact-
model and protocol-layout opportunities that were not represented in the earlier parameter and
micro-optimization plan. A genuinely clean timing measurement is also required before any wall-clock
claim about the task-complete system can be made.

## 4. Remaining opportunities and execution decision

The current roadmap covers depth reduction, eight-token packing, a CPU diagonal-vector cache, one
native linear-transform primitive, and a process-level Q/K/V split. It does not close the broader
packing and client-boundary design space.

### Highest-priority exact-model gates

1. **Profile the current 103-token block.** The detailed profile in the task history belongs to the
   older two-token graph. Current server timing combines CPU packing/encoding, transfer, kernels, and
   synchronization, so it cannot yet select the next systems optimization.
2. **Use the four ciphertext copies for different dense projections.** Q/K/V and the four MLP chunks
   can potentially be evaluated in parallel copies rather than repeating the same matrix in every
   copy. Before masks and repair operations, the derived schedule reduces 156 dense products to about
   52 and the complete ciphertext–plaintext multiplication count by roughly 60 percent. This is an
   unmeasured engineering estimate pending a complete packing contract.
3. **Compute complete LayerNorm at its existing client boundary.** The client already decrypts data
   for the exact nonlinear statistic. Returning a fully normalized fresh ciphertext could remove
   encrypted reductions and affine work, lower required depth, and fuse inter-block refresh with the
   next block's first normalization. The model and server privacy boundary remain unchanged, but the
   client/server work allocation changes and must be reported.
4. **Replace per-head attention reductions and test specialized matrix layouts.** The retained
   schedule performs separate reductions for every head/alignment. Segmented head reductions and
   compact plaintext–ciphertext/ciphertext–ciphertext matrix layouts need exact local contracts and a
   representative GPU gate. Testing one FIDESlib `LinearTransform` call did not evaluate this broader
   algorithm family.
5. **Finish encoded-weight reuse.** The retained cache stores host vectors, not encoded plaintexts.
   The failed GPU-object cache identifies a backend bug; it does not prove safe per-level encoding,
   batched upload, or a corrected backend path cannot help.
6. **Evaluate wider and stage-specific layouts.** B=8 beat B=4 only under the fixed four-copy layout.
   B=16/two-copy, B=32/one-copy, real/imag packing, boundary-specific layouts, and a smaller final-head
   context remain untested.

### Separate protocol and model branches

The client could compute the complete attention context at the already-declared softmax boundary,
removing most encrypted weight-tile work. That preserves privacy from the server but moves substantial
linear work to the client and must be presented as a protocol variant. Low-rank or structured weights,
pruning, quantization, distillation, and token dropping change the released model and require separate
task-accuracy claims.

### Execution decision

Use a dedicated host first for a paired one-block baseline and current CPU/CUDA profile. Prove the
packing and boundary changes locally, microbenchmark them one at a time, combine only retained changes,
and then rebuild the 12-block driver. Run the existing driver immediately only to close full-model
arithmetic correctness; do not present that baseline as the optimized latency result. After arithmetic
closure, a real client/server transport experiment is still required because the current in-process
boundary count is not a network round-trip measurement.

The gate order and acceptance conditions are canonical in [roadmap.md](roadmap.md).
