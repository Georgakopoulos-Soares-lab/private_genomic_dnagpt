# DNAGPT under encrypted computation (Scheme B)

**Date:** 2026-07-31 (updated with the minimum-passing-depth result)
**Audience:** anyone on the team who wants the current state without the engineering log.

## The idea in one paragraph

We want to run DNAGPT (a genomic language model) on a compute provider we don't fully trust, without that provider ever seeing the raw DNA sequence. We use homomorphic encryption (CKKS) so the provider computes on encrypted data the whole one relaxation we allow ("Scheme B" / hybrid): a handful of math operations inside the model (things like normalization and attention weighting) aren't efficient to do while encrypted, so at those specific, pre-agreed points the *data owner's own client* — the only party who ever holds the decryption key — briefly decrypts just that intermediate value, does the small calculation in the clear, and re-encrypts it before sending it back. The untrusted server never sees plaintext DNA, never sees the key, and never gets to choose what gets decrypted.

## Is it accurate? Yes — and now proven at the real target size too

Every test has remained far inside the unchanged `4e-2` arithmetic-accuracy
gate. We have now run a **complete released-weight block at 103 tokens**, the
task-representative length, with the B=8 token-batching layout. It passes at
`global_rel_inf=4.78e-9`: about 8.37 million times inside the gate. This covers
normalization, Q/K/V, causal attention and softmax, attention projection,
GELU, the complete MLP, and both residual paths. Accuracy is not the current
single-block constraint. The still-unmeasured question is how error propagates
through all 12 blocks and the task head.

## Is it fast enough yet? No — but Token-SIMD is real progress

The complete packed T=103 block took **6,281.75 seconds (104.7 minutes)** of
encrypted evaluation: 6,140.14 seconds of server linear algebra and 141.61
seconds at client boundaries. This is a real correctness run, but not a clean
speed benchmark. It launched after a clean capacity check and later acquired
multiple co-tenants; one cotenant alone reached 65.9 GB on the same GPU.

The good news is that processing several tokens together in one encrypted
calculation works:

- Bundling 8 tokens together cut the number of encrypted calculations needed for that piece of the model from 103 down to 13 — almost exactly the 8x reduction we expected.
- In the earlier matched-source linear micro-gate, observed wall-clock time
  dropped by about **7.4x**. Both sides were co-tenant-contaminated and ran in
  different windows, so this is an observed event ratio, not a dedicated-GPU
  guarantee.
- It also used noticeably less GPU memory doing it this way (about 6 GB less peak usage) — a welcome side benefit, not the main goal.
- The complete block now proves that the packing composes across every block
  stage. Its exact dense-product count is 156 instead of 1,236
  serial-equivalent products, a structural 7.92x reduction.
- The complete block's own process peaked at only 12.92 GB, so one T=103 block
  comfortably fits an 80GB A100.

## How much of the unused depth budget can we actually reclaim? Answer: down to 13, not lower

The complete block above ran with a deliberately conservative 16-level CKKS
budget, and it only actually used 6 of those levels by the time the answer
comes out — a lot of apparent slack. Reducing that budget is exactly the kind
of change that can make every encrypted operation cheaper, so we spent this
round finding the real floor instead of guessing.

It took four attempts to get there, and each failure taught us something
concrete rather than being a dead end:

- **Depth 8, 4 key-switching digits:** rejected immediately by the crypto
  library itself — the math doesn't distribute cleanly, before any encrypted
  computation even starts.
- **Depth 8, 3 digits:** passed that check, but the library picked too small
  a ring size for our packing; a one-line fix pins the ring explicitly.
- **Depth 8, ring fixed:** ran for real this time, computed the entire
  causal-attention step correctly, and then crashed on the very last
  (deepest) step with a low-level bug in the crypto library itself — reading
  the library's own source code pinned the exact cause (an internal
  bookkeeping counter isn't guarded against going negative when the budget
  runs out one operation too early).
- **Depth 9:** fixed that crash (one more level of headroom), got all the
  way through attention, and then hit a *different*, later problem: the
  decryption at that same deepest step came back numerically untrustworthy —
  one level short of enough precision, not a crash this time.
- **Depth 10:** fixed that too, and got all the way to the start of the
  final feed-forward layer before the code's own built-in safety check
  said, in effect, "not enough budget left to do this safely" — a clean,
  intentional stop, not a bug.

Rather than keep guessing "+1" each time, we added one line to print the
exact number the safety check was working with, ran it once, and read off
the answer directly: **13 levels is the true minimum.**

Depth 13 then ran the complete block correctly: accuracy was
`4.35e-9` — slightly *better* than the original 16-level run, still about
9.2 million times inside the accuracy gate. Encrypted evaluation took 4,662
seconds versus 6,282 seconds for the 16-level run, roughly **26% faster**,
and peak GPU memory dropped from 12.92 GB to 9.83 GB. Both numbers are
directional only — the machine was shared with other jobs both times, and
we'd want two or three clean-GPU repeats before calling either number
"the" speed or memory figure.

## What this means for the full picture

`[A]` Naively multiplying this contaminated block observation by 12 gives
about **20.9 hours** of encrypted evaluation on one GPU. That is not a measured
end-to-end runtime: the 12 blocks and task head have not been run at T=103, and
the single-block timing is contaminated. “12 hours as-is” is therefore an
optimistic assumption, not a verified result.

The arithmetic-error budget is enormous, so a deliberate depth/precision sweep
is justified. However, accepting more error only helps if it lets us remove
RNS levels or change the packing. The current B=8 layout already uses all
32,768 complex slots and therefore requires `ring_dim>=65536`; lower precision
alone cannot move this exact layout to a 32,768 ring. Re-deriving the minimum
depth may still reduce per-operation work. Reaching a few-hour POC will likely
require that reduction **and** real multi-GPU sharding, not accuracy sacrifice
alone.

## What's still missing before I can call this solved

- Run a clean dedicated-A100 complete-block benchmark at the new 13-level
  depth; both the 16-level and 13-level timings so far are contaminated by
  other jobs sharing the machine.
- Sweep scale bits and/or whether fewer key-switching digits still pass at
  depth 13, then check the actual downstream task metric, not just block
  error.
- Check whether the B=4 token-batching alternative (untested) beats B=8 once
  paired with the smaller depth.
- Extend the already-passing T=2 two-block refresh mechanism to T=103, then to
  all 12 blocks and the task-output head.
- Measure multi-GPU sharding of algebraically independent Q/K/V and MLP chunks.

## Bottom line

- **Accuracy:** a complete T=103 block passes with a 9.2-million-fold margin
  at the new, smaller depth.
- **Memory:** one complete T=103 block peaks at 9.83 GB at depth 13 (down
  from 12.92 GB at depth 16) and fits an 80GB A100 comfortably either way.
- **Speed:** the minimum CKKS depth is now a measured fact (13, not a guess),
  and it directionally shows ~26% faster encrypted evaluation — but that
  number, like the 16-level one before it, is still a contaminated
  single-machine sample, not a clean benchmark or an end-to-end 12-block
  measurement. The only defensible current 12-block number remains a
  contaminated one-block ×12 projection (about 20.9 hours at depth 16;
  proportionally less at depth 13, but not separately measured).
- **Next milestone:** a clean-GPU repeat of the depth-13 timing, then combine
  the smaller depth with multi-GPU execution and T=103 multi-block
  composition.
