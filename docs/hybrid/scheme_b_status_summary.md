# DNAGPT under encrypted computation (Scheme B)

**Date:** 2026-07-30 (updated with the T=103 and token-batching results)
**Audience:** anyone on the team who wants the current state without the engineering log.

## The idea in one paragraph

We want to run DNAGPT (a genomic language model) on a compute provider we don't fully trust, without that provider ever seeing the raw DNA sequence. We use homomorphic encryption (CKKS) so the provider computes on encrypted data the whole one relaxation we allow ("Scheme B" / hybrid): a handful of math operations inside the model (things like normalization and attention weighting) aren't efficient to do while encrypted, so at those specific, pre-agreed points the *data owner's own client* — the only party who ever holds the decryption key — briefly decrypts just that intermediate value, does the small calculation in the clear, and re-encrypts it before sending it back. The untrusted server never sees plaintext DNA, never sees the key, and never gets to choose what gets decrypted.

## Is it accurate? Yes — and now proven at the real target size too

Every test I've run keeps coming back correct to about 9-10 decimal digits of precision, roughly 100-150 million times inside our accuracy bar. That was already true at small test sizes (2-32 tokens). The update: we've now tested the attention step at **103 tokens — the actual real-world input length this genomic task needs**, not just a toy example — and it still passes at that same tiny error margin (about 109 million times inside tolerance). The new token-batching approach described below is very slightly less precise (still roughly 6 million times inside tolerance, comfortably fine) because it reuses one shared calculation across several tokens at once. Bottom line: accuracy has never been, and still isn't, a concern at any scale we've tested, including the real one.

## Is it fast enough yet? Not yet overall — but I just got the first real

## evidence that it can be

Running just the attention step (one piece of one layer) at the real 103-token length took about **3.9 hours** on one GPU. The model has 12 of these layers, and attention is only part of one layer — so left unchanged, this approach is far too slow for practical use.

The good news: I had a theory that processing several tokens together in one encrypted calculation, instead of one token at a time, should make encrypted math much more efficient — encryption schemes like this one are built to batch work that way. I tested that theory for real for the first time, with a clean, apples-to-apples comparison (identical program, identical GPU, identical input — the only thing that changed was "one token per calculation" vs. "eight tokens per calculation"):

- Bundling 8 tokens together cut the number of encrypted calculations needed for that piece of the model from 103 down to 13 — almost exactly the 8x reduction we expected.
- The measured wall-clock time dropped by about **7.4x** — very close to the ideal 8x, meaning the batching trick has very little overhead in practice.
- It also used noticeably less GPU memory doing it this way (about 6 GB less peak usage) — a welcome side benefit, not the main goal.
- This was a real, measured GPU result, not a paper estimate.

The catch: this speed test only covered one small building block (one normalization step plus one matrix multiply), not the full attention step or a full model layer. It's a strong, real signal — not yet a full-model proof.

## What this means for the full picture

Scenario > Estimated time for one full run (12 layers, 103 tokens, 1 GPU)

Today's unchanged, one-token-at-a-time approach > ~4 days

If the newly measured ~7x batching speedup carries through the rest of the model (not yet proven, but the natural next test) > plausibly well under a day — this is a projection, not a measurement, until I actually run a full batched layer

Spreading either scenario across multiple GPUs in parallel would shrink it further, but I haven't measured that yet either.

## What's still missing before I can call this solved

- I have only run the *attention* step at the real 103-token length — not a complete model layer (attention is followed by more steps I haven't yet measured at that length).
- I have chained multiple layers together and confirmed it works correctly, but only at a tiny test size (2 tokens), not at the real 103-token length.
- The token-batching speedup has only been proven for one small piece, not applied to the rest of the model yet.
- I haven't yet confirmed that a full-size encrypted run fits comfortably in one GPU's memory.

## Bottom line

- **Accuracy: solved, and now confirmed at real scale.** The encrypted version matches the unencrypted version to about 9-10 decimal digits, at every size we've tested including the actual 103-token real-world input length.
- **Speed: still the open problem, but I just got real traction on it.** Today's design is still roughly a 4-day job for one full run. But I just measured, for real on a GPU (not just estimated), that processing tokens in batches of 8 delivers close to the full theoretical speedup with little overhead and even saves memory.
- **Next milestone:** prove the same batching trick works across a whole model layer (not just one small piece), then chain layers together at the real 103-token length to get an honest, measured end-to-end estimate instead of a projected one.
