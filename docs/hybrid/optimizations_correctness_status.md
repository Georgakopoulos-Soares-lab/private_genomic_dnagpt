# Scheme B — every optimization tried, correctness status (2026-08-01)

Gate for all entries: output must match the plaintext oracle within
global_rel_inf ≤ 4e-2 on a real A100. "PASS" below means that bar was met
on real GPU hardware, not just in local/CPU simulation, unless stated.

## Passed on real GPU

- **Core hybrid design** (client decrypts only at LayerNorm/GELU/attention
  boundaries, server stays encrypted for all linear algebra) — PASS,
  T=2 block-0, rel_inf 3.32e-10.
- **Reduced encryption depth** (16, then 13, instead of Scheme A's 43) —
  PASS at both depths, same rel_inf band (~1e-9–1e-10).
- **Client-boundary round-trip batching** — PASS, no accuracy change.
- **Context/key caching**, in-process and cross-process (serialize once,
  reload in another process) — PASS both ways, proves state can be reused
  without regenerating keys.
- **General causal-attention circuit** (works for any token count, not
  just the original 2-token special case) — PASS at T=3, T=8, T=32, and
  T=103 (the real task length). T=32 and T=103 also passed as a complete
  block (attention + LayerNorm + GELU + MLP together), not just attention
  alone.
- **Multi-block chaining** (output of block 0 decrypted/re-encrypted and
  fed into block 1) — PASS, but only tested at the tiny T=2 size, not yet
  at T=103.
- **Token-batching / Token-SIMD** (processing 8 DNA positions per
  ciphertext instead of 1) — PASS at T=103, both as an isolated linear-op
  test and as a complete full block. This is the change that also cut
  compute ~8x.
- **FIDESlib's built-in batched matrix-multiply primitive**
  (`LinearTransform`) — PASS on correctness (rel_inf 4.59e-10), but it
  measured 1.6–1.9x *slower* than our existing method, so it was not
  adopted beyond the one test call.
- **CPU-side diagonal cache** (cache repeated intermediate values on the
  CPU only, never on the GPU) — PASS, rel_inf 4.50e-9, 99.99% cache-hit
  rate confirming the redundant work it targets was real. Speed benefit
  looks directionally positive but not yet cleanly proven (see below).
- **2-GPU sharding, first slice** (splitting the Q/K/V computation across
  two physical GPUs running at the same time) — PASS on both GPUs
  independently, against the real oracle, with no shared secret-key
  material. First actual proof (not just a plan) that two GPUs can share
  one encrypted computation correctly.
- **CPU-side diagonal cache + 2-GPU sharding, combined** (2026-08-01) —
  PASS on both GPUs, same ~1e-9 band, each worker's cache hit/miss count
  exactly matches the expected formula. The one remaining not-yet-combined
  pair from "Has everything been combined?" below is now combined too.

## Tried and abandoned (failed correctness or unsafe)

- **GPU-resident diagonal cache** (an earlier version of the caching idea
  that reused an object already loaded on the GPU) — FAILED. Crashed 6/6
  times on real GPU runs; root-caused to a FIDESlib internal issue with
  reusing that specific GPU object. Abandoned, replaced by the CPU-side
  version above, which avoids the exact mechanism that crashed.
- **Native multi-GPU mode inside FIDESlib itself** (letting the library
  spread one computation across GPUs on its own) — FAILED, crashes on
  setup. This is why the project instead builds sharding by hand across
  separate processes (see "2-GPU sharding" above).

## Decided against without needing a GPU run

- **4-token batching instead of 8** — not tested on GPU; ruled out by
  plain arithmetic (it's worse on every measured operation count than
  8-token batching), so no GPU time was spent confirming it.

## Built but not yet run for correctness at full scale

- **Full 12-block, end-to-end run at T=103** (the real task size, start
  to finish) — the code is written, compiles, and passes all local
  (non-GPU) checks, but the actual real-GPU run has deliberately not been
  launched yet. It would take roughly 16–45 hours on one GPU, so it is
  waiting for an explicit go-ahead rather than being run automatically.
  This is the single biggest remaining gap: correctness has never been
  confirmed for the complete model, only for one block at a time.

## Has everything been combined into one run?

Getting closer. Three runs now stack optimizations together:

- Depth reduction + token-batching + CPU-side cache, all in one binary —
  PASS (rel_inf 4.50e-9).
- Depth reduction + token-batching + 2-GPU sharding — PASS on both GPUs.
- Depth reduction + token-batching + CPU-side cache + 2-GPU sharding, all
  four together (2026-08-01) — PASS on both GPUs, ~1e-9 band on each
  requested part, cache hit/miss invariant exact on each worker. This was
  a correctness-only check (not a speed run), confirming the recipe
  "each of the two GPU workers also independently carries its own cache"
  works exactly as the design predicted, with no cross-process cache
  interference and no accuracy change.

This still hasn't gone past one block (still the block-0 gate, not the
12-block end-to-end run), and it doesn't include the Stage-2 MLP-chunk
sharding lever, which turned out to be unbuildable on the pinned FIDESlib
for an unrelated reason (no ciphertext serialization in the library —
see tasks.md's 2026-08-01 Stage-2 entry) rather than a correctness
failure.

## Bottom line

Every optimization that reached a real GPU test either passed correctness
cleanly or was caught and dropped before being relied on (the two
"abandoned" items above). Nothing incorrect has been left in place. The
open item is scale, not correctness: the full 12-block run at real input
size has not been executed yet, so we don't have end-to-end proof — only
proof that each individual piece, and now the combined cache+sharding
recipe together, works correctly at the single-block scale.
