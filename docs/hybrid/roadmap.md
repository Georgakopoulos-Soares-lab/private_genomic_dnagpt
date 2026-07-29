# Roadmap — Scheme B (hybrid client-assisted CKKS), active

The active FHE architecture. See [../roadmap.md](../roadmap.md) for the shared
acceptance contract and rules, and [../pure/roadmap.md](../pure/roadmap.md) for the
frozen Scheme A history that motivated this pivot.

## Scheme B: hybrid client-assisted CKKS

Adopted 2026-07-25 after three independent chained-composition failures
(`multiplicative_depth` in `{50, 58, 64}`) all traced to GPU memory exhaustion during
rotation-key/bootstrap-plaintext loading, not accuracy — i.e. the "measured GPU
correctness failure" condition below is met. Full comparison and rationale in
[../shared/architecture_options.md](../shared/architecture_options.md). Plan:

1. Reuse the existing real-weight block-0 CKKS/FIDESlib graph unchanged for every linear
   op (Q/K/V projection, attention matmul, output projection, FFN, residual adds).
2. Remove `EvalChebyshevFunction` calls for LayerNorm invsqrt, the T=2 sigmoid attention
   identity, and GELU. Replace each with an explicit decrypt (client, secret-key holder,
   own data only) -> exact plaintext function -> re-encrypt boundary.
3. Re-measure against the unchanged Phase-A oracle and `4e-2` gate; record round-trip
   count and the GPU-encrypted vs. client-plaintext wall-time split. `[done, 2026-07-26]`
   — complete real-weight D=768/T=2 block-0 gate passes (`rel_inf=3.32e-10`, 24 round
   trips, `372.27s` = `365.78s` server + `6.49s` client boundary); see
   [tasks.md](tasks.md) and `fhe_fides_real_d768_t2_block0_scheme_b_a100_20260725.json`.
4. Because Scheme B removes the bootstrap-driven depth requirement, re-derive the
   minimum viable `multiplicative_depth`/`batch_slots` before assuming depth 43 is still
   needed; a smaller context may clear the exact memory wall that blocked Scheme A.
   `[done, 2026-07-26]` — the passing block-0 gate above already lands at
   `ring_dim=65536`/depth 16, half of Scheme A's `ring_dim=131072`/depth 43.
5. Scale sequence length and block count only after one Scheme B block gate passes,
   mirroring the "Scale and optimize" and "Compose only after one block scales" order
   in [../roadmap.md](../roadmap.md). The naive ×12-block T=2 linear extrapolation
   ([tasks.md](tasks.md), `~74.6 min` total) is `[A]`-only.
   `[done, 2026-07-28]` — a general causal-attention Scheme B circuit (real per-row
   exact softmax at the client boundary, generalizing the T=2 closed-form identity
   rather than replacing it) is designed, numpy-proven against a real T=3 oracle, and
   passes both the `attention` and `full` real-GPU gates at T=3
   (`global_rel_inf` `3.19e-10`/`3.65e-10` against the unchanged `4e-2` tolerance; see
   [tasks.md](tasks.md)'s 2026-07-28 "General causal-attention Scheme B circuit"
   section and `fhe_fides_real_d768_t3_{attention,full}_general_attention_scheme_b_a100_20260728.json`).
   `[U]` Only T=3 is measured; scaling to task-representative lengths (`T=32/64/103`)
   and multi-block composition remain open, and no speed claim is made from this
   result (T=3 is not compared against the T=2 anchor; round-trip/score-count growth
   is `O(T)`/`O(T^2)` by construction, not yet empirically confirmed beyond T=3).
   `[done, 2026-07-29]` — growth-trend confirmed at a second point, T=8
   (`attention` gate, `global_rel_inf=3.69e-10`, `round_trips=36=28+8`, matching the
   `T*(T-1)/2` prediction exactly), after finding and fixing a real, previously
   undiscovered bug along the way: the frozen output-packing step
   (`finish()`/`output_block_plain()`) silently assumed `T<=COPIES=4`, causing a heap
   buffer overflow at T=8 (two crashes, root-caused via added flushed per-row
   logging that also exonerated the attention circuit itself — all 28 round trips
   completed before the crash both times). Fixed generally (groups tokens into
   `ceil(T/COPIES)` output ciphertexts), not patched for one T. See
   [tasks.md](tasks.md)'s 2026-07-29 "T=8 growth-trend check and output-packing
   ceiling" section. `[U]` A second, still-open ceiling remains in the
   causal-score packing (`HEADS*SCORE_SLOT_STRIDE<=PACK_WIDTH-D`, currently
   `T<=21`, `T<=85` with an already-scoped but unimplemented 4-copy fix) — this
   would need addressing before reaching task-representative lengths
   (`T=32/64/103`). `full` gate (LN2/GELU/MLP) not re-verified at T=8, only
   `attention`. No speed claim: host was fully occupied by another tenant for
   hours before this run's GPU freed up.
6. `[done, 2026-07-27]` GPU-side profiling of `server_linear_algebra_seconds` found
   the 6 matmul stages are 99.5% of server time, with ciphertext-plaintext
   multiply-and-accumulate ~100% of one matmul call's internal cost (rotation/
   keyswitch <0.1%, not the bottleneck at any BSGS split). See
   [tasks.md](tasks.md)'s 2026-07-27 profiling subsection.
7. A diagonal-plaintext-cache fix motivated by that profiling
   (`real_dnagpt_fides_scheme_b_diagcache.cpp`) was implemented and contract-tested
   but abandoned after 6 consecutive real-GPU crashes, root-caused via `addr2line`
   to `Ciphertext::multPt`'s plaintext-level-adjustment path rejecting a reused
   GPU-resident `Plaintext` object. A follow-up warm-up-prelude hypothesis test
   passed correctness but returned an inconclusive timing result (host contention
   swamped the signal). See [tasks.md](tasks.md)'s 2026-07-27/28 subsections for
   the full evidence trail.
8. **Closed, measured-negative.** FIDESlib's own `LinearTransform`/
   `LTdotProductPtBatch` native batched BSGS primitive (verified present in the
   pinned commit, used by FIDESlib's own `examples/bert-tiny/src/MatMul.cu`)
   was scoped against this repo's exact packing layout (compatible, three
   concrete adaptations required -- not a drop-in), implemented for one call
   site (QKV query projection) in `real_dnagpt_fides_scheme_b_lintransform.cpp`,
   validated locally (106 tests, including a numpy reconstruction against the
   real oracle fixture), and run on a real A100. `[done, 2026-07-28]`
   Correctness passes (`global_rel_inf=4.59e-10`). `[V]` Timing regresses: the
   converted call measures `1.6-1.9x` *slower* than its unconverted sibling
   calls in the same run (the only comparison not confounded by host load).
   Converting the remaining 23 call sites was not attempted given this
   negative signal. See [tasks.md](tasks.md)'s 2026-07-28 "`LinearTransform`
   scoping", "Step 2: single-call-site implementation", and "isolated
   single-call-site gate" subsections for the full evidence trail.
