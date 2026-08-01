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
   `[U]` Depth 16 is still explicitly labeled a conservative placeholder in the
   source, not a proven minimum. The T=2 two-block gate below measures `LN2 max=8`
   and packed output level 6 in both blocks, leaving material unexamined headroom.
   Re-deriving the smallest passing depth remains open. The passing B=8 Token-SIMD
   block below consumes all 32768 configured complex slots, which itself requires
   `ring_dim>=65536`; lowering depth cannot move this exact packing to
   `ring_dim=32768`. Depth reduction may still reduce the number of live RNS limbs
   and per-operation cost. A smaller ring would require a separate B<=4 packing and
   would double the dense-group count, so the two effects must be measured jointly.
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
   `[done, 2026-07-29]` — the first Scheme B multi-block gate now passes for released
   blocks 0 -> 1 at T=2. The client decrypts/unpacks block 0's packed hidden state and
   re-encrypts both token states at level 0 under the same context/key lineage before
   block 1; block 0/1 pass at `global_rel_inf=9.24e-11`/`8.48e-11`. PID-specific
   telemetry peaks at `10872 MiB` during block 0 and does not rise in block 1, so the
   Scheme A per-block memory accumulation failure does not transfer to this T=2
   sequential-refresh composition mechanism. See [tasks.md](tasks.md)'s 2026-07-29
   "First Scheme B multi-block composition gate" entry. `[U]` All 12 blocks and the
   task-output head remain untested; this two-block result does not yet constitute the
   project's embedded-vector-to-task-output graph. `[U]` Memory scaling with T also
   remains unmeasured, so this does not establish that one 80GB A100 can run T~100.
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
8. **T=103 attention packing and complete-block gates closed.** `[done,
   2026-07-30]` The two-phase chunked causal softmax is implemented with a fixed
   `CHUNK_WIDTH=85`, passes all 27 local static/numpy contracts, builds under the
   pinned FIDESlib commit, and passes the real A100 `attention` gate at the full
   task-representative `T=103`: `global_rel_inf=3.66e-10`, `round_trips=5476`,
   and `final_decrypt_calls=26=ceil(103/4)`. This is the first real-GPU validation
   beyond the earlier `T<=85` score-packing ceiling. Immutable evidence:
   `results/runs/fhe_fides_real_d768_t103_attention_general_attention_t103_scheme_b_a100_20260729.json`
   (`sha256=8f22ccf7c34125710197b84ef0b571f2095f8fbb800f7eb42afbfdbf38b8daee`).
   The follow-on complete B=8 Token-SIMD block passes the full
   LN1/QKV/causal-softmax/attention-projection/LN2/GELU/MLP/residual graph:
   `global_rel_inf=4.78e-9`, `156` dense products versus `1236`
   serial-equivalent, `857` client crossings, and `packed_output_level=6`.
   PID-specific telemetry peaks at `12920 MiB`, proving this complete T=103 block
   fits one 80GB A100. Immutable correctness/VRAM evidence:
   `results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_scheme_b_{a100,vram_a100}_20260730.json`.
   `[U]` Its `6281.75s` encrypted-evaluation timing is heavily contaminated: a
   later cotenant reached `65868 MiB` and whole-GPU allocation reached
   `76757 MiB`. `[U]` All 12 blocks and the task-output head remain untested at
   T=103; one block fitting does not prove that composition preserves the same
   bounded live set.
9. **Closed, measured-negative.** FIDESlib's own `LinearTransform`/
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
10. **Token-SIMD leverage and complete packed block proven.** `[done,
    2026-07-30]` An additive T=103 micro-gate packs B=8 logical token lanes into
    each ciphertext and compares exact client LN1 plus the real encrypted query
    projection against a same-source serial control. Both real-A100 modes pass
    the unchanged accuracy/privacy gates. Matrix products fall from 103 to 13
    (`7.923x`), while observed encrypted-evaluation time falls from `2260.08s`
    to `306.54s` (`7.3728x`). PID-specific peaks are `15992 MiB` serial and
    `9848 MiB` packed. Immutable correctness and telemetry evidence is in the
    four `results/runs/*simd_linear_t103*20260730_v2.json` files; exact hashes
    and reproduction commands are in [tasks.md](tasks.md).
    `[U]` The timing pair is shared-host/co-tenant-contaminated and ran on
    different physical A100s in different windows, so it is an observed event
    ratio, not a clean dedicated-A100 claim. The additive complete packed block
    subsequently passes as recorded in item 8. `[U]` The remaining gates are a
    clean dedicated-A100 speed measurement, minimum-depth/precision sweep,
    T=103 multi-block refresh composition, all 12 blocks, and the task head.
11. **Minimum `MULT_DEPTH` re-derived and B=4 packing width closed.** `[done,
    2026-07-31]` Minimum passing `MULT_DEPTH=13` (3 HYBRID digits, ring
    `65536`) found via the diagnostic-print method and confirmed passing,
    `~25.8%` directionally faster than depth 16. `[done, 2026-07-31]` B=4 vs
    B=8 Token-SIMD packing width closed by local operation-count evidence
    (no GPU time spent): B=4 is worse on every measured operation category
    (dense products, ct-ct/ct-pt multiplications, rotations, round trips),
    not only the previously-known dense-product ratio -- B=8 remains the
    packing width to carry forward. See [tasks.md](tasks.md)'s 2026-07-31
    entries for both. `[V]` Two clean-timing repeats for depth 13 landed via
    one-shot host cron (the reliable detached-launch mechanism on this host,
    since plain `nohup`/`disown` did not reliably detach) -- **both PASS
    correctness but are `2.1`-`2.9x` SLOWER than the original single sample
    and the depth-16 baseline alike** (`13658.27s`/`13202.35s` vs
    `4662.22s`), despite *lighter* GPU-memory contamination than the
    original run. This converts the earlier directional "25.8% faster"
    figure into an explicit non-claim: no depth-13-vs-depth-16 speed
    advantage is demonstrated by the evidence in hand, and host-wide CPU
    contention (load average `500`-`1042`, this project's worst observed)
    rather than GPU-memory pressure is the better-correlated explanation.
    See [tasks.md](tasks.md)'s 2026-07-31 "both landed" entry for the full
    3-sample table and root-cause analysis. A 2-GPU process-per-GPU
    sharding design (Q/K/V split first as a zero-merge infra proof, then
    MLP-chunk split as an exact-`EvalAdd`-merge proof, then independent
    attention token/query groups) is sketched, and a local math contract
    for the Q/K/V split passes (`shard_layout.py`/`test_shard_layout.py`,
    7/7); the actual C++ (a new Token-SIMD-parameter-matched writer fork is
    a discovered prerequisite -- the existing writer/reader pair is built
    for the incompatible T=2/4096-slot layout) is scoped but not yet
    written, on hold pending user direction.
12. **Three parallel tracks: CPU-diagonal-cache retry PASSES, 2-GPU Q/K/V
    sharding proves real concurrency, 12-block+head T=103 driver built and
    ready.** `[done, 2026-08-01]` **CPU-side diagonal-plaintext cache**
    (caches only the host `packed_values` vector, never a GPU-resident
    `Plaintext` -- structurally avoids the 2026-07-27 crash mechanism):
    PASSES (`global_rel_inf=4.50e-9`), `99.99%` cache hit rate confirming
    the redundant-recompute hypothesis, target-PID memory unchanged
    (`9834 MiB`, identical to every uncached sample). `[U]` Its
    `7324.91s` timing sits between the best and worst uncached samples
    under this run's own (heaviest-yet) GPU contamination -- directionally
    consistent with real benefit, not yet a clean speed claim.
    **2-GPU process-per-GPU sharding, Stage 1 (Q/K/V split)**: writer
    built with Token-SIMD-matched context params, two shard-reader workers
    ran **concurrently on two genuinely clean physical GPUs** (first fully
    clean preflight of the session) and both PASSED against the real
    oracle. Measured `~1.65x` wall-clock speedup from splitting Q/K/V
    across 2 GPUs (`max(915.09s, 591.54s)` concurrent vs `1506.63s` serial
    equivalent) -- the first measured (not just designed) 2-GPU
    concurrency benefit in this project, though a single, unrepeated,
    unevenly-split sample. **12-block + released GSR head T=103 driver**
    is built and compiled (new source
    `real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head.cpp`,
    Token-SIMD-adapted refresh between every block pair, final head
    driver is built and compiled (new source
    `real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head.cpp`,
    Token-SIMD-adapted refresh between every block pair, final head
    evaluator) with its own T=103 12-block+head fixture (a new
    `export_fixture_t103.py` sibling was required -- the existing exporter
    hard-rejects any T other than 2) and passing local contract, but **the
    actual 15-45 hour run was deliberately not launched** pending an
    explicit go and a quiet host window (a new, much stricter quiet-window
    gate requiring load `<50` and zero compute processes across all 8 GPUs,
    sustained 5 polls, was written and live-validated as correctly
    reporting "not quiet" against the actual host). See
    [tasks.md](tasks.md)'s 2026-07-31 "Three parallel tracks" entry for
    full detail, the revised 16-45h time estimate, and a mechanical
    concurrent-edit drift in the shared build script that was caught and
    fixed (all 442 local tests pass).
    `[V]` **2-GPU Stage-2 MLP-chunk sharding is DROPPED as infeasible on
    the pinned FIDESlib** (commit `786c7600`): the library exposes no
    `Ciphertext` serialization (only `CryptoContext`/`PublicKey`/
    `PrivateKey` in `Serialize.hpp`), so the cross-GPU `EvalAdd` partial-sum
    merge cannot be transported between two processes -- both Stage-2 sources
    fail `nvcc` on exactly the `SerializeToFile`/`DeserializeFromFile`
    ciphertext call, everything else compiles clean. The merge math itself is
    exact (`[V]` local contract `max abs error 2.84e-14`); only the transport
    is unavailable. Stage-1 Q/K/V sharding (which serializes only context+keys,
    never a ciphertext) is unaffected and remains valid. See
    [tasks.md](tasks.md)'s 2026-08-01 "2-GPU Stage-2 MLP-chunk sharding: NOT
    buildable on the pinned FIDESlib" entry.
    `[V]` **Combined-lever correctness micro-gate PASSES**: the CPU-side
    diagonal-vector cache and 2-GPU Q/K/V Stage-1 sharding -- each previously
    proven correct in isolation -- run together for the first time and both
    concurrent shard workers pass against the real T=103 oracle at the
    unchanged `~1e-9` band, with an exact per-shard cache hit/miss invariant
    (`(TOKEN_GROUPS*BSGS_N1*BSGS_N2 - 1) * requested.size()` hits). A first
    attempt failed closed on an arithmetic bug in the new invariant itself
    (not the cache or the sharding); fixed, cross-checked against an
    independent Python simulation, and re-run clean. Timing not claimed as a
    speedup (host under concurrent contention from the Stage-2 workstream
    above). See [tasks.md](tasks.md)'s 2026-08-01 "Combined-lever correctness
    micro-gate" entry.
