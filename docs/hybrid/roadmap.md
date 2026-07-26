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
   in [../roadmap.md](../roadmap.md). **Active next step.** The naive ×12-block T=2
   linear extrapolation ([tasks.md](tasks.md), `~74.6 min` total) is `[A]`-only; real
   sequence-length scaling is `[U]` and blocked on designing a general causal-attention
   Scheme B circuit, since the current T=2 schedule uses a closed-form identity specific
   to a two-token window.
