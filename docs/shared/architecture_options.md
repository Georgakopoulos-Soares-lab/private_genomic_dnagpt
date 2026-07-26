# Architecture options: pure CKKS vs. hybrid vs. scheme-switching

## Trigger

Every chained multi-block gate under the original Scheme A contract (one uninterrupted
CKKS ciphertext lineage, zero intermediate decrypt) has failed closed on GPU memory, not
accuracy, at `multiplicative_depth` in `{50, 58, 64}`:
`fhe_fides_gpu_multiblock_bisect_depth50_FAIL_20260724.json`,
`fhe_fides_gpu_multiblock_bisect_depth58_backtrace_FAIL_20260724.json`,
`fhe_fides_gpu_multiblock_multigpu_2gpu_sigsegv_setupconstants_FAIL_20260725.json`. The
symbolized backtrace traces every failure to `GPUmalloc` during rotation-key/bootstrap
plaintext loading, i.e. `batch_slots`/rotation-key footprint exceeding one A100's 80GB —
not a depth- or digit-count-specific bug. Two physical GPUs additionally SIGSEGV inside
`FIDESlib::SetupConstants`, an unexercised, less-tested code path per FIDESlib's own docs.

Root cause: `multiplicative_depth=64` exists only because block 0's single end-of-block
bootstrap must buy back 35 levels in one shot for all of block 1
(`require_remaining_depth(restored, 35, ...)` in `dnagpt_two_block_fides.cpp`). Those 35
levels are consumed almost entirely by degree-13 Chebyshev polynomial approximations of
non-polynomial functions — LayerNorm invsqrt, attention's T=2 sigmoid identity, GELU —
each costing roughly 4 levels via Paterson–Stockmeyer, per block. **The nonlinear
function approximations, not the linear algebra, are what is driving depth, and depth is
what is driving GPU memory past the point that fits on one card.**

## Three candidate architectures

| | A. Pure non-interactive CKKS (current) | B. Hybrid client-assisted CKKS | C. CKKS↔FHEW scheme switching |
|---|---|---|---|
| Mechanism | Everything — linear and nonlinear — stays in one CKKS ciphertext lineage; nonlinearities are Chebyshev polynomial approximations; occasional bootstrap refreshes depth | Server does linear algebra (QKV, attention matmuls, FFN) under CKKS on GPU (FIDESlib, unchanged); at each nonlinearity, ciphertext returns to the client, who decrypts, computes the exact function in plaintext on its own data, and re-encrypts | Arithmetic stays in CKKS; at each nonlinearity, the ciphertext is scheme-switched to FHEW/TFHE, evaluated *exactly* via LUT/programmable bootstrapping (no polynomial degree, no approximation error), then switched back to CKKS — server-only, no client round trip |
| Threat model | Fully non-interactive; server never has plaintext or key | Server never has plaintext or key; client (= data owner, already holds the secret key) decrypts only ciphertexts derived from its own query, at explicit pre-declared boundaries | Fully non-interactive; server never has plaintext or key |
| Solves the actual bottleneck? | No — same Chebyshev/depth/bootstrap cost that is currently failing | Yes — deletes Chebyshev evaluation and the resulting depth pressure entirely | Yes — replaces polynomial approximation with LUT evaluation; FHEW bootstrap also refreshes noise as a side effect |
| Reuses existing code | Yes, unmodified | ~90%: keep all FIDESlib GPU matmul/rotation/plaintext-packing code; delete `EvalChebyshevFunction` calls for invsqrt/sigmoid/GELU; add decrypt/compute/re-encrypt at each boundary | Partial: keep CKKS linear-algebra code; need new FHEW context, LUT construction, and switching glue — none of this exists in the repo yet |
| GPU support | FIDESlib, proven (this repo's own evidence) | FIDESlib for 100% of the encrypted compute; boundary round trips are plaintext CPU work (fast) | **None.** OpenFHE mainline scheme switching is CPU-only; no public library GPU-accelerates the FHEW half. "Chameleon" (2024) claims up to 67x GPU speedup for scheme switching but is a research paper, not an integrated, usable artifact |
| Precedent | This repo (partial) | THOR, ARION, Hamster, SHAFT, EncFormer — published, minutes-scale, BERT/GPT-2-class hybrid FHE(+MPC) results use exactly this split | None at transformer scale; primitive-level research only |
| Novelty for a paper | Highest, if it can be finished | Lower — well-trodden pattern in the literature | Highest of all three, but unproven risk |
| Engineering risk to a first number | High — root cause identified but no completed fix yet | Low — deletes code, does not add a new dependency | High — new library integration, no GPU path, no existing OpenFHE/FIDESlib interop for switching |
| Interactivity | None | Client must stay reachable across every block's nonlinearity boundaries (~4 round trips/block x 12 blocks) | None |

## Decision

**Adopted: Scheme B (hybrid client-assisted CKKS).** Rationale:

- It directly removes the measured bottleneck (Chebyshev depth pressure -> GPU memory
  exhaustion) rather than working around it.
- It reuses the GPU linear-algebra implementation and Phase-A oracles already proven in
  this repo; no new cryptographic backend, container, or library is introduced.
- The privacy requirement stated for this project — no sensitive DNA data is exposed to
  or inferable by the untrusted compute provider — is satisfied: the client is the data
  owner, already holds the secret key, and only ever decrypts ciphertexts derived from
  its own query. The server never observes plaintext, a partial decrypt, or the secret
  key at any point.
- Scheme C (scheme switching) is the more novel, fully non-interactive answer and is
  **not rejected** — it is deferred as future/stretch work because no GPU-accelerated
  implementation exists anywhere to build on; attempting it first would mean restarting
  primitive-level validation (a new Phase-B-style CPU/GPU parity effort) before any
  DNAGPT-shaped result, which conflicts with "as fast as possible."
- Scheme A's existing evidence (complete real-weight block 0, refresh gate, both
  multiblock failures) is **not discarded**. It is retained as the frozen non-interactive
  baseline/ablation for the same paper: "Scheme A closes one full block correctly, then
  hits a quantified, root-caused GPU memory wall at chained composition; Scheme B
  removes that wall by relocating nonlinear evaluation off the encrypted lineage."

This reverses, for the hybrid *protocol shape* only, the earlier rejection in
[backend_selection.md](backend_selection.md) of "Concrete ML hybrid LLM." That
rejection was specific to the Concrete ML/TFHE-rs *backend* (Boolean/integer-only,
no CKKS, ~1.2–2x GPU speedup, 2.2–18MB/token ciphertext expansion) and to using it as a
silent, undocumented shortcut. Scheme B keeps OpenFHE/FIDESlib CKKS as the sole
cryptographic backend and documents the client-decrypt boundary explicitly as a first
class, measured part of the protocol, not an implicit compromise.

## Scheme B acceptance contract

Replaces, for Scheme B runs only, the "no intermediate decryption feeding evaluation"
line in `docs/roadmap.md`'s "One rule." Scheme A's contract is unchanged for Scheme A
runs.

Every Scheme B correctness gate uses:

- `HEStd_128_classic`, one context and key lineage, same as Scheme A;
- linear algebra (linear maps, attention matmuls, residual adds) stays in one
  uninterrupted encrypted lineage on GPU;
- decrypt boundaries occur **only** at pre-declared nonlinearity ops (LayerNorm,
  softmax/sigmoid identity, GELU) and are fixed before the run, never chosen adaptively
  from decrypted content;
- the party performing every decrypt/re-encrypt step is the data owner (secret-key
  holder) operating on its own data only — the untrusted compute provider never executes
  or observes a decrypt;
- global and worst-token relative-infinity error no greater than `4e-2`, same oracle as
  Scheme A;
- every run additionally records: round-trip count, boundary count, and wall time split
  between GPU-encrypted compute and client-side plaintext compute, so the interactivity
  cost is measured, not hidden.

## Next steps

1. Update `docs/roadmap.md` "One rule" and `docs/overview.md`/`CLAUDE.md` Phase B
   framing to name both schemes explicitly (done alongside this file).
2. Scope the first Scheme B gate: reuse the existing real-weight block-0 CKKS graph,
   remove the Chebyshev invsqrt/sigmoid/GELU evaluation, add decrypt/compute/re-encrypt
   at each of those three boundaries, and re-measure against the same Phase-A oracle and
   `4e-2` gate used by every Scheme A result.
3. Because Scheme B has no bootstrap-driven depth pressure, first estimate whether
   `multiplicative_depth` can be cut sharply below 43 — cheaper context, smaller
   `batch_slots`/rotation-key footprint, and a plausible route past the exact memory
   wall that blocked the Scheme A two-block gate.
4. Keep Scheme A's evidence frozen and cite it as the ablation baseline in the eventual
   paper; do not re-run or rewrite existing Scheme A result files.
