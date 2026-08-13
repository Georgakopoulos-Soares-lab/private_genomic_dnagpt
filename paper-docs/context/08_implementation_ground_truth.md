# Implementation ground truth

Derived by reading the implementation, not the prose. Where this note and any other document
disagree about what the system does, this note wins; where it and the evidence ledger disagree
about a number, raise it as a defect.

Tags: `[V]` verified in code, `[A]` derived, `[U]` could not verify.

**Primary sources**

| What | Where |
|---|---|
| Frozen layout and operation-count contract | `fhe/gpu_real_scheme_b/src/simd_t103_schedule.hpp` |
| Single-block driver (the measured one) | `fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp` |
| Fixture and reference generator | `fhe/multiblock/export_fixture_t103.py` |
| Checkpoint key contract | `fhe/realweights/contract.py` |
| Upstream model | `DNAGPT/dna_gpt/model/gpt.py` |

---

## 1. The verification chain is two-stage, and the paper should say so

This is the most important thing in this note. It is a genuine methodological strength that the
current draft undersells.

`[V]` The encrypted result is **not** compared against a reimplementation that nobody checked. The
chain is:

```
released checkpoint
   │  strict load; the set of missing keys is asserted exactly     contract.py:145
   ▼
upstream PyTorch block  ──(deep-copied to float64 as a shadow)──►  upstream64
   │
   │  an independent NumPy float64 re-derivation of the block
   ▼
manual reference  ──assert np.allclose(manual, upstream64, rtol=2e-5, atol=2e-5)──►  per block,
   │              all twelve, raising AssertionError on divergence   export_fixture_t103.py:175-198
   ▼
fixture weights + oracle__block_output.bin
   │
   ▼
encrypted evaluation  ──compared at 4.64e-9──►  the reference
```

`[V]` The classifier head is validated the same way, against upstream logits at the same tolerance
(`export_fixture_t103.py:224-240`).

So the reference the encrypted run reproduces has itself been shown to reproduce the released
PyTorch model. **State both stages in §4 and §7.** Reporting only the second invites the obvious
objection that the encrypted circuit and its reference could be wrong together.

### The released model has no biases, and that is verified rather than assumed

`[V]` `DNAGPT/dna_gpt/model/gpt.py` defaults to `bias=True` throughout (`Block` at :153-158 gives
`LayerNorm`, `c_attn`, `c_proj`, `c_fc`, `c_proj` biases). The evaluated 0.1b checkpoint, however,
carries **no bias tensors**: `BLOCK0_EXPORT_KEYS` (`contract.py:47-53`) maps only `.weight` keys,
and the checkpoint is loaded with the missing-key set asserted exactly against
`EXPECTED_MISSING_KEYS` (`contract.py:38-45`), which contains only unrelated numeric-head keys. A
bias tensor expected by the module but absent from the checkpoint would appear in `missing_keys`
and fail that assertion. `export_fixture_t103.py:366` accordingly records `"bias": False`.

`[V]` The encrypted circuit therefore omits biases because the model has none — not as a
simplification. Worth one clause in §5, because a reader who knows GPT will otherwise assume a
missing affine term.

---

## 2. Data layout

`[V]` `simd_t103_schedule.hpp:38-45`:

```
physical_slot(copy, feature, lane) = (copy · PACK_WIDTH + feature) · BATCH + lane
```

with `COPIES=4`, `PACK_WIDTH=1024`, `BATCH=8`, giving `4 · 1024 · 8 = 32,768` slots — half the ring
degree `N = 65,536`. Token lane is the **innermost** index, so a rotation by `BATCH·k` moves whole
features and a rotation by `k < BATCH` moves token lanes.

`[V]` Padding: `D = 768` occupies features `0..767`; features `768..1023` are padding, and
`static_assert(HEADS_PER_COPY · TOKEN_BATCH <= PACK_WIDTH - D)` reserves part of that padding for
the per-head score staging area used by attention (`t123.cpp:132`).

`[V]` `GROUPS = ⌈103/8⌉ = 13`, and `static_assert(T % BATCH == 7)` freezes the tail: the last group
carries 7 live tokens and **one** padded lane. (The figure previously said seven padded lanes; that
was wrong and is fixed.)

`[V]` The four copies exist because `MLP_DIM == COPIES · D` (`t123.cpp:130`) — the MLP's 4× widening
is expressed as four parallel copies of a `D`-wide activation rather than one 3072-wide vector.

---

## 3. Every published operation count re-derives exactly

`[V]` All twelve quantities were re-derived from the layout parameters alone, independently of the
recorded run, by porting `simd_t103_schedule.hpp` and re-executing it. **All twelve agree exactly.**

| Quantity | Paper | Re-derived | Formula |
|---|---:|---:|---|
| Slots per ciphertext | 32,768 | 32,768 | `COPIES · PACK_WIDTH · BATCH` |
| Token groups | 13 | 13 | `⌈T/BATCH⌉` |
| Dense matrix products | 156 | 156 | `12 · GROUPS` |
| Serial-equivalent products | 1,236 | 1,236 | `12 · T` |
| Score tiles | 91 | 91 | `GROUPS(GROUPS+1)/2` |
| Weight tiles | 727 | 727 | `Σ |active_weight_shifts(q,k)|` over causal tile pairs |
| Ciphertext × ciphertext | 1,506 | 1,506 | `2 · 727 + 4 · GROUPS` |
| Ciphertext × plaintext | 177,734 | 177,734 | `159,744 + 17,448 + 360 + 104 + 78` |
| Explicit rotations | 8,173 | 8,173 | `2,821 + 4,836 + 360 + 156` |
| Reduction calls | 8,776 | 8,776 | `HEADS · 727 + 4 · GROUPS` |
| Client crossings | 857 | 857 | `91 + 727 + 3 · GROUPS` |
| Logical instances | 129,162 | 129,162 | `206 + 412 + 2 · HEADS · T(T+1)/2` |

`[A]` The packing reduction is `1236/156 = 7.923`, so **7.92× is correct**.

`[A]` The ciphertext–plaintext decomposition, which the paper should give because it locates the
remaining optimization target:

| Component | Count | Source |
|---|---:|---|
| dense diagonals | 159,744 | `156 · PACK_WIDTH` |
| attention masks and score isolation | 17,448 | `2 · HEADS · 727` |
| lane masks | 360 | `4 · 90` unique cached lane shifts |
| MLP copy masks | 104 | `8 · GROUPS` |
| LayerNorm scalars | 78 | `6 · GROUPS` |

`[V]` The 17,448 mask plaintexts are the ones still re-encoded on every use — this is the "roughly
17,400" figure in the optimization section, and it is exact.

`[A]` The 129,162 logical instances decompose as: LayerNorm site 1, 103; LayerNorm site 2, 103;
GELU, 412 (`= T · COPIES`); softmax scores in, 64,272; softmax weights out, 64,272 (each
`= HEADS · T(T+1)/2`). Softmax dominates at **99.5%** of all values crossing the boundary. Worth
stating: the boundary cost is essentially the attention nonlinearity, not LayerNorm or GELU.

---

## 4. Protocol pseudocode

Transcribed from the driver. Line references are to the single-block driver.

### Algorithm 1 — EvaluateBlock  (`t123.cpp:762-935`)

```
Input : ct_in[0..G-1]           encrypted activations, G = 13 token groups
        W                       public released weights (no biases)
Output: ct_out[0..G-1]

# Stage 1 — LayerNorm and Q/K/V                                        :780-790
for g in 0..G-1:
    n      ← LayerNorm(ct_in[g], active(g), γ_1)                       # Algorithm 2
    baby   ← BabyRotations(n)                                          # 31 rotations, reused
    Q[g]   ← PackedMatMul(baby, W.q)                                   # Algorithm 4
    K[g]   ← PackedMatMul(baby, W.k)
    V[g]   ← PackedMatMul(baby, W.v)
FlushEncodedWeights()                    # Q/K/V weights never recur   :793

# Stage 2 — causal attention scores                                    :798-833
for k_g in 0..G-1:
    Ksh ← {LaneShift(K[k_g], δ) : δ needed by any query group ≥ k_g}    :800
    for q_g in k_g..G-1:
        S ← 0
        for δ in ActiveWeightShifts(q_g, k_g):
            P ← Q[q_g] ⊙ Ksh[δ]                       # ciphertext × ciphertext
            for h in 0..HEADS-1:
                s ← ReduceSum(P ⊙ headmask(h))        # over PACK_WIDTH, stride BATCH
                S ← S + s ⊙ isolate(q_g,k_g,δ,h, 1/√HEAD_DIM)
        Client.ReduceScoreTile(S, q_g, k_g)           # decrypt; 91 crossings total

Client.FinalizeAttention()        # stable softmax over the whole causal matrix   :834
discard Q, K                                                            :835-836

# Stage 3 — attention context                                          :839-858
for k_g in 0..G-1:
    Vsh ← {LaneShift(V[k_g], δ) : δ needed}
    for q_g in k_g..G-1:
        for δ in ActiveWeightShifts(q_g, k_g):
            A   ← Client.EmitWeightTile(q_g, k_g, δ)   # re-encrypt; 727 crossings
            C[q_g] ← C[q_g] + A ⊙ Vsh[δ]               # ciphertext × ciphertext

# Stage 4 — projection and first residual                              :862-879
for g: P[g] ← PackedMatMul(BabyRotations(C[g]), W.attn_proj)
for g: R[g] ← ct_in[g] + P[g]                          # residual, no rescale
FlushEncodedWeights()                                                   :879

# Stage 5 — LayerNorm, MLP, GELU, second residual                      :882-930
for g in 0..G-1:
    n2 ← LayerNorm(R[g], active(g), γ_2)
    require_remaining_depth(level(n2), 4)              # fails closed    :885
    baby ← BabyRotations(n2)
    for c in 0..COPIES-1:  H[c] ← PackedMatMul(baby, W.mlp_fc[c])
    Hpacked ← Σ_c  H[c] ⊙ copymask(c)                  # gather 4 chunks into 4 copies
    Gact    ← Client.GeluBoundary(Hpacked, active(g))  # 13 crossings
    for c in 0..COPIES-1:  M[c] ← PackedMatMul(BabyRotations(Gact ⊙ copymask(c)), W.mlp_proj[c])
    ct_out[g] ← R[g] + Σ_c M[c]
```

`[V]` Note the ordering in Stage 2/3: **all 91 score tiles are collected before any softmax is
computed**, because `FinalizeAttention` normalizes over complete rows of the causal matrix
(`t123.cpp:629-665`). The client therefore holds the entire `T × HEADS × T` score array at once. At
`T=103` that is 103·12·103 float64 ≈ 1.0 MiB, so it is not the client's memory driver, but it is a
real synchronization point: attention cannot be streamed under this design without changing the
softmax normalization.

### Algorithm 2 — LayerNorm  (`t123.cpp:1168-1181`)

```
mean     ← ReduceSum(x) ⊙ (1/D)                        # broadcast over the packed width
centered ← x − mean
var      ← ReduceSum(centered ⊙ centered) ⊙ (1/D)  + ε      # ε = 1e-5, matches upstream
inv      ← Client.InvSqrtBoundary(var, active)         # decrypt, 1/√·, re-encrypt
return   (centered ⊙ inv) ⊙ γ                          # γ is public plaintext
```

`[V]` The server computes both statistics under encryption and crosses the boundary only for the
inverse square root. `[V]` `ε` is added **before** the crossing, server-side. `[V]` There is no `β`
because the checkpoint has none (§1). `[A]` Depth cost: two ciphertext×ciphertext multiplications
per LayerNorm invocation, which is the `4 · GROUPS` term in the ciphertext×ciphertext count (two
sites × two multiplications × 13 groups).

### Algorithm 3 — ClientBoundary  (`t123.cpp:514-538`, `540-568`, `583-627`, `667-715`)

```
ClientBoundary(ct, f):
    pt     ← Decrypt(sk, ct)                  # only the key holder can do this
    v      ← Decode(pt)                        # 32,768 real slots
    v'     ← f(v)                              # evaluated exactly, in double precision
    return Encrypt(pk, Encode(v'))             # fresh ciphertext at level 0
```

`[V]` Four instantiations, all structurally identical: `invsqrt_boundary` (`f = 1/√·`),
`gelu_boundary` (`f = ` exact tanh-form GELU, applied to `COPIES · D · active` live slots),
`reduce_score_tile` (decrypt only, accumulating into the client's score array), and
`emit_weight_tile` (encrypt only, from the client's softmax output).

`[V]` Re-encryption is always at level 0, so **every crossing fully resets the multiplicative
budget**. This is why depth 13 suffices for a 12-block model: depth bounds the longest run between
two boundaries, not the depth of the whole network. This is the central mechanism of the design and
§5 should state it in exactly these terms.

`[U]` No noise flooding is applied at re-encryption. The client returns a freshly encrypted exact
value, never a decryption, but the server chooses which ciphertext gets decrypted, 857 times per
block. The measured `4.64e-9` error against a `4e-2` tolerance leaves ample room to add flooding;
budgeting it is future work and the limitations section already says so.

### Algorithm 4 — PackedMatMul, baby-step giant-step  (`t123.cpp:1183-1203`, `1264-1295`)

```
BabyRotations(x):                                      # once per group, shared by all matmuls
    return [x, Rot(x, BATCH·1), ..., Rot(x, BATCH·(N1-1))]        # 31 rotations

PackedMatMul(baby, W):
    ℓ      ← level(baby[0])
    diag   ← EncodedDiagonals(W, ℓ)                    # cached; see §5
    result ← 0
    for j in 0..N2-1:                                   # giant steps, N2 = 32
        inner ← Σ_{i=0}^{N1-1}  baby[i] ⊙ Clone(diag[N1·j + i])
        if j ≠ 0:  inner ← Rot(inner, BATCH·N1·j)
        result ← result + inner
    return result
```

`[V]` `N1 = N2 = 32`, `N1·N2 = PACK_WIDTH = 1024`. Rotation accounting: `13 · 31 = 403` baby
rotations if counted per group — but the contract charges `GROUPS · 7 · (N1-1) = 2,821`, i.e. seven
`BabyRotations` calls per group (one for Q/K/V shared, one for the attention projection, one for the
MLP up-projection, four for the MLP down-projection chunks), plus `156 · 31 = 4,836` giant-step
rotations. `[V]` Both terms re-derive exactly.

`[V]` The diagonal at index `d` for giant step `j` is built with a rolled row index
(`t123.cpp:1027-1031`), which is what lets one encoded plaintext serve every row of a giant step.
Rows and columns outside `D` contribute zero, so the `768 → 1024` padding costs multiplications but
never corrupts a result.

### Algorithm 5 — LaneShift  (`t123.cpp:1297-1314`)

```
LaneShift(x, δ):                                       # δ ∈ 1..BATCH-1
    a ← Rot(x, δ)                                      # borrows from the next feature
    b ← Rot(x, δ − BATCH)                              # borrows from the same feature
    return a ⊙ mask_nowrap(δ)  +  b ⊙ mask_wrap(δ)
```

`[V]` Two rotations and two masked multiplications per shift, which is the `4 · 90 = 360` lane term
in both the rotation and the plaintext counts. `[V]` `build_group_shifts` (`:1316-1341`) computes
which `δ` any later query group will need and materializes each **once per key group**, giving 90
unique cached shifts rather than one per tile — a real optimization already in the code that the
paper does not currently mention.

---

## 5. Optimization mechanisms, verified against the code

| Claim | Verdict | Evidence |
|---|---|---|
| Baseline re-encoded every diagonal at every multiplication, 159,744 per block, ~90% redundant | `[V]` | `t123.cpp:963-966` states it; `156 · 1024 = 159,744`, and 144 of 156 lookups are cache hits (`:1003-1004`), so 92% |
| Encode once into an unloaded template, clone per use | `[V]` | `clone_encoded`, `:975-982` — copies `cpu` and `parent_context`, sets `loaded = false`, `gpu = 0` |
| Why a clone and not object reuse | `[V]` | `:969-973` — the backend's `LoadPlaintext()` returns early when `loaded` is set, so a template loaded at one level was reused at another with stale limbs; `~PlaintextImpl` only evicts when `(loaded && gpu)`, so the clone's per-operation lifetime is identical to the baseline's |
| Parallel batched encoding across cores | `[V]` | `#pragma omp parallel for schedule(dynamic)` at `:1050`, over all 1,024 diagonals |
| Encode at the level of use | `[V]` | cache keyed on `(&weight, level)` at `:1007`; `MakeCKKSPackedPlaintext(..., level, ...)` at `:1053` |
| Encoding at use level is exact | `[V]` | `:1000-1002` — the library would drop a level-0 plaintext to that level anyway |
| Flush between stages bounds the peak to the largest stage | `[V]` | `flush_diagonal_plains()` called at `:793` (after Q/K/V) and `:879` (after the attention projection); rationale at `:989-994` |
| Fails closed on any schedule deviation | `[V]` | `:1732-1741` compares matmul, ciphertext×ciphertext, ciphertext×plaintext, rotation, reduction, crossing and logical-instance counts against frozen constants and aborts on mismatch |
| Depth is checked, not assumed | `[V]` | `require_remaining_depth(level, 4)` before every MLP group, `:885-887` |
| Thread affinity / NUMA-local allocation | `[V]` | not in the driver — set by the launcher (`OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `numactl --localalloc`), `kimon/configs/config3_all_opts/block_T123_v2_slurm.sbatch:46-54` |
| Release host-side key copies after upload | `[V]` | the `_v3` variant only; not present in the measured driver |

**Correction for the ledger:** the measured 652 s run used the driver *without* the host key-map
release, which lands in the `_v3` variant. `evidence/optimizations.yaml` lists
`opt.release_host_key_copies` in the retained chain that produced the reported result. It should be
moved to a "validated, not in the measured run" position, or the reported run re-identified.

---

## 6. Claims audit

Adversarial pass over `context/00_terminology.md`, `context/01_scenario_and_motivation.md`, and the
evidence ledger.

### Must fix

1. **`opt.release_host_key_copies` is credited to the measured run but is not in it.** See §5. This
   is the one substantive defect found.

2. **"Two clean-host repeats … pegged at 3,900–4,700 % CPU" is used to support the ~2.1 h baseline,
   but the two facts are separate.** The CPU-saturation observation is `[V]`; the 2.1 h wall time
   remains attested only. Already tracked in `open_provenance`; keep them distinct in prose.

3. **The figure claim "the four copies exist because the MLP expands to four times the hidden
   width" is right but incomplete.** Part of the padding also stages per-head attention scores
   (`static_assert` at `t123.cpp:132`). One clause, so a reader reconstructing the layout does not
   come up short.

### Should sharpen

4. **The verification chain is two-stage and the draft states one stage.** §1 above. This is the
   biggest missed opportunity in the current draft.

5. **"Depth 13" needs its mechanism attached.** Depth bounds the longest run between two client
   boundaries, not the network's depth, because re-encryption returns to level 0. Without that
   sentence, depth 13 for a 12-block model looks impossible.

6. **Softmax is 99.5% of the boundary traffic.** The paper treats the three nonlinearity classes as
   comparable. They are not, and this reframes the optimization discussion: the boundary cost is
   attention.

7. **"No biases" should be stated as verified, not omitted silently.** §1.

### Verified correct — no change needed

- All twelve operation counts, the 7.92× packing reduction, and the 17,448 mask encodings.
- The slot indexing function, the 13 groups, and the single padded lane.
- `ε = 1e-5` matching upstream; exact tanh-form GELU; stable (max-subtracted) softmax.
- 128-bit classical security, ring 65,536, 32,768 slots, depth 13, 3 key-switching digits,
  50/60-bit scaling.
- The clone-versus-reuse mechanism, in every detail the ledger states.
- The server never decrypts and never holds the secret key: all four boundary methods are on
  `Client`, which is the sole holder of `keys_.secretKey`.

---

## 7. In the code, missing from the paper

Worth adding; each is a real design decision currently invisible to a reader.

1. **Lane-shift sharing.** `build_group_shifts` materializes each needed shift once per key group —
   90 shifts instead of one per tile alignment. A structural saving the paper never claims.

2. **Contracts that fail closed.** Every run asserts the full operation schedule, the boundary
   counts, and the remaining depth before proceeding, and aborts on any deviation. This is why "the
   operation schedule is unchanged" is a checked invariant rather than an assurance, and it is the
   single strongest support for attributing the speedup to systems work. Put it in §6.

3. **Level-0 re-encryption as the depth mechanism.** See §6.5.

4. **The global softmax synchronization point.** All 91 score tiles are collected before any
   normalization. It bounds how the design could be streamed or pipelined, and belongs in the
   discussion of remaining headroom.

5. **Padding is arithmetically inert.** Out-of-range rows and columns contribute exact zeros, so the
   `768 → 1024` padding costs multiplications and never affects accuracy. One clause in §5 forecloses
   a reviewer question about the padding's numerical effect.

6. **`EvalFastRotation` with a shared precomputation** is used for baby rotations and cross-copy
   replication (`:1190`, `:1350`), which is why 31 rotations per call are affordable.
