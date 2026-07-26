# Tasks — Scheme B (hybrid client-assisted CKKS), active

The active FHE architecture. See [../roadmap.md](../roadmap.md) for the shared
acceptance contract, [../shared/architecture_options.md](../shared/architecture_options.md)
for the full Scheme A/B/C comparison and decision rationale, and
[../pure/tasks.md](../pure/tasks.md) for the frozen Scheme A evidence this pivot replaced.

## Architecture pivot: Scheme A frozen, Scheme B active (2026-07-25)

Three independent chained-composition attempts (`fhe_fides_gpu_multiblock_bisect_depth50_FAIL_20260724`,
`..._bisect_depth58_backtrace_FAIL_20260724`,
`..._multigpu_2gpu_sigsegv_setupconstants_FAIL_20260725`) all fail closed on GPU memory
exhaustion, not accuracy, and all trace to the same root cause: block 0's single
end-of-block bootstrap buying back 35 levels for all of block 1 in one shot, driven by
degree-13 Chebyshev approximation of LayerNorm invsqrt, the attention sigmoid identity,
and GELU. Full comparison of alternatives in
[../shared/architecture_options.md](../shared/architecture_options.md).

**Decision:** freeze pure non-interactive CKKS (**Scheme A**) as the paper's baseline —
its existing evidence (complete real-weight block 0, refresh gate, the three failures
above) stands unchanged. Active development moves to **Scheme B**: hybrid
client-assisted CKKS. Server-side linear algebra (FIDESlib GPU) stays in one encrypted
lineage; only the data-owning client (already the secret-key holder) decrypts, at
pre-declared nonlinearity boundaries (LayerNorm, attention nonlinearity, GELU), its own
data, then re-encrypts. This removes the Chebyshev-driven depth pressure that caused
every Scheme A composition failure, while keeping the untrusted compute provider fully
blind to plaintext at every step.

**Next concrete step (not yet run):** reuse the existing real-weight block-0
CKKS/FIDESlib graph, delete the three `EvalChebyshevFunction` calls, add
decrypt/compute/re-encrypt at each boundary, and re-measure against the unchanged
Phase-A oracle and `4e-2` gate, recording round-trip count and the GPU-encrypted vs.
client-plaintext wall-time split per Scheme B's acceptance contract in
[../roadmap.md](../roadmap.md).

### Scheme B toy prototype (2026-07-25)

`fhe/block/toy_block_ckks_scheme_b.py` (D=8, T=4, 2 heads) implements the
`Client`/`EvalOnlyContext` protocol: the client holds the secret key and
performs `cross_boundary()` decrypt/exact-compute/re-encrypt at LN1, the
attention softmax, LN2, and GELU; the server-side context asserts zero
decrypts. Two real bugs were found and fixed en route: (1) `DEPTH=24` landed
in a needlessly expensive `ring_dim=131072` bucket for a measured max level of
only 14-16 — reduced to `DEPTH=20` (`ring_dim=65536`); (2) OpenFHE's
`EvalSum(ct, n)` is a sliding-window sum, not a per-block broadcast (only slot
0 holds the true sum) — `sum_bcast()` now isolates slot 0 via a one-hot mask
and replicates via rotation-doubling.

```bash
source .venv/bin/activate
python fhe/block/toy_block_ckks_scheme_b.py
```

Evidence: `fhe_toy_block_scheme_b_native_cpu_20260725.json` — `[V]` PASS,
`rel_inf=2.22e-12`, 16 client round trips (168 values crossed total), depth 20,
`ring_dim=65536` (vs. Scheme A's toy at depth 49/`ring_dim=131072`). The
1433s wall time is native CPU on a **heavily contended shared Brev host**
(concurrently observed load average 1000-1460 throughout this session) and is
explicitly not used as a GPU-extrapolation input.

### Scheme B real D=768/T=2 GPU gate (2026-07-25)

`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b.cpp` forks
`fhe/gpu_real_sigmoid`'s FIDESlib source and replaces the three
`EvalChebyshevSeries` calls (LN1/LN2 invsqrt, attention sigmoid, GELU x4) with
genuine `Client::cross_boundary()` round trips, using the exact elementwise
function instead of a calibrated-domain polynomial. Same released fixture,
same packing, same `4e-2` oracle gate as Scheme A's `gpu_real_sigmoid` gates.
No public per-block domain contract is needed (the client computes the exact
function on the true decrypted value, whatever its magnitude).

```bash
FIDESLIB_ARCH=80-real BUILD_JOBS=4 fhe/gpu_real_scheme_b/build_in_fideslib.sh

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_scheme_b/run_scheme_b.sh 0 ln1 "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_ln1_scheme_b_a100_20260725.json
```

Evidence: `fhe_fides_real_d768_t2_ln1_scheme_b_a100_20260725.json` — `[V]` PASS,
`rel_inf=2.35e-10`, 2 round trips, `ring_dim=65536` at `multiplicative_depth=16`
(vs. Scheme A's `ring_dim=131072` at depth 43 — LN1's segment level_max is only
2, far below either budget, but Scheme B never needs to reserve depth for the
*rest* of the block since every nonlinearity resets to level 0). The 2.41s
`encrypted_evaluation_seconds_gpu` splits into `server_linear_algebra_seconds`
(1.04s, GPU BSGS/mean/variance ops) and `client_boundary_seconds_total` (1.37s,
2x decrypt+invsqrt+re-encrypt of a `SLOTS=4096` vector) — this split is the key
input for reasoning about how much of Scheme B's cost is server-GPU vs.
client-CPU-boundary work.

`[U]` The `attention` and `full` gates were attempted the same session but the
run was killed after 8.5+ minutes of CPU-bound rotation-key generation
stalling under extreme host contention (load average ~1000-1127 on this
255-core shared machine; the GPU itself was idle at 1% utilization) — for
comparison, Scheme A's equivalent keygen took 25.9s on a presumably quieter
host. `fhe/gpu_real_scheme_b/wait_and_run_scheme_b.sh` now runs detached on the
Brev host, polling load average and GPU occupancy every 60s, and will launch
both remaining gates unattended once capacity is available (log at
`gpu_real_scheme_b_v1/fhe/gpu_real_scheme_b/scheme_b_orchestrator.log` on the
host). The full-block-0-to-e2e Scheme B extrapolation is blocked on those two
results — LN1 alone is too small a fraction of one block to extrapolate from.

### Scheme B real D=768/T=2 GPU gates complete: attention + full block-0 (2026-07-26)

The orchestrator (`fhe/gpu_real_scheme_b/wait_and_run_scheme_b.sh`) ran unattended
overnight and completed both remaining gates once host load dropped, launching each
on whichever physical GPU its own idle scan picked at that moment (GPU 0 for
`attention`, GPU 4 for `full`) — proof its capacity-detection logic generalizes across
GPU indices, not just the one used in prior manual runs. One real orchestrator bug was
found and fixed en route: `wait_for_capacity()`'s internal `log()` calls used `tee`,
which writes to stdout; since the caller captures the function's return value via
`gpu="$(wait_for_capacity)"`, every poll line printed during a multi-iteration wait
leaked into the captured GPU index, corrupting it into a multi-line string once passed
to `docker run --gpus device=<corrupted value>` (`exit 125`, "unresolvable CDI
devices"). Fixed by redirecting those internal log calls to stderr so only the
function's final `echo "${gpu}"` reaches the caller. Verified with a positive test
(clean single-value capture across several forced "not idle yet" poll iterations) and
a negative control against the un-patched script (reproduces the multi-line
corruption) on the live host before redeploying and restarting the orchestrator.

```bash
FIDESLIB_ARCH=80-real BUILD_JOBS=4 fhe/gpu_real_scheme_b/build_in_fideslib.sh

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_scheme_b/run_scheme_b.sh 0 attention "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_attention_scheme_b_a100_20260725.json

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_scheme_b/run_scheme_b.sh 0 full "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_block0_scheme_b_a100_20260725.json
```

Evidence: `fhe_fides_real_d768_t2_attention_scheme_b_a100_20260725.json` — `[V]` PASS,
`rel_inf=2.87e-10`, 14 round trips (12 attention sigmoids + 2 LN1 invsqrts),
`240.29s [gpu]` total = `236.61s` server (98.5%) + `3.68s` client boundary (1.5%).

Evidence: `fhe_fides_real_d768_t2_block0_scheme_b_a100_20260725.json` — `[V]` PASS,
`rel_inf=3.32e-10`, 24 round trips (12 attention sigmoids + 8 GELU chunks + 4
LN1/LN2 invsqrts), first **complete** real-weight D=768/T=2 Scheme B block: LN1, QKV,
12-head causal attention (T=2 sigmoid identity), projection, residual, LN2, 4-chunk
GELU MLP, residual, encrypted output pack. `372.27s [gpu]` total = `365.78s` server
(98.3%) + `6.49s` client boundary (1.7%), at `ring_dim=65536`/`multiplicative_depth=16`
— versus Scheme A's equivalent block-0 gate
(`fhe_fides_real_d768_t2_block0_sigmoid13_a100_asymfix2_20260724`, `2461.8s` at
`ring_dim=131072`/depth 43): roughly **6.6x faster wall time on half the ring
dimension**, because every nonlinearity resets to level 0 at the client boundary
instead of consuming Chebyshev depth.

**`[A]` Twelve-block, T=2 linear extrapolation** (naive ×12 of the measured block-0
full-gate timings, assuming key reuse across blocks and unchanged per-block cost —
not yet independently verified per additional block):

| Quantity | Value |
|---|---:|
| per-block encrypted evaluation (measured, block 0) | `372.27 s` |
| — server linear algebra | `365.78 s` (98.3%) |
| — client boundary (24 round trips) | `6.49 s` (1.7%) |
| 12-block encrypted evaluation total | `4,467.2 s` (74.45 min / 1.24 h) |
| — server total | `4,389.4 s` (73.16 min) |
| — client boundary total | `77.9 s` (1.30 min, 288 round trips) |
| one-time context keygen + input encrypt (not ×12) | `6.9 s` |
| **grand total, 12 blocks, T=2** | **`4,474.1 s` (~74.6 min)** |

Reproduction: `python3 -c "import json; b=json.load(open('results/runs/fhe_fides_real_d768_t2_block0_scheme_b_a100_20260725.json')); t=b['timings_seconds']; print(t['encrypted_evaluation']*12, t['server_linear_algebra_seconds']*12, t['client_boundary_seconds_total']*12, t['context_keygen_load']+t['encrypt_embedded_inputs'])"`

`[U]` This extrapolation is **T=2 only and does not generalize to real downstream
sequence lengths.** The T=2 attention schedule uses an exact closed-form identity
(`softmax([s0,s1])[1]=sigmoid(s1-s0)`) specific to a two-token causal window; no
general causal-attention Scheme B circuit for T>2 has been designed or measured yet.
Scaling to the real T used by GSR/GUE/mRNA (up to several hundred–thousand tokens)
therefore remains blocked on that design work, and the task-head compute is not
included above. Reporting a T>2 or full end-to-end number before that circuit exists
and is measured would violate the "never tune claims to force practical" rule.

### Scheme B client-boundary batching (2026-07-26)

**Scope before implementation.** `[V]` The existing `SLOTS=4096` layout already has
four `PACK_WIDTH=1024` copies, with only the first `D=768` slots active in each copy
(source: `real_dnagpt_fides_scheme_b.cpp`, packing constants and
`repeated_plain()`). No CKKS-parameter or slot-count change is needed:

- `[V]` For attention, each T=2 head contributes one scalar
  `score_11-score_10`, already broadcast across its copy by `attention_score()`.
  Masking head `h` into slots `[64h,64(h+1))` places all 12 deltas in disjoint
  spans of the same ciphertext. One client decrypt can therefore apply exact
  sigmoid to all active spans and re-encrypt once (source: the float64 oracle
  contract below and `attention_sigmoid_heads_batched` in the C++ source).
- `[V]` For GELU, one 768-wide MLP chunk fits in the active portion of one
  1024-slot copy, so the four chunks for one token fill the four existing copies.
  After one exact-GELU client crossing, masks isolate each chunk and rotations
  `+1024`, `-1024`, and `+2048` restore the four identical copies expected by
  the existing BSGS matmul. Eight chunks need two crossings, one per token:
  `8*768=6144` values do not fit in 4096 real slots without changing the packing
  contract (source: the float64 oracle contract below and `replicate_copies()`).
- `[V]` The measured physical count is seven crossings for the full
  gate: four unchanged per-token LN1/LN2 invsqrt crossings, one 12-head attention
  crossing, and two four-chunk GELU crossings. The evidence JSON separately
  records 24 logical nonlinearity instances so batching cannot hide protocol
  work (source:
  `fhe_fides_real_d768_t2_block0_batched_scheme_b_a100_20260726.json`).

`[V]` The correctness-first fixture contract uses the unchanged released-weight oracle,
checks the attention merge and both tokens' GELU outputs at float64
`atol=1e-12`, checks padding/copy restoration, and statically asserts that the
encrypted evaluator contains neither `Decrypt(` nor `secretKey`:

```bash
python3 fhe/gpu_real_scheme_b/test_batching_contract.py -v
```

`[V]` All three tests pass without changing `TOL=4e-2` (source: command above,
`fhe/gpu_real_scheme_b/test_batching_contract.py`, SHA-256
`60787dd5fab55ce617ee6de527fd46f5b30d3d6acdd5d135983693939f655e76`).

`[V]` The immutable GPU evidence was produced by the capacity-aware remote orchestrator.
The command below is the exact launch configuration used; immutable guards make it
skip these tags now that their JSON files exist, so a reproduction must supply new
tags rather than overwrite them (source:
`fhe/gpu_real_scheme_b/wait_and_run_scheme_b.sh` and the two evidence JSONs):

```bash
brev exec awesome-gpu-name -- "env \
  SCHEME_B_REMOTE_ROOT=/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724 \
  SCHEME_B_SOURCE_SUBDIR=gpu_real_scheme_b_batched_v2/fhe \
  SCHEME_B_FIXTURE_SUBDIR=real_fixture/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0 \
  SCHEME_B_IMAGE=dnagpt-fideslib:786c-asymfix2 \
  SCHEME_B_ATTENTION_TAG=fhe_fides_real_d768_t2_attention_batched_scheme_b_a100_20260726 \
  SCHEME_B_FULL_TAG=fhe_fides_real_d768_t2_block0_batched_scheme_b_a100_20260726 \
  /bin/bash /data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724/gpu_real_scheme_b_batched_v2/fhe/wait_and_run_scheme_b.sh"
```

Evidence:

- `fhe_fides_real_d768_t2_attention_batched_scheme_b_a100_20260726.json` —
  `[V]` PASS at unchanged `TOL=4e-2`: global/worst-token
  `rel_inf=2.18e-10`, 3 physical round trips representing 14 logical boundary
  instances (2 LN1 invsqrts + one ciphertext containing all 12 attention
  sigmoids), zero intermediate decrypts, and one final oracle decrypt. This
  verifies the attention crossing reduction from 14 to 3 without changing the
  exact T=2 attention identity or the plaintext oracle (source: the named JSON,
  SHA-256
  `4df29c742506811ef6b118491be327258b5f4dfdea1f5b6742ac20293f1c9a6f`).
- `fhe_fides_real_d768_t2_block0_batched_scheme_b_a100_20260726.json` —
  `[V]` PASS at unchanged `TOL=4e-2`: global/worst-token
  `rel_inf=3.51e-10`, 7 physical round trips representing the same 24 logical
  boundary instances as the serial protocol (4 LayerNorm + 12 attention + 8
  GELU), zero intermediate decrypts, and one final oracle decrypt. This verifies
  the complete round-trip reduction from 24 to 7, including both four-chunk
  GELU batches (source: the named JSON, SHA-256
  `07ad381e8395452035e9825d24acfc020c6a0884a1755e07e904c6ab12a715d4`).

`[U]` Do **not** use the batched runs as performance evidence. Although the
immutable JSONs honestly retain raw `client_boundary_seconds_total` fields
(attention: serial `3.6813s`, batched `1.3958s`; full: serial `6.4882s`,
batched `3.2627s`), host load surged after each confirmed-idle preflight and
reached roughly 750-1100 during evaluation. The batched total evaluation times
(`743.0s` attention, `1789.4s` full) are therefore not comparable to the clean
serial anchors. These numbers are registered only as raw, contention-affected
observations, not speedups (sources: the four serial/batched JSONs and Brev host
`gpu_real_scheme_b_batched_v2/fhe/scheme_b_orchestrator.log`).

`[U]` No real-network or artificial added-RTT experiment was attempted, so no
network-latency speedup is claimed. Context/key caching across blocks remains a
separate follow-up and was not implemented in this prototype (source: scope in
`docs/hybrid/roadmap.md` and the unchanged per-invocation setup path in
`real_dnagpt_fides_scheme_b.cpp`).
