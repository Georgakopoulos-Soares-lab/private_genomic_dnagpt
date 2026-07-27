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
`e1b4f6c88b5113f8012f195b2b1d815b3610600eeebce10e103d0ea87157647b`).

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

### Scheme B context/key caching: scoping + in-process prototype (2026-07-27)

The 12-block, T=2 extrapolation above assumes context/key setup
(`context_keygen_load`, `~6.9s`) is a one-time cost amortized across 12 block
evaluations. Nothing in the repo had ever actually reused a context/key
lineage across more than one evaluation before this session -- that
assumption was `[A]`, not `[V]`. Scoped and prototyped this session; **no GPU
evidence yet** (that is the pending next step, see below).

**Scoping, sourced directly from the pinned FIDESlib commit
(`786c7600fb2f16b724e0acf73df367b27b8afed6`, `api/CryptoContext.cpp`/`.hpp`,
`api/Serialize.hpp`/`.cpp`, cloned and read directly) plus vanilla OpenFHE
example code:**

1. `[V]` In-process context/key reuse across many operations is the standard
   OpenFHE pattern (e.g. `openfhe-development/src/pke/examples/advanced-real-numbers.cpp`
   calls `KeyGen`/`EvalMultKeyGen`/`EvalRotateKeyGen` once, then runs many
   `Encrypt`/`Eval*` calls against the same context). Nothing FIDESlib-specific
   forbids this for the linear-algebra ops our evaluator uses.
2. `[V]` Hard FIDESlib constraint: `EvalMultKeyGen`/`EvalRotateKeyGen` both
   `OPENFHE_THROW` if called after the context is loaded
   (`api/CryptoContext.cpp:307-330`, `"...must be called before LoadContext"`).
   `LoadContext` itself is idempotent -- it returns immediately, a no-op, if
   `this->loaded` is already true (`api/CryptoContext.cpp:199-202`).
   Consequence: every rotation key any evaluation in a run will ever need must
   be generated in one batch before the single `LoadContext` call -- no
   incremental/lazy key generation afterward.
3. `[V]` One rotation-key set already covers every gate: reading
   `required_rotation_keys()` (`real_dnagpt_fides_scheme_b.cpp:408-427`),
   `full`'s key set is a strict superset of `attention`'s, which is a strict
   superset of `ln1`'s (BSGS + accumulate keys, plus 3 extra `full`-only
   rotations). Generating `full`'s set once therefore suffices for any gate
   mix -- no separate union computation needed.
4. `[V]` The GPU-side context persists in-process automatically once loaded:
   `LoadContext` populates `this->gpu` (a `FIDESlib::CKKS::Context`) once;
   every later `Eval*` call reads that same object, freed only in
   `~CryptoContextImpl` (`DeregisterCryptoContextGPU`,
   `api/CryptoContext.cpp:126-135`). No per-evaluation GPU reconstruction is
   needed within one process once loaded.
5. `[U]` Cross-process caching is architecturally possible but not
   implemented or measured here. FIDESlib's own `Serialize.hpp`/`.cpp` expose
   `SerializeToFile`/`DeserializeFromFile` for `CryptoContext`/`PublicKey`/
   `PrivateKey` (cereal, delegating to vanilla OpenFHE) plus static
   `SerializeEvalMultKey`/`SerializeEvalAutomorphismKey` for the actual
   keyswitching key material -- meaning the CPU-bound `KeyGen`/
   `EvalRotateKeyGen` cost (the same cost that stalled 8.5+ minutes under host
   contention in the 2026-07-25 session, see above) could in principle be
   computed once and reloaded by a later process. But `LoadContext`'s
   GPU-side context is never serialized; every new process still needs one
   `GenCryptoContextGPU` + `LoadContext` call from the (possibly
   deserialized) keys. This remains blocked/not yet implemented -- flagged
   here as the concrete next step, not claimed as done.
6. A repeated-gate harness that runs the same block-0 `full` gate N times
   against the same fixture, sharing one context/key setup, proves the cache
   *lifecycle* (setup-once, evaluate-many, no state leakage) without claiming
   two repetitions equal two distinct DNAGPT blocks -- must be labeled as
   such everywhere it appears.

**Prototype implemented:** new sibling file
`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_cached.cpp` (the
frozen, hash-pinned single-shot `real_dnagpt_fides_scheme_b.cpp` is
untouched -- confirmed unchanged, `shasum -a 256` still
`d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df`). The new
binary: loads the fixture once; requests `required_rotation_keys("full")`
once (finding 3); runs one `GenCryptoContext`/`KeyGen`/`EvalMultKeyGen`/
`EvalRotateKeyGen`/`LoadContext`/`Synchronize` setup, timed as
`context_keygen_load_seconds`; then loops `--repeats N` (fail-closed below
2) evaluations of the complete `full` gate against the same released-weight
fixture, constructing a **new** `Client`/`EncryptedEvaluator` each iteration
so every iteration's round-trip/logical-boundary counters start at zero by
construction (the checkable "no state leakage" property), each measured
independently against the unchanged `4e-2` oracle gate. The evidence schema
separates a one-time `setup` block from a per-iteration `iterations` array,
plus an explicit `label` field stating this is an in-process cache-lifecycle
proof, not multi-block or cross-process evidence. Same safety invariants as
the original: `O_EXCL` immutable-output guard, pinned FIDESlib-commit check,
pinned fixture-manifest hash check.

Build/run plumbing added, additive only (no existing target changed):
`CMakeLists.txt` gained a second `add_executable`;
`build_in_fideslib.sh` now builds both targets; new
`run_scheme_b_cached.sh`/`launch_brev_scheme_b_cached.sh` mirror the existing
scripts' guard structure, with output tags required to contain both
`_scheme_b_` and `_cached_`.

```bash
python3 fhe/gpu_real_scheme_b/test_caching_contract.py -v
python3 fhe/gpu_real_scheme_b/test_batching_contract.py -v
shasum -a 256 fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b.cpp
```

`[V]` All 9 new static/text contract tests pass (one-time setup calls;
rotation keys requested once, before the repeat loop; fresh `Client`/
`EncryptedEvaluator` per iteration; `--repeats < 2` rejected; evidence schema
separates `setup` from `iterations`; `EncryptedEvaluator` still contains
neither `Decrypt(` nor `secretKey`). The existing 3-test
`test_batching_contract.py` still passes unchanged, and the frozen source
hash is confirmed unchanged -- both run today on this Mac, no FIDESlib/CUDA
build required (same style as the existing contract test: source-text and
float64-oracle checks, not a compiled binary).

### Scheme B context/key caching: GPU evidence (2026-07-27)

Built on Brev (`FIDESLIB_ARCH=80-real BUILD_JOBS=4
fhe/gpu_real_scheme_b/build_in_fideslib.sh`, both targets), then launched via
the capacity-aware orchestrator (`fhe/gpu_real_scheme_b/wait_and_run_scheme_b_cached.sh`,
deployed to a versioned remote dir `gpu_real_scheme_b_cached_v1/`, driven by a
one-shot host `cron` entry so the launch survives local session/SSH loss --
confirmed via `ps` that the process tree is parented by `cron`, not by any
shell tied to the launching session). One real operational bug found and
fixed en route: the chosen run tag `..._cached2x_...` did not contain the
literal substring `_cached_` (underscore on both sides) required by
`launch_brev_scheme_b_cached.sh`'s own naming guard, so two genuine capacity
windows (an idle GPU appearing) were correctly refused before the tag was
fixed to `..._cached_2x_...` and the job relaunched -- the guard did exactly
its job (fail closed on a bad tag) rather than writing bad evidence.

Evidence: `fhe_fides_real_d768_t2_full_cached_2x_scheme_b_a100_20260727.json`
-- `[V]` PASS, both iterations independently pass the unchanged `4e-2` oracle
gate (`global_rel_inf` `3.73e-10` and `1.98e-10`; `worst_token_rel_inf`
`4.51e-10` and `2.40e-10`), zero intermediate decrypts, evaluator holds no
private key, 7 round trips / 24 logical boundary instances each (identical
protocol shape to the existing batched full-block gate). Source hash
`de79d6e7145b3ac9035b09cf7bc86354c755faab139c46f942bcc03f6ab72cc6` confirmed
matching the local file bit-for-bit.

`[V]` **Setup reuse is now proven on real GPU hardware, not just by
source-level construction.** `context_keygen_load_seconds` (one
`GenCryptoContext`/`KeyGen`/`EvalMultKeyGen`/`EvalRotateKeyGen`/`LoadContext`)
measured **`5.86s`**, paid exactly once, shared by both evaluations
(`467.24s` and `342.70s` respectively). This closes finding 5's in-process
half: the same context/key lineage evaluated the identical block-0 `full`
gate twice with independently-passing correctness and zero state leakage
between iterations (each iteration's own `round_trips`/`logical_boundary_instances`
in the evidence JSON starts fresh at 7/24, not 14/48 -- confirming the
per-iteration `Client`/`EncryptedEvaluator` construction works as designed).

**Improvement captured, honestly bounded:**

| Quantity | Value |
|---|---:|
| one-time setup (measured) | `5.86 s` |
| iteration 0 total (encrypt+eval+decrypt) | `467.74 s` |
| iteration 1 total (encrypt+eval+decrypt) | `343.12 s` |
| measured grand total, `repeats=2` | `816.80 s` |
| setup share of grand total | `0.72%` |
| `[A]` setup saved vs. 2 independent single-shot processes | `5.86 s` (one avoided re-setup) |
| `[A]` setup saved, naive ×12-block extrapolation (11 avoided re-setups) | `~64.5 s` out of the existing `4,474.1 s` 12-block grand total in this file above -- `~1.4%` |

`[U]` The `27%` gap between iteration 0 (`467.24s`) and iteration 1
(`342.70s`) is host-contention noise, not a caching effect: both iterations
run the identical setup-shared work, and the launch itself only cleared the
capacity gate at `load1=176.92` (still non-trivial load on this 255-core
shared host) after multiple hours oscillating between `load1` `176` and
`882` while waiting for an idle GPU (see
`scheme_b_cached_orchestrator.log`). No clean/uncontended re-run of this
gate has been attempted; the `5.86s` setup number is the more
contention-resistant of the two claims here, since it is small and CPU-bound
(the same kind of cost that previously stalled 8.5+ minutes under *extreme*
contention) yet completed normally at this run's moderate contention level.

`[U]` Cross-process (serialized context/keys reloaded by a separate process
invocation) reuse remains unimplemented and unmeasured -- finding 5 stands as
scoped, not closed. Setup reuse is `[V]` proven only for multiple evaluations
sharing one live process.

### Scheme B context/key caching: cross-process serialization prototype (2026-07-27)

Closes the remaining half of finding 5 above: whether a *second, independent
process* can skip the CPU-bound `KeyGen`/`EvalMultKeyGen`/`EvalRotateKeyGen`
entirely by deserializing a context/key lineage a prior process wrote to
disk, versus what every process must still pay regardless.

**Re-verified independently against the pinned FIDESlib commit
(`786c7600fb2f16b724e0acf73df367b27b8afed6`, cloned fresh and read directly:
`api/Serialize.hpp`/`.cpp`, `api/CryptoContext.hpp`/`.cpp`,
`examples/serial/src/serial.cpp`, `examples/resnet/src/controller.cpp`) --
the prior session's finding 5 summary is confirmed accurate, with more
precision on the mechanism:**

1. `[V]` `fideslib::Serial::SerializeToFile`/`DeserializeFromFile` for
   `CryptoContext` (`api/Serialize.cpp:14-72,102-208`) delegate to vanilla
   OpenFHE cereal serialization of the CPU-side `lbcrypto::CryptoContext`,
   plus a FIDESlib-specific `.dev` sidecar (plain text) recording
   `devices`/`auto_load_ciphertexts`/`auto_load_plaintexts`/
   `rotation_indexes`/`keyDist`/`slots_bootstrap`. `PublicKey`/`PrivateKey`
   `SerializeToFile`/`DeserializeFromFile` are pure vanilla-OpenFHE
   delegation, no FIDESlib-specific data.
2. `[V]` `EvalMultKeyGen`/`EvalRotateKeyGen` only `OPENFHE_THROW` once
   `this->loaded` is true (`api/CryptoContext.cpp:307-330`, exact message
   `"...must be called before LoadContext"`) -- so a process that never
   calls `LoadContext` is free to call them at any time, including *after*
   it would otherwise be "too late" relative to some other process's state.
3. `[V]` `SerializeEvalMultKey`/`SerializeEvalAutomorphismKey`/
   `DeserializeEvalMultKey`/`DeserializeEvalAutomorphismKey`
   (`api/CryptoContext.cpp:390-484`) are thin dispatchers onto vanilla
   OpenFHE's own **process-global static key maps**
   (`s_evalMultKeyMap`/`s_evalAutomorphismKeyMap`, explicitly instantiated at
   `api/CryptoContext.cpp:39-41`). `EvalMultKeyGen`/`EvalRotateKeyGen` and
   `DeserializeEvalMultKey`/`DeserializeEvalAutomorphismKey` populate the
   *same* maps -- `LoadContext`'s later key lookups cannot distinguish
   "generated live" from "deserialized." Both `examples/serial/src/serial.cpp`
   and `examples/resnet/src/controller.cpp`'s `deserialize_context()` prove
   this end-to-end: they call `DeserializeEvalMultKey`/
   `DeserializeEvalAutomorphismKey` then go straight to `LoadContext`, with
   **no** `EvalMultKeyGen`/`EvalRotateKeyGen` call anywhere in that path.
4. `[V]` One caveat that matters: `LoadContext`'s rotation-key upload loop
   iterates `this->rotation_indexes` (`api/CryptoContext.cpp:236`), which is
   FIDESlib's own bookkeeping vector populated only by `EvalRotateKeyGen`
   (`api/CryptoContext.cpp:329`) or by deserializing the **CryptoContext
   itself** (the `.dev` sidecar's `RotationIndexes:` line,
   `api/Serialize.cpp:170-182`) -- `DeserializeEvalAutomorphismKey` alone
   does not touch it (it is `const`, header line 95). A cross-process reader
   therefore needs both: deserialize the `CryptoContext` (restores
   `rotation_indexes`) *and* `DeserializeEvalAutomorphismKey` (restores the
   actual key material in OpenFHE's static map).
5. `[V]` **Answering finding 5's open question directly: yes, a
   deserializing second process can skip `EvalMultKeyGen`/
   `EvalRotateKeyGen` entirely** and go straight from deserialization to
   `LoadContext`.
6. `[V]` The GPU-side `FIDESlib::CKKS::Context` (`this->gpu`, populated by
   `LoadContext` at `api/CryptoContext.cpp:247`) has **no serialize path at
   all** (grepped `src/CKKS/Context.cuh`/`.cu` -- no such symbols exist).
   `LoadContext` unconditionally calls `GenCryptoContextGPU`
   (`api/CryptoContext.cpp:223`) whenever it actually runs
   (`!this->loaded && !this->devices.empty()`), rebuilding GPU-side NTT
   tables/memory from the CPU-side crypto parameters and re-uploading
   key-switching keys from the (possibly just-deserialized) key maps. Every
   process that wants to evaluate on GPU must pay this, regardless of what
   was deserialized -- it is not avoidable by any serialization scheme
   FIDESlib currently exposes.

**Prototype implemented:** two new sibling binaries under
`fhe/gpu_real_scheme_b/src/`, both forked from the caching prototype
(`real_dnagpt_fides_scheme_b_cached.cpp`, confirmed unchanged, `shasum -a
256` still `de79d6e7145b3ac9035b09cf7bc86354c755faab139c46f942bcc03f6ab72cc6`)
without editing it:

- `real_dnagpt_fides_scheme_b_serialize_writer.cpp` ("process A"): builds a
  fresh context/keys exactly like every other gate (`GenCryptoContext` ->
  `Enable`s -> `KeyGen` -> `EvalMultKeyGen` -> `EvalRotateKeyGen` for the
  `full` gate's rotation set, per the same "request the covering gate once"
  rationale as the caching prototype) but **deliberately never calls
  `LoadContext`** -- point 1 above shows everything the `.dev` sidecar needs
  is already populated by the time `EvalRotateKeyGen` returns, so
  `LoadContext` is not a prerequisite for serialization. It then serializes
  `CryptoContext`+`PublicKey`+`PrivateKey`+`EvalMultKey`+
  `EvalAutomorphismKey` to `--state-dir` and exits without evaluating
  anything. Defines no `Client`/`EncryptedEvaluator` at all.
- `real_dnagpt_fides_scheme_b_serialize_reader.cpp` ("process B"):
  deserializes everything from `--state-dir` (contains **zero** textual
  occurrences of `GenCryptoContext(`, `->KeyGen()`, `->EvalMultKeyGen(`, or
  `->EvalRotateKeyGen(` -- verified by `test_serialization_contract.py`, not
  merely asserted in the source), then calls the one unavoidable GPU step
  (`LoadContext`, internally `GenCryptoContextGPU`, per point 6), timed
  separately from deserialization, then evaluates the same released-weight
  block-0 `full` gate against the unchanged `4e-2` oracle exactly like every
  other gate. The evidence JSON records `writer_keygen_seconds`/
  `writer_serialize_to_disk_seconds` (folded in from the writer's own
  non-secret `writer_timing.txt`), `reader_deserialize_seconds`,
  `reader_load_context_gpu_seconds`, and
  `reader_encrypted_evaluation_seconds` separately, plus an explicit
  `avoidable_via_cross_process_reload_seconds` (= writer's keygen cost, now
  proven skippable) vs. `unavoidable_per_process_gpu_load_seconds` (=
  `LoadContext`/`GenCryptoContextGPU`, paid by every process, proven
  unavoidable) summary.

**Secret-key hygiene (`~/.agents/policies/secrets.md`):** this repo's real
protocol keeps the secret key inside `Client` and never persists it; this
prototype breaks that rule deliberately, only to simulate a restarted client
process for measurement -- an explicit testbed simplification, not a
production client/server key-custody design (out of scope per `CLAUDE.md`).
The writer serializes the secret key to `--state-dir/secret-key.txt` with
`chmod 0600` immediately after writing it; neither binary ever prints that
filename or path to stdout/stderr, and the writer's own evidence JSON
explicitly omits it from its `serialized_artifacts` list (a real bug was
caught and fixed here during development: the first draft *did* list
`"secret-key.txt"` in that array, directly contradicting its own
`secret_key_handling` field -- `test_serialization_contract.py`'s
`test_evidence_json_never_records_secret_key_path` catches this class of
mistake for both binaries). The orchestrator
(`wait_and_run_scheme_b_serialize.sh`) deletes the entire state directory
after the reader phase completes, regardless of pass/fail, logging only that
cleanup happened, never the filename.

Same safety invariants as every other gate: `O_WRONLY | O_CREAT | O_EXCL`
immutable-output guards, pinned FIDESlib-commit check, pinned
fixture-manifest hash check (reader only -- the writer never touches the
fixture). New evidence tags must contain both `_scheme_b_` and
`_serialized_` (a substring distinct from the caching prototype's
`_cached_`, so the three prototypes' evidence namespaces can never
collide -- checked textually by the contract test, not just asserted).

```bash
python3 fhe/gpu_real_scheme_b/test_serialization_contract.py -v
python3 fhe/gpu_real_scheme_b/test_caching_contract.py -v
python3 fhe/gpu_real_scheme_b/test_batching_contract.py -v
shasum -a 256 fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b.cpp
shasum -a 256 fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_cached.cpp
```

`[V]` All 29 new static/text contract tests pass (writer builds fresh
context/keys exactly once and never calls `LoadContext`; writer defines no
`Client`/`EncryptedEvaluator`; writer restricts and never logs the
secret-key path; reader has zero occurrences of the key-generation calls;
reader deserializes before the one `LoadContext` call; reader's
`EncryptedEvaluator` still contains neither `Decrypt(` nor `secretKey`;
neither binary's evidence JSON ever records the secret-key filename; CMake
and both new run scripts reference the new binaries and require the
`_serialized_` tag marker). The existing 9-test caching contract and 3-test
batching contract suites still pass unchanged, and both frozen source hashes
are confirmed unchanged -- all three suites run today on this Mac, no
FIDESlib/CUDA build required (same style as the existing contract tests:
source-text and float64-oracle checks, not a compiled binary).

### Scheme B context/key caching: cross-process GPU evidence (2026-07-27)

Built on Brev (`FIDESLIB_ARCH=80-real BUILD_JOBS=4
fhe/gpu_real_scheme_b/build_in_fideslib.sh`, all four targets, from a fresh
`build/` dir since the directory was root-owned from an earlier container
run). One real build-time bug found and fixed en route: the reader's
`EncryptedEvaluator` inlined `cc_->EvalMult(ct, some_plain_fn(...))` calls
directly -- FIDESlib's `EvalMult(Ciphertext, Plaintext)` overload takes the
plaintext by non-`const` lvalue reference, so a temporary `Plaintext`
returned straight from a helper cannot bind to it (9 compile errors,
`nvcc`). The original/cached gates avoid this via a `multiply_plain(Ct,
Plaintext plaintext)` wrapper that takes the plaintext by value into a named
parameter (an lvalue); the reader lost that wrapper when its
`EncryptedEvaluator` was trimmed down during authoring. Fixed by
reintroducing the identical wrapper and routing all eight affected call
sites through it -- confirmed against real `nvcc` on the pinned container,
not just locally. Then launched via the capacity-aware two-phase
orchestrator (`fhe/gpu_real_scheme_b/wait_and_run_scheme_b_serialize.sh`,
deployed to a versioned remote dir `gpu_real_scheme_b_serialized_v1/`,
driven by a one-shot host `cron` entry so the launch survives
local session/SSH loss -- confirmed via `ps` that the process tree is
parented by a cron-spawned `/bin/sh -c`, not any shell tied to the launching
session). Host load dropped from `838.66` to `175.85` over about ten minutes
of polling before the writer phase launched; the reader phase followed
immediately after (load `153.17`) with a confirmed idle-GPU preflight
(`mem=0MiB util=0%`) for both phases.

Evidence: `fhe_fides_real_d768_t2_full_serialized_writer_scheme_b_a100_20260727.json`
(writer) and `fhe_fides_real_d768_t2_full_serialized_reader_scheme_b_a100_20260727.json`
(reader) -- `[V]` both PASS. `source_sha256` in each evidence file matches
the local repo's `shasum -a 256` of the corresponding `.cpp` bit-for-bit
(`ea720f1b...` writer, `4dbb003f...` reader), confirming the binaries that
ran on Brev are the same code reviewed here.

`[V]` **Finding 5 is now closed on real GPU hardware, not just by source
reading.** The reader's own source contains zero occurrences of
`GenCryptoContext(`, `->EvalMultKeyGen(`, or `->EvalRotateKeyGen(` (checked
by `test_serialization_contract.py`), and it reached the identical passing
block-0 `full` gate result (`global_rel_inf`/`worst_token_rel_inf`
`3.4569e-10`, 7 round trips / 24 logical boundary instances -- the same
protocol shape as every other full-block gate) as a genuinely separate OS
process (its own `docker run`, sharing only a bind-mounted state directory
with the writer) that never generated a single key itself.

**Honest cost accounting -- this is a negative/boundary result, not tuned to
look favorable:**

| Quantity | Value |
|---|---:|
| writer: `GenCryptoContext`+`KeyGen`+`EvalMultKeyGen`+`EvalRotateKeyGen` | `1.274 s` |
| writer: serialize to disk (context+all keys) | `0.970 s` |
| reader: deserialize (context+public+secret key+eval keys) | `3.822 s` |
| reader: `LoadContext`/`GenCryptoContextGPU` (unavoidable every process) | `4.053 s` |
| reader: encrypted evaluation (`384.01s` server + `1.98s` client boundary) | `385.985 s` |
| **reader's own total setup** (deserialize + `LoadContext`) | **`7.875 s`** |
| for comparison: in-process cached run's one-time setup (same gate) | `5.862 s` |

`[V]` The reader's own setup cost (`7.875s`) is **larger**, not smaller,
than the in-process cached prototype's one-time setup (`5.862s`,
`fhe_fides_real_d768_t2_full_cached_2x_scheme_b_a100_20260727`) -- and even
larger than a hypothetical fresh setup using this run's own numbers
(writer's `1.274s` keygen + this reader's `4.053s` `LoadContext` =
`5.327s`). The reason is not the unavoidable GPU step: `LoadContext`
(`4.053s`) is in the same ballpark as the combined in-process number
(`5.862s`, which itself included `KeyGen`/`EvalMultKeyGen`/`EvalRotateKeyGen`).
The reason is that **deserialization itself costs more (`3.822s`) than the
`KeyGen`+`EvalMultKeyGen`+`EvalRotateKeyGen` it replaces (`1.274s`)** at this
context size (`ring_dim=65536`, 66 rotation keys) -- cereal parsing plus
repopulating OpenFHE's process-global static key maps is not free. **No
cross-process speedup is claimed from this evidence.**

`[A]` This does not mean cross-process serialization has no value --
`EvalRotateKeyGen` is the specific call that stalled 8.5+ minutes under
*extreme* host contention in the 2026-07-25 session (load average
~1000-1127), while deserialization is a fixed, contention-insensitive cost
independent of host CPU load. So the plausible value proposition is
**avoiding repeated exposure to CPU-contention risk across many readers**,
not raw wall-clock savings for a single reader -- and that specific claim
has not been tested here (this run's host load had already dropped to
~150-175 by launch time, nowhere near the ~1000+ regime that caused the
original stall). `[U]` Only one writer and one reader were exercised; a
many-reader-per-writer scenario, and a reader launched under genuinely
extreme host contention, both remain unmeasured.

```bash
brev exec awesome-gpu-name -- "cat .../evidence/fhe_fides_real_d768_t2_full_serialized_writer_scheme_b_a100_20260727.json"
brev exec awesome-gpu-name -- "cat .../evidence/fhe_fides_real_d768_t2_full_serialized_reader_scheme_b_a100_20260727.json"
```
