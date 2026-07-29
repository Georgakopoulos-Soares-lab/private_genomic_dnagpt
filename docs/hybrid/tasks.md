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

### Scheme B GPU-side profiling: where the 384s server time actually goes (2026-07-27)

The 2026-07-27 caching/serialization sessions above closed setup/key reuse as a speed
lever (`< 2%` of wall time even in the worst measured case). The remaining cost is the
server-side GPU linear algebra inside one block-0 `full` gate. This session profiles
that cost directly with real GPU-synchronized wall-clock timing rather than a static
call-count model.

New sibling binary `fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_profiled.cpp`
(forked from the frozen single-shot gate, hash confirmed unchanged) brackets every one
of the 24 `matmul()` call sites with `Synchronize()`-before/after timers by graph
stage (`matmul_qkv_query`/`_key`/`_value`/`_attention_projection`/`_mlp_fc`/
`_mlp_projection`), additionally instruments one representative call (token 0's Q
projection) at full giant-step granularity, and times every other linear-algebra
stage (LayerNorm, attention score/delta, GELU pack/restore) coarsely. It changes
nothing about the encrypted computation: same packing, same `BSGS_N1=32`/`BSGS_N2=32`
split, same `TOL=4e-2` oracle gate. `test_profiling_contract.py` (16 tests) statically
verifies the 24 call sites, the one detailed call, the bucket labels, and that the
frozen sources are untouched -- no FIDESlib/CUDA build required.

```bash
python3 fhe/gpu_real_scheme_b/test_profiling_contract.py -v
FIDESLIB_ARCH=80-real BUILD_JOBS=4 fhe/gpu_real_scheme_b/build_in_fideslib.sh

FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_scheme_b/run_scheme_b_profiled.sh 0 "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_full_profiled_scheme_b_a100_20260727.json
```

Evidence: `fhe_fides_real_d768_t2_full_profiled_scheme_b_a100_20260727.json` --
`[V]` PASS, `global_rel_inf=2.51e-10`, well inside the unchanged `4e-2` gate.
`encrypted_evaluation=533.13s` (`server_linear_algebra_seconds=530.70s` +
`client_boundary_seconds_total=2.43s`), source hash
`4ca9c66470c412140ed483479ae869963e8b097e2880b3ffd4d9287a81374d61`.

**`[V]` Finding 1: the 6 matmul stages are 99.5% of server time.** Summing the six
`matmul_*` bucket totals (78.87+74.87+96.97+46.83+124.44+106.26 = 528.25s) against
`server_linear_algebra_seconds` (530.70s) gives `99.54%`. Everything else --
LayerNorm (4 calls), the 12-head attention score/delta loop, GELU pack/restore,
residual adds, baby-rotation batches -- is collectively under 0.5%.

**`[V]` Finding 2: within one matmul call, rotation is not the bottleneck.** The one
detailed call's three sub-buckets: `detail_ciphertext_plaintext_multiply_accumulate`
(32 giant-steps, `44.15s` total, `1.38s` mean), `detail_giant_rotation_keyswitch` (31
calls, `0.036s` total -- `<0.1%` of the call), `detail_giant_result_accumulate` (31
calls, `0.0024s` total). Ciphertext-plaintext multiply-and-accumulate is essentially
100% of a matmul call's internal cost; giant-step `EvalRotate`/keyswitch and
baby-step hoisted `EvalFastRotation` are both negligible. This is a stronger, more
specific result than the already-ruled-out "retune `BSGS_N1`/`BSGS_N2`" lever
(CLAUDE.md) -- it shows the entire rotation operation class, not just its N1/N2
balance, is immaterial here. (Note: the JSON's own `unaccounted_seconds` field reads
`-45.58s` because the three `detail_*` buckets nest inside the coarse
`matmul_qkv_query` bucket's already-counted time -- summing all buckets naively
double-counts that one call. Removing the double-count gives a corrected
reconciliation of `532.09s` against `530.70s` measured, `99.7%` accounted for.)

**`[U]` Finding 3: per-call cost is not uniform -- a 3.15x front-to-back gap with no
settled explanation.** The 6 QKV-stage calls average `~41.8s/call`; the 8
`matmul_mlp_projection` calls average `~13.3s/call`. Both stages operate on
similarly "fresh" ciphertexts (2 levels consumed since their respective
client-boundary resets -- checked directly, so a depth/level-cost explanation does
not hold). This finding drove two follow-up investigations below, one abandoned as
unsafe and one left inconclusive by host conditions.

```bash
python3 -c "import json; d=json.load(open('results/runs/fhe_fides_real_d768_t2_full_profiled_scheme_b_a100_20260727.json')); b=d['profiling_breakdown']['buckets']; print({k:v['mean_seconds'] for k,v in b.items() if k.startswith('matmul_')})"
```

### Scheme B diagonal-plaintext-cache fix: designed, implemented, abandoned as unsafe (2026-07-27/28)

Motivated by Finding 3 above: each of the block's 24 `matmul()` calls corresponds to
one of only **12 distinct D×D weight matrices**, each used by both of the T=2 tokens.
Reading `matmul()`/`bsgs_inner_loop`, every call independently rebuilds and
re-encodes its own 1024 BSGS diagonal `Plaintext` objects from scratch via
`raw_plain()`/`MakeCKKSPackedPlaintext`, even on the token-1 call reusing the
identical weight (and therefore identical diagonal values) as the token-0 call for
that site. Pinned FIDESlib source (`api/CryptoContext.cpp:251-258`,
`LoadPlaintext`) shows `LoadPlaintext(Plaintext& pt)` returns immediately if
`pt->loaded` is already true -- so reusing the *same* `Plaintext` object (not merely
an equal one) across a weight's two token-calls should, in principle, let the second
call skip CPU-encode+GPU-upload entirely.

**Implementation:** new sibling `fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_diagcache.cpp`
(forked from the profiled binary, keeping identical instrumentation for direct
comparison), source hash `ea2e9eca3f40233eed6105c5d13e1e39c9a3e440d193d2f9ec5579970cc14593`.
Adds `std::map<const Matrix*, std::vector<Plaintext>> diagonal_cache_` to
`EncryptedEvaluator`, keyed by weight-matrix address (stable for the whole
`evaluate()` call); `cached_diagonal_plaintexts()` builds a weight's 1024 diagonals
once and memoizes them, `bsgs_inner_loop` looks them up instead of rebuilding.
Changes nothing about the encrypted computation (same packing/BSGS/TOL) and reuses
Plaintext object *identity*, never their content. `test_diagcache_contract.py` (16
tests) statically verifies the cache is keyed by address, the inner loop no longer
builds plaintexts inline, and both frozen sources remain untouched.

**`[V]` GPU evidence: 6 consecutive real-GPU crashes across 6 different physical
GPUs (0, 4, 7, 4, 1, 4), all with the identical signature**, immediately after
context setup succeeds (the `[context] ...` line prints, then nothing else, then
`Cuda failure /opt/FIDESlib/src/CudaUtils.cu:400: 'out of memory'`). No evidence JSON
was ever produced by any diagcache attempt (correctness of the fix on real GPU
hardware remains unverified). Attempts (no `results/runs/` rows added -- none
produced valid evidence):

| Attempt | Physical GPU | Wall time to crash | Preflight reading |
|---|---:|---:|---|
| 1 (no suffix) | 0 | ~3 min | mem=541MiB util=0% (all 8 GPUs spiked to 53-84% util moments later) |
| retry1 | 4 | ~3 min | mem=0MiB util=0% compute_processes=0 |
| retry2 (+ live memory sampler) | 7 | ~3 min | mem=1608MiB util=0% |
| retry3 | 4 | ~13 min | mem=3272MiB util=0% **compute_processes=4** |
| retry4 | 1 | ~3 min | mem=0MiB util=0% |
| retry5 | 4 | ~24 min | mem=2878MiB util=5% **compute_processes=4** |

`[V]` **`addr2line` against the actual crashed binary (independently reproduced, not
just asserted) resolves the crash to:** `Ciphertext::multPt` ->
`Plaintext::adjustPlaintextToCiphertext` -> `Plaintext::copy` -> `RNSPoly::copy` ->
`RNSPoly::grow` -> `LimbPartition::generateLimbToLevel`/`generate` -> `Limb::Limb` ->
`VectorGPU::VectorGPU` -> `FIDESlib::GPUmalloc`. Identical (or near-identical, one
offset byte apart across different process loads) across all 6 crashes.

**`[V]` Two same-session control runs of the byte-for-byte unmodified profiled
baseline both passed cleanly**, bracketing the 6 diagcache failures:
`fhe_fides_real_d768_t2_full_profiled_scheme_b_a100_20260727.json` (before) and
`fhe_fides_real_d768_t2_full_profiled_scheme_b_a100_20260728_control.json` (after,
`global_rel_inf=2.87e-10`, `encrypted_evaluation=515.09s`, launched specifically as a
control experiment once the first several diagcache attempts had failed). A 6-fail /
2-pass split under overlapping host-load conditions the same day is not explainable
as pure external contention.

**`[U]` Root cause narrowed but not fully resolved.** A prior hypothesis (this
repo's own, and independently proposed by an external code review shown mid-session)
that the failure is "unbounded `diagonal_cache_` map accumulation, peak memory only
grows" does **not** hold up quantitatively:
- FIDESlib's `device_plaintexts`/`device_ciphertexts` registries
  (`api/CryptoContext.cpp`) have **zero callers** of their paired
  `EvictDevicePlaintext`/`EvictDeviceCiphertext` functions anywhere in the FIDESlib
  codebase (confirmed by `git grep` across the whole pinned tree) -- both the
  original baseline and the diagcache fix accumulate GPU-registered plaintexts
  without eviction for the life of the process.
- The diagcache fix registers **fewer** total distinct plaintexts than the original
  (`12*1024=12288` vs the original's `24*1024=24576`, since it skips the second
  token's redundant registration rather than adding to it) -- the opposite of what
  an "unbounded accumulation" story predicts.
- The object that actually grows at the crash site (`Plaintext b_(cc_);` inside
  `Ciphertext::multPt`, `src/CKKS/Ciphertext.cpp:388`) is a **fresh, local, per-call
  stack variable in both the original and the diagcache fix** -- it is never part of
  the cache, and this exact `GPUmalloc` call site fires on every one of the 24666
  ciphertext-plaintext multiplications in *either* design.

What the evidence does point to: something about **reusing a GPU-resident
`Plaintext` object across two separate `multPt` calls** (the fix's core mechanism)
interacts badly with this pinned FIDESlib build's `adjustPlaintextToCiphertext`/
`RNSPoly::grow` path in a way a never-before-multiplied object does not trigger. The
exact low-level reason (why growth on a second use needs more memory than the first)
was not resolved within this session's time budget -- it would require reading
`RNSPoly::grow`/`LimbPartition::generate`'s CUDA-side limb/partition allocation logic
in more depth than was productive to pursue via further blind GPU retries.

**Conclusion: this specific caching approach is not safe against the pinned
FIDESlib build (`786c7600fb2f16b724e0acf73df367b27b8afed6`) and is abandoned, not
fixed.** A proposed workaround ("cap the cache to one hot weight, evict on the next
distinct weight touched") would not resolve this -- it would still reuse each
weight's plaintexts across its own two token-calls, which is exactly the reuse
pattern that crashes; any design that captures *any* benefit from this optimization
target reuses a plaintext for a second `multPt` call. This is reported as a valid,
honest negative result per CLAUDE.md's explicit allowance for negative boundaries --
no `results/runs/` row is added since no attempt produced passing (or any) evidence.

```bash
python3 fhe/gpu_real_scheme_b/test_diagcache_contract.py -v   # local, no GPU build needed
# GPU attempts (all failed, no evidence produced -- see table above)
```

### Scheme B warm-up-prelude hypothesis: correctness reconfirmed, timing inconclusive (2026-07-28)

A different, much lower-risk hypothesis for Finding 3's front-to-back gap: a CUDA
allocator/kernel warm-up effect (first `cudaMalloc` calls of a new size class,
first-launch kernel JIT/module loading being slower than steady-state reuse) rather
than anything about object reuse. New sibling
`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_warmup.cpp` (forked from the
profiled baseline; `matmul()`/`bsgs_inner_loop`/`EncryptedEvaluator`/`Client` are
byte-identical to it, verified by diff and by `test_warmup_contract.py`, 14 tests)
adds one untimed dummy pass -- one ct-pt multiply, one ct-ct multiply, one `EvalAdd`,
one giant `EvalRotate`, one baby `EvalFastRotation` batch, all at the real problem's
exact shapes on a throwaway all-zero ciphertext/plaintext -- immediately after
context setup and before the real timed `evaluate()` call, timed separately as
`warmup_prelude` and never folded into `encrypted_evaluation_seconds`.

Evidence: `fhe_fides_real_d768_t2_full_warmup_scheme_b_a100_20260728.json` -- `[V]`
PASS, `global_rel_inf=2.93e-10`, correctness reconfirmed a third time this session.
Source hash `d76c71fbe058ce4f2b4afd8f977d02e719ef7f130c2265c7fee88e7d4fd58716`.
`warmup_prelude=0.226s` (negligible, as expected).

`[U]` **Timing result is inconclusive, not negative or positive.** This run's
`context_keygen_load` alone took the process ~43 minutes of wall-clock to reach
(under host load pinned at 800-1218 for that entire window, the worst sustained
contention observed this session), and `encrypted_evaluation=2595.60s` -- roughly
5x the clean baseline's `533.13s`. Comparing each stage's per-call mean against the
clean baseline shows the inflation is *not* uniform: QKV is only `2.49x` slower here
(`104.05s` vs `41.79s`/call) while `matmul_mlp_projection` is `8.83x` slower
(`117.28s` vs `13.28s`/call) -- growing worse through the run, the signature of host
contention *increasing* over the ~43-minute `evaluate()` window, not evidence the
warm-up prelude flattened anything. This run cannot confirm or refute the warm-up
hypothesis; the signal is swamped by time-varying external load. `[U]` No clean,
low-contention window was available on this host during the session to re-attempt
this specific comparison.

```bash
python3 fhe/gpu_real_scheme_b/test_warmup_contract.py -v
FIDES_CONTAINER_IMAGE=dnagpt-fideslib:786c-asymfix2 \
FIDES_RUN_ENVIRONMENT='Brev A100-SXM4-80GB [gpu]' \
fhe/gpu_real_scheme_b/run_scheme_b_warmup.sh 0 "$FIXTURE" \
  /work/results/runs/fhe_fides_real_d768_t2_full_warmup_scheme_b_a100_20260728.json
```

### Next optimization candidate identified: FIDESlib's native batched `LinearTransform` primitive (2026-07-28, not yet implemented)

`[A]` Scoped but not attempted this session. Grepping the pinned FIDESlib commit
(`786c7600fb2f16b724e0acf73df367b27b8afed6`) surfaced a purpose-built, already-tested
BSGS diagonal linear-transform primitive that our hand-rolled `matmul()`/
`bsgs_inner_loop` does not use:

- `FIDESlib::CKKS::LinearTransform(Ciphertext&, int rowSize, int bStep, const
  std::vector<Plaintext*>& pts, int stride, int offset)`
  (`src/CKKS/LinearTransform.cuh`/`.cu`) performs exactly the BSGS diagonal
  matrix-vector multiply our `matmul()` reimplements manually, but internally
  dispatches to `RNSPoly::LTdotProductPtBatch` ->
  `LimbPartition::LTdotProductPtBatch` -> the `dotProductLtBatchedPt___`/`Pt2___`/
  `Pt3___` CUDA kernels (`src/CKKS/ElemenwiseBatchKernels.cu`/`.cuh`), which process
  a **batch** of ciphertext-plaintext diagonal products in kernels launched over
  the whole batch, not our current design's 1024 sequential `EvalMult`+`EvalAdd`
  calls per matmul call.
- This is not experimental/unused code inside FIDESlib -- it is exercised by
  FIDESlib's own official transformer example,
  `examples/bert-tiny/src/MatMul.cu`, for the identical purpose (matrix
  multiplication inside a CKKS-encrypted transformer block), and by
  `CoeffsToSlots.cu`/`Bootstrap.cu` internally.
- Given Finding 2 above (ciphertext-plaintext multiply-and-accumulate is ~100% of a
  matmul call's cost, with rotation/keyswitch negligible), replacing the manual
  per-diagonal loop with this native batched primitive is the most promising
  remaining lever for real wall-clock speedup: it targets the actual bottleneck
  operation directly, is a stock, already-tested FIDESlib API (not a new
  correctness risk comparable to the abandoned diagcache reuse), and requires no
  change to packing, `BSGS_N1`/`BSGS_N2`, or `TOL`.
- Not yet scoped in detail against our exact `SLOTS=4096`/four-copies packing
  layout, nor measured. This is the concrete next step, not a claim of speedup.

```bash
git -C <pinned-FIDESlib-clone> grep -n "LinearTransform\|LTdotProductPtBatch" HEAD -- '*.cuh' '*.cu' '*.cpp'
```

### `LinearTransform` scoping: compatible, not a drop-in, three concrete adaptations required (2026-07-28)

Read `src/CKKS/LinearTransform.cu`/`.cuh`, `examples/bert-tiny/src/MatMul.cu`, the pinned
FIDESlib-install rules, and this repo's own `api/CryptoContext.cpp`/`.hpp` wrapper (the
"asymfix2" OpenFHE-API-compatible shim every Scheme B `.cpp` file actually calls) before
writing any implementation code, per this session's instructions. Verdict: **compatible
with our exact `SLOTS=4096`/`PACK_WIDTH=1024`/`BSGS_N1=32`/`BSGS_N2=32` packing, but three
adaptations are required** -- this is not a mechanical swap of one function call for
another.

**`[V]` Finding 1: our code never touches native FIDESlib types today; `LinearTransform`
does.** Every Scheme B `.cpp` file (`real_dnagpt_fides_scheme_b*.cpp`) only ever calls the
OpenFHE-API-compatible wrapper (`using Ct = Ciphertext<DCRTPoly>`; `cc_->EvalMult`/
`EvalRotate`/`EvalFastRotation`/`MakeCKKSPackedPlaintext`, all from the installed
`<fideslib.hpp>` shim, not `FIDESlib::CKKS::*` directly). Tracing `EvalMult(ct1, Plaintext&
pt)` in `api/CryptoContext.cpp` confirms our `multiply_plain()` bottoms out at
`res_gpu->multPt(*pt_gpu)` -- the native single-diagonal call the 2026-07-27 profiling
already measured. `FIDESlib::CKKS::LinearTransform` is a free function operating on native
`FIDESlib::CKKS::Ciphertext&`/`Plaintext*`, and is only `#include`d in
`api/CryptoContext.cpp` (for `ConvolutionTransform`/`SpecialConvolutionTransform`, which
*are* wired to public wrapper methods `ConvolutionTransformInPlace`/
`SpecialConvolutionTransformInPlace`) -- **no `LinearTransformInPlace` wrapper exists**.
Calling it requires our new `.cpp` to reach into native FIDESlib types itself.

**`[V]` Finding 2: this is possible without touching any frozen file or the wrapper.**
`CryptoContextImpl<DCRTPoly>::LoadCiphertext`/`LoadPlaintext`/`GetDeviceCiphertext`/
`GetDevicePlaintext` are `public` members (`api/CryptoContext.hpp:53-66,243-244`) --
exactly the calls `ConvolutionTransformInPlace`'s own implementation uses internally
(`api/CryptoContext.cpp:1850-1879`: `LoadCiphertext`/`LoadPlaintext`, `static_pointer_cast`
to `FIDESlib::CKKS::Ciphertext`/`Plaintext`, then call the native free function). Our new
sibling file can replicate that identical pattern locally. Header availability is also
confirmed, not assumed: the pinned FIDESlib's top-level `CMakeLists.txt` (`install(...)`
at line 344) installs `*.cuh`/`*.hpp`/`*.h`/`*.inc` from the *entire* `src/` tree alongside
the public API headers ("Also install private headers so downstream targets can include
implementation details when needed") -- `#include "CKKS/LinearTransform.cuh"` will resolve
for an external consumer linked against `fideslib::fideslib`, no FIDESlib-side build change
needed.

**`[V]` Finding 3: diagonal indexing and packing match exactly, no packing-contract change
needed.** `LinearTransform(ctxt, rowSize, bStep, pts, stride, offset)` computes
`gStep = ceil(rowSize/bStep)` and, in its `FUSED` branch (hardcoded on --
`constexpr bool FUSED = true`, the non-FUSED branch is dead code), builds
`Aptr[bStep*j + i] = pts[bStep*j + i]` for giant index `j` and baby index `i`. Our own
`bsgs_inner_loop`'s `diagonal = BSGS_N1*giant + small` is the identical indexing scheme.
With `rowSize=1024=PACK_WIDTH`, `bStep=BSGS_N1=32`, `gStep=BSGS_N2=32`, `stride=1`,
`offset=0`, a `pts` vector built as `pts[BSGS_N1*giant + small] = <existing diagonal
plaintext for (giant, small)>` maps directly onto `LinearTransform`'s expected layout.
`LinearTransform`'s internal baby-step rotation indices (`i*stride` for `i` in
`[0,bStep)`) also match our own `baby_rotations()` (rotations `0..BSGS_N1-1`, index 0 =
identity). **The existing `packed_values()` diagonal-value computation (including its
`rolled_row` pre-rotation, which compensates for one post-hoc giant-step rotation instead
of one per diagonal) can be reused unchanged** -- `LinearTransform` implements the same
"sum bStep diagonals, then rotate once per giant step" structure our manual code already
does by hand.

**`[U]`->`[V]` Finding 4 (the real blocker, now resolved): plaintexts must be pre-encoded
at the ciphertext's exact current level; ours are not.** `LinearTransform.cu:76` asserts
`pts[0]->c0.getLevel() == ctxt.getLevel()` -- an *exact* match, checked before any
internal adjustment, unlike `multPt`'s implicit `Plaintext::adjustPlaintextToCiphertext`
step (`Ciphertext.cpp:393`) that our current per-diagonal `EvalMult(ct, pt)` calls silently
rely on. Our `raw_plain()` always builds at OpenFHE level 0
(`MakeCKKSPackedPlaintext(values, 1, 0, ...)`), but per the 2026-07-27 profiling (Finding
3 there), ciphertexts entering `matmul()` are *not* always level 0 -- roughly 2 levels are
consumed since the last client-boundary reset by the time of e.g. the MLP-projection call
sites. This is real, not a formality: `examples/bert-tiny/src/MatMul.cu` independently
confirms the pattern -- callers there explicitly `dropToLevel(...)` the ciphertext to match
precomputed plaintexts before calling `LinearTransform`, with their own caller-side level
asserts; the primitive itself does no implicit leveling. Traced the fix: `CryptoContext.cpp`
confirms `MakeCKKSPackedPlaintext`'s `level` argument passes straight through to OpenFHE's
own CPU-side encoding (`context->MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, ...)`
at `CryptoContext.cpp:496/520`), and `CiphertextImpl<DCRTPoly>::GetLevel()`
(`Ciphertext.cpp:56-64`) shows the OpenFHE-visible level is `maxDepth - <native level>` --
the same inverted convention used at `CryptoContext.cpp:615` to convert a native level back
to an OpenFHE `level` argument. So **passing a ciphertext's own `Ct::GetLevel()` as
`raw_plain()`'s `level` argument produces a plaintext whose native level matches that
ciphertext's native level exactly** -- no separate `adjustPlaintextToCiphertext` call
needed, just a non-zero `level` parameter threaded through `raw_plain()` at each call site
(currently hardcoded to `0` everywhere). This is a real code change (Step 2 below), not a
config flag.

**`[A]` Finding 5 (safety-critical, motivates an added runtime check): the pinned build is
`-DCMAKE_BUILD_TYPE=Release`, which defines `NDEBUG` and compiles out `assert()`.**
`LinearTransform`'s own level-match assert will therefore **not** fire on the real GPU
binary if Finding 4's level threading has a bug -- the call would silently proceed into
`rescale`/rotation math built on a false level premise (wrong ciphertext values at best;
at worst, an RNS-limb-count mismatch feeding the batched CUDA kernels, a similarly
dangerous class of failure to the diagcache crash). Because of this, Step 2's
implementation must add its own explicit `if (mismatch) throw` check before calling
`LinearTransform` -- relying on the library's own (compiled-out) assert is not sufficient
here.

**`[A]` Finding 6 (distinguishes this from the abandoned diagcache fix): `LinearTransform`
requires pre-adjusted plaintexts rather than adjusting them internally, so it does not
obviously touch the crash path.** The diagcache crash was root-caused to
`Ciphertext::multPt`'s implicit `Plaintext::adjustPlaintextToCiphertext` -> `RNSPoly::grow`
path choking when a *reused* native `Plaintext` object was grown a second time. Because
`LinearTransform`'s precondition demands the plaintext already be at the matching level
(no internal grow-on-demand), and this design builds all 1024 diagonal plaintexts fresh
per `matmul()` call -- each used exactly once inside one batched call, never reused across
two separate `multPt`-like calls -- it does not exercise the same reuse pattern
CLAUDE.md's ban targets. This is reasoning from the source, not a GPU-verified guarantee;
the isolated single-call-site gate (Step 2) is what actually tests it.

**`[A]` Finding 7 (non-blocking efficiency caveat): converting a shared-`baby` call site
loses today's rotation-reuse, but Finding 2 (2026-07-27) says that shouldn't matter.**
`LinearTransform(ctxt, ...)` takes the *un-rotated* input ciphertext and computes its own
baby-step hoisted rotations internally (`ctxt.rotate_hoisted(...)`) -- it has no parameter
to accept externally precomputed baby rotations. Our code currently computes
`baby_rotations()` once and reuses it across sibling `matmul()` calls (QKV: 3 calls share
one; MLP FC: 4 calls share one) -- converting one of those call sites to `LinearTransform`
means its baby-step rotation work is redone per call instead of shared. Since the
2026-07-27 profiling already found giant-step rotation/keyswitch is `<0.1%` of one matmul
call's cost (rotation is not the bottleneck at any granularity), redoing baby-step
rotation an extra 2-3x is expected to stay immaterial -- but this is an expectation, not
yet measured, and the full-block re-measurement (once all 24 sites are converted) is what
actually checks it.

**Verdict for Step 2:** proceed. Required, concrete changes -- not a drop-in swap: (a)
give `raw_plain()` an optional `level` parameter (default 0, preserving every other call
site's existing behavior) so a matmul call site can request plaintexts pre-encoded at its
input ciphertext's current level; (b) implement the QKV query projection (token 0) call
site via `LoadCiphertext`/`LoadPlaintext`/`GetDeviceCiphertext`/`GetDevicePlaintext` +
`FIDESlib::CKKS::LinearTransform(...)`, leaving the other 23 call sites on the existing
manual path; (c) add an explicit throwing level-match check (Finding 5) since the
library's own assert is compiled out; (d) verify with an isolated gate comparing that one
call's output against the existing oracle before touching any other call site, exactly as
this session's instructions specify.

### Scheme B `LinearTransform` Step 2: single-call-site implementation, local validation passes (2026-07-28)

New sibling `fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_lintransform.cpp`
(forked from the profiled prototype, source hash
`3c182a8c786f628d896cf96a8149a6f9f8ed945b3098038ea7b8359c1e0c2aed` before this file's own
edits -- the frozen ancestors it was forked from remain untouched: single-shot
`d88f1a09...`, cached `de79d6e7...`, profiled `4ca9c664...`, all still verified by
`test_lintransform_contract.py`). Converts exactly one of the six textual `matmul()` call
sites -- the QKV query projection, both `T=2` tokens -- to a new `matmul_lintransform()`
member that: builds the identical 1024 diagonal plaintexts (`packed_values`/`rolled_row`
formula copy-pasted unchanged from `bsgs_inner_loop`) pre-encoded at the input
ciphertext's current level (`raw_plain()` gained an optional `level` parameter, default 0,
every pre-existing call site unaffected); clones the input into an independent native
ciphertext (`CiphertextImpl<DCRTPoly>`'s copy constructor deep-clones via
`CopyDeviceCiphertext`, confirmed by reading `Ciphertext.cpp`, so `LinearTransform`'s
in-place mutation never touches `baby.values[0]`/`normalized[token]`, still needed by the
sibling key/value matmul calls and the manual BSGS path); adds an explicit
`if (level mismatch) throw` check (scoping Finding 5 -- the library's own assert is
compiled out under this pinned build's Release/NDEBUG); then calls
`FIDESlib::CKKS::LinearTransform(*ctxt_gpu, BSGS_N1*BSGS_N2, BSGS_N1, pts_gpu, 1, 0)`. The
other 23 call sites are untouched, on the original manual path. `CMakeLists.txt`/
`build_in_fideslib.sh` gained the new target; `run_scheme_b_lintransform.sh`/
`launch_brev_scheme_b_lintransform.sh`/`wait_and_run_scheme_b_lintransform.sh` were forked
from the profiled prototype's scripts with the `_lintransform_` tag marker (disjoint from
every other prototype's marker), to a new versioned remote dir
(`gpu_real_scheme_b_lintransform_v1`, not reusing any prior version) per this project's
standing Brev deployment convention.

**`[V]` Local validation: `test_lintransform_contract.py` (19 tests) plus the full existing
suite (`test_batching_contract.py`/`test_caching_contract.py`/`test_diagcache_contract.py`/
`test_profiling_contract.py`/`test_serialization_contract.py`/`test_warmup_contract.py`,
106 tests total) all pass.** Beyond the static source-text checks (frozen ancestors
untouched, exactly one call site converted, level-match check present, native clone
present, tag-marker discipline), this file's own numpy contract goes further than a static
check: it transliterates `LinearTransform.cu`'s literal FUSED-branch reverse-order
accumulation loop (not just its outer diagonal indexing) from source, independently
re-derives (by induction, recorded in `test_lintransform_contract.py`'s docstring) that
this reverse loop reduces to the closed form `sum_m rotate(S_m, m*bStep*stride)` -- the
exact same formula our own `matmul()` computes via a differently-ordered forward loop --
and checks both agree. It then feeds the *real* fixture's `weights.attn_qkv` and
`oracle.ln1_output` through both formulations and checks the result against the *real*
`oracle.query`, all at `atol=1e-9`. **All three agree exactly.** This is a
mathematical-equivalence check against this session's own reading of
`DotProductPtInternal`'s C++-level indexing, not a substitute for the real GPU gate (the
low-level `LTdotProductPtBatch` CUDA kernel body was not read) -- but it is meaningfully
stronger than confirming the diagonal indexing "looks the same," since it caught that
`LinearTransform`'s reverse, incrementally-rotated accumulation order is a genuinely
different control-flow shape from our own forward loop, and confirms by direct
computation on real weights that the two are mathematically identical, not merely
structurally similar.

`[U]` Not yet run on real GPU hardware -- this is local-only validation (static contract +
numpy arithmetic), no FIDESlib/CUDA build performed yet. The isolated single-call-site
correctness gate (comparing the full block's `global_rel_inf` against the unchanged 4e-2
oracle tolerance, exactly as the profiled/diagcache/warmup prototypes were gated) and the
full-block timing re-measurement are the next steps, pending a decision on Brev capacity
spend.

```bash
python3 fhe/gpu_real_scheme_b/test_lintransform_contract.py -v
```

**`[V]` Build-time finding: including FIDESlib's native `CKKS/Ciphertext.cuh`/
`Plaintext.cuh` from an external consumer needs OpenFHE's own headers on the include
path -- `fideslib::fideslib`'s exported CMake package does not provide them.** First real
build attempt failed with two rounds of errors, both fixed without touching any frozen
file or the FIDESlib install itself:

1. `#include <CKKS/LinearTransform.cuh>` alone only forward-declares
   `FIDESlib::CKKS::Ciphertext`/`Plaintext` (via `forwardDefs.cuh`) -- calling `->c0`/
   `->getLevel()` on them needs the full class definitions. Fixed by also including
   `<CKKS/Ciphertext.cuh>`/`<CKKS/Plaintext.cuh>` directly, matching
   `api/CryptoContext.cpp`'s own include list.
2. Those two headers transitively pull in `openfhe.h` (via
   `CKKS/openfhe-interface/RawCiphertext.cuh`), which is **not** reachable from this
   target: `fideslibTargets.cmake`'s exported `INTERFACE_INCLUDE_DIRECTORIES` covers only
   FIDESlib's own installed headers, confirmed by grepping the installed
   `fideslibTargets.cmake` directly. `examples/bert-tiny/CMakeLists.txt` independently
   confirms this is expected, not a packaging bug -- it adds OpenFHE's include dirs
   itself for the same reason. Fixed by adding a `find_package(OpenFHE CONFIG REQUIRED
   PATHS /usr/local/lib/OpenFHE)` call and `target_include_directories(... PRIVATE
   ${OpenFHE_INCLUDE} ${OpenFHE_INCLUDE}/core ${OpenFHE_INCLUDE}/pke
   ${OpenFHE_INCLUDE}/binfhe)` scoped to only the new
   `real_dnagpt_fides_scheme_b_lintransform` target in `CMakeLists.txt` -- every other
   target in this tree only uses the OpenFHE-API-compatible `<fideslib.hpp>` wrapper and
   is untouched by this change.

After both fixes, `cmake --build build --target real_dnagpt_fides_scheme_b_lintransform`
exits 0 and produces the binary. Build log:
`/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724/gpu_real_scheme_b_lintransform_v1/gpu_real_scheme_b/build3.log`
on the Brev host.

### Scheme B `LinearTransform` isolated single-call-site gate: correctness passes, timing regresses (2026-07-28)

Ran `real_dnagpt_fides_scheme_b_lintransform` (source hash `915cbd38a9ca5b6e94865f08956328533eeba3d8248d007e583a28397fa5accd`) on Brev physical GPU 0, host load `~6` (`load average: 6.17, 6.26, 21.94` -- by far the cleanest host conditions of any session in this project's history; every prior Scheme B session saw load in the 150-1500+ range). Evidence:
`fhe_fides_real_d768_t2_full_lintransform_scheme_b_a100_20260728.json`.

**`[V]` Correctness: PASS.** `global_rel_inf=4.59e-10`, `worst_token_rel_inf=5.54e-10`, both far inside the unchanged `4e-2` gate and consistent with every prior Scheme B run's `~1e-10` range. The single converted call site (QKV query projection, both `T=2` tokens, via `FIDESlib::CKKS::LinearTransform`) produces output indistinguishable from the manual path's historical accuracy once it propagates through the rest of the unmodified block. This confirms the scoping analysis's diagonal-layout and level-threading translation (Findings 3-4 above) is correct on real hardware, not just in the numpy reference check.

**`[V]` Timing: the converted call is measurably SLOWER than its unconverted siblings in the same run, not faster.** This is the load-bearing, apples-to-apples comparison -- same run, same host load, same GPU, so ambient contention cancels out:

| bucket | path | mean seconds/call | vs `matmul_qkv_query` |
|---|---|---:|---:|
| `matmul_qkv_query` | **LinearTransform** | **17.29** | -- |
| `matmul_qkv_key` | manual (unconverted) | 10.69 | query is `1.62x` slower |
| `matmul_qkv_value` | manual (unconverted) | 9.86 | query is `1.75x` slower |
| `matmul_attention_projection` | manual (unconverted) | 9.18 | query is `1.88x` slower |
| `matmul_mlp_fc` | manual (unconverted) | 9.39 | query is `1.84x` slower |
| `matmul_mlp_projection` | manual (unconverted) | 9.41 | query is `1.84x` slower |

`[A]` The lost baby-rotation-reuse (scoping Finding 7 -- `LinearTransform` recomputes its own 31 baby-step rotations instead of sharing the `baby_rotations()` object Q/K/V already share) does **not** plausibly explain a `~7-8s` gap: the 2026-07-27 profiling's Finding 2 measured giant/baby rotation at `<0.1%` of one manual call's cost (tens of milliseconds on a call costing tens of seconds), regardless of how many times it is repeated. Something about the batched primitive itself -- not the rotation overhead around it -- is the more likely explanation, but the exact mechanism is `[U]` unresolved (the low-level `LTdotProductPtBatch` CUDA kernel was not profiled internally the way the manual path's `detail_*` sub-buckets were in the 2026-07-27 session; that instrumentation doesn't exist for this new call path).

`[A]` **Do not read the overall wall-time drop (`encrypted_evaluation=247.51s` vs the 2026-07-27 baseline's `533.13s`, a `2.16x` improvement) as evidence the swap helped.** Only 1 of 24 calls changed, and that one call got *slower*, not faster, within this run. The overall drop is fully explained by this run's dramatically lower host contention (`load~6` vs whatever the 2026-07-27 session experienced, unrecorded but evidently far worse given every *unconverted* sibling bucket also dropped 1.4-3.5x versus its 2026-07-27 counterpart, e.g. `matmul_mlp_projection` `9.41s` here vs `13.28s` then). Cross-run absolute-time comparisons are confounded by host load exactly as every prior session's caveats describe; the within-run comparison above is the only comparison not subject to that confound, and it points the opposite direction from the hypothesis motivating this work.

**Verdict: the native batched `LinearTransform` primitive does not deliver the hoped-for speedup at this one call site, on this pinned build, under real GPU measurement.** This is a valid negative finding per CLAUDE.md, not a failure to force around. Converting the remaining 23 call sites is very likely to reproduce the same regression at each site (nothing about the QKV query call site is structurally special versus the others) and was not attempted -- spending further Brev capacity to mechanically confirm an already-observed direction was judged not worthwhile without first understanding *why* the batched primitive is slower here, which would require reading `RNSPoly::LTdotProductPtBatch`'s CUDA kernel body (not done this session; out of scope without further explicit direction). The lever identified 2026-07-28 ("Next optimization candidate... targets the actual bottleneck directly") is **closed as measured-negative for this pinned FIDESlib build**, joining the diagcache fix as an explored-and-abandoned lever, though for a different reason (measured slower, not GPU-crash-unsafe).

```bash
brev exec awesome-gpu-name -- "cat .../gpu_real_scheme_b_lintransform_v1/gpu_real_scheme_b/evidence/fhe_fides_real_d768_t2_full_lintransform_scheme_b_a100_20260728.json"
```

### General causal-attention Scheme B circuit (T>2): design, proof, and first real-GPU gate (2026-07-28)

Every Scheme B speed lever above (batching, caching, serialization, profiling,
diagcache, warm-up, `LinearTransform`) was measured against the **T=2 closed-form
identity** (`softmax([s0,s1])[1]=sigmoid(s1-s0)`), which does not generalize: it is
specific to a two-token causal window, not a real attention circuit. Before spending
further effort on speed tuning a non-representative T=2 circuit, this session designed
and measured the actual general causal-attention Scheme B circuit, closing the `[U]`
flagged at line 172 above and in `docs/hybrid/roadmap.md` step 5.

**Design.** The frozen file's `Client::cross_boundary_impl` (decrypt -> arbitrary
plaintext transform -> re-encrypt) was already fully general — nothing about its
mechanics is specific to a sigmoid or to two tokens. For causal row `i>=1` (query token
`i` attends to key/value tokens `0..i`), the general circuit:

1. builds one packed ciphertext per row holding, for all 12 heads, the `i+1` raw causal
   scores `s_i0..s_ii` — each isolated to its own reserved slot
   (`head*SCORE_SLOT_STRIDE+col`, `SCORE_SLOT_STRIDE=T`) via a one-hot plaintext mask,
   disjoint from any head's real 64-wide value span (`static_assert`-enforced at
   compile time: `HEADS*SCORE_SLOT_STRIDE <= PACK_WIDTH-D`);
2. for each non-trivial causal column `j=1..i`, one new client boundary call
   (`Client::cross_boundary_softmax_select`, the only new method added to `Client` —
   `cross_boundary`/`cross_boundary_active`/`cross_boundary_impl` are byte-for-byte
   unchanged) computes the exact, numerically-stable softmax over the row's `i+1`
   scores per head and returns weight `j` broadcast across that head's real value span,
   all 12 heads batched into the single round trip (the same "many logical values, one
   ciphertext" batching already adopted for the T=2 file's attention sigmoid and GELU
   chunks);
3. `context_i = v0 + sum_{j=1}^{i} weight_ij*(v_j-v0)` — reusing, not replacing, the T=2
   file's own algebraic reduction that folds the `j=0` weight into an implicit
   "1 minus the rest" term.

Row 0 is trivial (`context_0=value_0`, no boundary call). This makes round trips grow as
`sum_{i=1}^{T-1} i = T*(T-1)/2` (1 at T=2, exactly matching the frozen file — proof this
is a strict generalization, not a different algorithm coincidentally agreeing at one
data point) and raw-score computation grow as the standard `O(T^2)` any causal-attention
implementation pays, plaintext or encrypted.

**Numpy proof before any C++.** `fhe/realweights/export_fixture.py` was already
T-agnostic (`--tokens N` flag, `fhe/realweights/manual.py`'s `block_reference()` exports
`attention_scores`/`causal_mask`/`attention_probabilities`/`attention_context_*` for any
T) — only `fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b.cpp`'s `T=2` constant
and sigmoid identity were T-locked. Exported a new real fixture
(`gsr_pos0_block0_t3_d63353abdc1a_52d046d1fcf0`, `--tokens 3`, same checkpoint/GSR FASTA
as the T=2 fixture — weight file hashes are bit-identical, confirming no drift) and
transliterated the proposed circuit in numpy against it: reconstructed
`attention_context_heads`/`attention_context_merged`/`attention_projection` matched the
real oracle to `atol=1e-9`, and the T=2 identity was confirmed as the `row_length=2,
select_index=1` special case of the same softmax-select formula (`sigmoid` match to
`1e-12` over 20 random score pairs). Captured as
`fhe/gpu_real_scheme_b/test_general_attention_contract.py` (16 tests, all pass, no GPU
required) alongside static structural checks: frozen T=2 source hash unchanged, shared
constants match except `T`, exactly one new `Client` public method, `EncryptedEvaluator`
still has zero `Decrypt`/`secretKey` references, run/launch/orchestrator scripts and
CMake reference the new binary with a `_general_attention_` tag marker disjoint from
every other prototype's.

**Real-GPU gates: both PASS.** Forked
`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_general_attention.cpp` from the
frozen T=2 file (confirmed untouched, `source_sha256
d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df`), built clean via
`build_in_fideslib.sh`, ran on Brev physical GPU 0:

| gate | `global_rel_inf` | round trips | `encrypted_evaluation_seconds` |
|---|---:|---:|---:|
| `attention` | `3.19e-10` | 6 (3 LN1 + 3 softmax, `T*(T-1)/2=3`) | 130.61 |
| `full` | `3.65e-10` | 12 (+3 GELU + 3 LN2) | 519.57 |

Evidence: `fhe_fides_real_d768_t3_attention_general_attention_scheme_b_a100_20260728.json`,
`fhe_fides_real_d768_t3_full_general_attention_scheme_b_a100_20260728.json` (manifest
rows in `results/hybrid/manifest.yaml`). Both pass the unchanged `4e-2` gate at the same
`~1e-10` precision band as every prior Scheme B run — going from T=2 to T=3 cost no
accuracy, and the `full` gate confirms the general-attention circuit composes correctly
with the rest of the (unmodified) block, not just in isolation.

**`[V]` This closes the T>2 design gap.** A general causal-attention Scheme B circuit
now exists, is measured correct on real GPU hardware, and requires no algebraic
re-derivation per sequence length (unlike the T=2 identity). **`[U]` No speed claim is
made.** This run's `519.57s` is not compared against the T=2 block-0 anchor (`372.27s`)
— T differs (3 vs 2 tokens, 36 vs 24 `matmul()` calls) and host load was not controlled
for. `[U]` Only T=3 has been measured; scaling to task-representative lengths
(`T=32/64/103` per `docs/roadmap.md`) is unmeasured, and the `O(T)`/`O(T^2)` round-trip/
score-count growth is a design property, not yet empirically confirmed beyond T=3. The
speed levers closed earlier in this document (multi-GPU sharding, kernel fusion, depth
re-derivation) remain unexplored and are now better-motivated to revisit against this
general circuit rather than the T=2-only one.

### T=8 growth-trend check and output-packing ceiling (2026-07-29)

Empirically confirming the O(T)/O(T^2) growth the T=3 design predicted, at a second
data point, before any speed optimization. Forked
`real_dnagpt_fides_scheme_b_general_attention_t8.cpp` from the T=3 anchor (only T and
the pinned fixture manifest changed at first), exported a T=8 fixture (`--tokens 8`,
same checkpoint/GSR FASTA — weight file hashes bit-identical to T=2/T=3's), and
verified the causal-softmax design in numpy first (as with T=3): matched the oracle,
predicted `round_trips=28` (`T*(T-1)/2`) and `attention_score` calls `=420`
(`heads*(T*(T+1)/2-1)`).

**Two attempts crashed with zero diagnostics.** Real-GPU run 1 crashed after ~28
minutes: `malloc(): unaligned tcache chunk detected` (glibc heap corruption). Run 2
(after an SSH-blip caused a concurrent double-launch on the same GPU, unrelated
crash) confirmed the same signature on a clean, confirmed-idle GPU. Both times the
run log showed almost nothing — not even the `[context]`/`[stage]` prints that
should have appeared before any real work — because `std::cout` is fully buffered
when redirected to a file (as the launch scripts do) and a hard crash never flushes
that buffer.

**Localizing the crash.** Added `std::endl`-flushed versions of every existing
`[stage]`/`[context]` print, plus a new per-`(row, col)` progress line inside the
causal-attention loop (`[stage] attention row=R col=C done, level=L`) — the T=3 file
only logs once before and once after the *entire* attention block, which cannot
localize a mid-block crash. Rebuilt, retried: **the run log now showed all 28
predicted `(row, col)` pairs complete successfully**, followed by `attention_context`
and `attention_projection`'s stage logs — then the crash. This fully exonerates the
causal-attention circuit itself (identical to the just-closed T=3 design, now proven
correct at T=8 too) and pins the bug to whatever runs immediately after: `finish()`.

**Root cause.** `finish()`/`output_block_plain()`/`measure()` (unchanged since the
frozen T=2 file) silently assume `T<=COPIES=4` — the final output-packing step
addresses each token by its own `PACK_WIDTH`-wide physical ciphertext slot, and there
are only `COPIES=4` such slots in the `SLOTS=COPIES*PACK_WIDTH=4096` layout:

```cpp
Plaintext output_block_plain(std::size_t token) {
    if (token >= T) { throw ...; }                    // only checked against T
    std::vector<double> packed(SLOTS);                 // SLOTS = 4096
    std::fill_n(packed.begin() + token * PACK_WIDTH, D, 1.0);  // token=4: offset 4096 -- OOB
```

At T=2/T=3 every token index is `< COPIES`, so this never manifested. At T=8, tokens
4-7 write `std::fill_n` past the end of a heap-allocated `std::vector<double>`,
corrupting adjacent heap metadata — the corruption is detected later, on a
subsequent `malloc()`, which is why the crash message pointed nowhere near the real
bug and why localization required the flushed per-row logging above. This is a
**second, more restrictive scaling ceiling** than the causal-score packing one found
during T=3 design (`HEADS*SCORE_SLOT_STRIDE<=PACK_WIDTH-D`, i.e. `T<=21`): the
output-packing ceiling was `T<=COPIES=4`, tighter and previously undiscovered because
no gate had ever run above T=3.

**Fix.** Group tokens into `OUTPUT_GROUPS=ceil(T/COPIES)` output ciphertexts instead
of one; within a group, address a token by its position *inside* that group
(`local_slot`, bounded by `COPIES`) rather than its raw index. `EvaluationResult`'s
`packed_output` becomes `std::vector<Ct>` (one per group); `main()` decrypts each
group and reassembles a flat `T*D` output vector (simplifying `measure()`, which no
longer needs to know about the raw `PACK_WIDTH`-slot layout at all — it now compares
two plain `T*D` arrays). `output_block_plain()`'s bounds check moved from `token>=T`
to `local_slot>=COPIES`, the actual invariant that matters.

**Result: PASS.** `fhe_fides_real_d768_t8_attention_general_attention_t8_scheme_b_a100_20260729_fixed.json`
— `global_rel_inf=3.69e-10` (same `~1e-10` band as every other Scheme B gate),
`round_trips=36` (28 attention + 8 LN1, exactly as designed), `final_decrypt_calls=2`
(`=ceil(8/4)`, confirming the fix's own arithmetic). `[V]` The output-packing fix is
general — works for any `T`, not just `T<=8` — closing this ceiling rather than
patching around it for one value.

`[U]` No speed claim: `523.65s` (T=8, 32 `matmul()` calls) is not compared against the
T=3 anchor's `130.61s` (T=3, 12 `matmul()` calls) — different T, and the shared Brev
host was fully occupied by another tenant across all 8 GPUs for several hours before
this run's GPU freed up (confirmed via `nvidia-smi` before and during — not inferred).
`[U]` Only the `attention` gate was re-run at T=8; `full` (LN2+GELU+MLP) was not.
`[U]` Scaling to task-representative lengths (`T=32/64/103`) remains unmeasured and
would need the causal-score packing ceiling addressed too (`T<=21` currently, `T<=85`
with the already-scoped 4-copy fix, per the T=3 entry above) — the output-packing fix
in this entry only removes the *other* ceiling.

Reproduction:
```bash
python -m fhe.realweights.export_fixture --tokens 8
# build via fhe/gpu_real_scheme_b/build_in_fideslib.sh (target
# real_dnagpt_fides_scheme_b_general_attention_t8), then
fhe/gpu_real_scheme_b/run_scheme_b_general_attention_t8.sh 0 \
  checkpoints/fhe_exports/gsr_pos0_block0_t8_d63353abdc1a_52d046d1fcf0 \
  <output.json> attention
```
