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

### T=32 spread-across-copies causal-score packing gate (2026-07-29)

`[V]` The T=32 real-GPU evidence has landed and passes:
`results/runs/fhe_fides_real_d768_t32_attention_general_attention_t32_scheme_b_a100_20260729.json`
(result SHA-256
`a6ae130008bd42a4f5d142cd5b7a964114fd3242dac5d76dfc2ad9724a98ddba`).
The run is bound to source SHA-256
`607e9c429bdea107149733e10f19765578175a5301b350556bd8604a1d384252`,
fixture manifest SHA-256
`6ae018b419a533dd9c77e6e0415bf7adf2b67c1f444013e68489a2fc84cca003`,
and fixture contract SHA-256
`44d5dbf1d23a3a2c1ebcee43e69ff3a43e0c1ab1c8a11b6947282d287b52cb41`;
all three match the corresponding local files.

`[V]` `global_rel_inf=2.14e-10`, `worst_token_rel_inf=2.65e-10`,
`max_abs_error=3.56e-09`, and all outputs are finite against the unchanged
`4e-2` gate. The run completed all 496 predicted causal-attention column
crossings plus 32 LN1 crossings (`round_trips=528`), represented 5,984
logical boundary instances, and used `final_decrypt_calls=8=ceil(32/4)`.
This validates both the spread-across-copies score mapping at T=32 and the
general output-grouping fix beyond its T=8 checkpoint.

`[V]` The measured encrypted attention gate took `3823.09s`, split into
`3703.50s` server linear algebra and `119.59s` client boundaries. This is
evidence of execution cost, not a clean throughput comparison: the shared
host was uncontrolled after a clean per-GPU preflight. `[U]` This is still
block 0 and the `attention` gate only; it does not validate the T=32 MLP,
multi-block composition, or any sequence length beyond the raised `T<=85`
packing ceiling.

Reproduction:
```bash
python -m fhe.realweights.export_fixture --tokens 32
# build via fhe/gpu_real_scheme_b/build_in_fideslib.sh (target
# real_dnagpt_fides_scheme_b_general_attention_t32), then
fhe/gpu_real_scheme_b/run_scheme_b_general_attention_t32.sh 0 \
  checkpoints/fhe_exports/gsr_pos0_block0_t32_d63353abdc1a_52d046d1fcf0 \
  <output.json> attention
```

### Scoped (not implemented): chunked/streaming causal-score packing for arbitrary T (2026-07-29)

After launching the T=32 real-GPU run (spread-across-copies causal-score packing fix,
now passed and recorded above), scoped the actual
fix needed to reach task-representative sequence lengths (`T~103` per `docs/roadmap.md`,
also `32/64`). The spread-across-copies fix raises the packing ceiling from `T<=21`
(`HEADS*SCORE_SLOT_STRIDE<=PACK_WIDTH-D`) to `T<=85`
(`HEADS_PER_COPY*SCORE_SLOT_STRIDE<=PACK_WIDTH-D`) — a real improvement, but `T~103`
still exceeds it. Any *fixed* per-copy reservation scheme has a hard ceiling regardless of
how cleverly heads are spread across copies, because it requires an entire causal row's
raw scores to sit in one ciphertext simultaneously so the client can compute
max/sum in a single decrypt. Removing the ceiling requires not packing the whole row into
one ciphertext at all.

**Design: two-phase chunked softmax, chunk width independent of T.**

- Fix a chunk width `C = (PACK_WIDTH-D)/HEADS_PER_COPY` (`=85` at current constants) —
  a constant, not derived from `T`. Any row, however long, is split into
  `ceil(row_length/C)` chunks of at most `C` columns, each packed exactly like today's
  single-row packing (spread across copies, one-hot isolated) but only for that chunk's
  local columns.
- **Phase A (reduction, `ceil(row_length/C)` crossings per row):** for each chunk, the
  client decrypts and appends the recovered per-head values into its own in-process
  cache for that row. Once every chunk for the row has arrived, the client computes the
  row's max and softmax denominator over the *full* reassembled row — this needs no
  incremental/online-softmax rescaling trick (the numerical-stability concern that
  hardware-streaming flash-attention kernels have to solve): the client is a single
  process with the whole row in float64 memory by the time the last chunk lands, so
  computing max/sum directly is exact and trivial. The ciphertext this phase returns to
  the server carries no information the rest of the computation depends on (the
  server-side graph never uses it) — it exists purely to preserve the existing
  `cross_boundary_impl` request/response symmetry and honest crossing-count bookkeeping;
  a leaner decrypt-only variant is possible but changes what "one round trip" means and
  was not pursued here.
- **Phase B (selection, exactly `row` crossings per row, same count as today):**
  unchanged from the current per-column loop — for column `j`, the client already has
  `s_ij` and the row's max/sum cached from phase A, so it needs no new decrypt at all,
  just an encrypt-only crossing emitting `weight_ij = exp(s_ij-max)/sum` broadcast across
  all copies exactly as today.

**Numpy proof (this session, before any C++ change):**

1. `[V]` Chunk partitioning is a bijection over every column `0..row_length-1` — checked
   for `row_length in {1,2,9,32,85,86,103,200,1000}` and `chunk_width in
   {1,5,85,86}`: every column covered exactly once, no chunk exceeds the fixed width.
2. `[V]` The two-phase chunked design reproduces the current single-shot softmax
   bit-for-bit (`atol=1e-13`) for every `(row_length, chunk_width)` tested at
   `row_length<=85` (inside today's ceiling, so directly comparable).
3. `[V]` The chunked design matches a plain reference softmax at `row_length in
   {86,103,200,1000}` — lengths the *current* packing scheme cannot run at all
   (`HEADS_PER_COPY*T<=256` fails past `T=85`), proving the ceiling genuinely
   disappears, not just moves.
4. `[V]` Honest round-trip growth, `chunk_width=85`: `new_round_trips(T) =
   old_round_trips(T) + sum_{row=1}^{T-1} ceil((row+1)/85)`. At `T<=85` (still inside
   the old ceiling, so both are computable) the overhead is exactly `T-1` extra
   crossings (one reduction chunk per row, since the whole row fits in one chunk) —
   measured: `T=8` old=28/new=35 (+7), `T=32` old=496/new=527 (+31), `T=85`
   old=3570/new=3654 (+84). Past `T=85` there is no "old" to compare against; `T=103`
   is `5373` total crossings, `T=200` is `20244` — polynomial, not exponential, growth.

**Not done:** no chunked-softmax C++ source, contract test file, fixture, or GPU run. This is
scoping only, in the same spirit as the 2026-07-27 `LinearTransform`/diagonal-cache
scoping sessions — recorded so the next session can implement directly from this
design rather than re-deriving it. `[U]` Phase A's "return a ciphertext the server
ignores" choice versus a leaner
decrypt-only crossing is an open implementation decision, not yet resolved.

### T=32 causal-score packing fix: spread across copies, real-GPU attention gate (2026-07-29)

Implements the fix scoped in the T=8 entry above. Forked
`real_dnagpt_fides_scheme_b_general_attention_t32.cpp` from the frozen T=8 file
(source hash confirmed unchanged by the new contract test, alongside the T=3 and T=2
sources), carrying its output-packing fix forward unchanged
(`OUTPUT_GROUPS=ceil(32/4)=8`) and replacing the causal-score packing: instead of
`one_hot_plain` isolating each head's raw score into the SAME slot in all `COPIES=4`
physical copies (pure redundancy — `packed_scores` never goes through
`sum_broadcast`/cross-copy rotation, so three of the four copies carried no
information the client boundary ever used), `HEADS_PER_COPY=HEADS/COPIES=3` heads are
now spread one-per-slot into each copy via a new `one_hot_single_copy_plain(copy,
slot)`. The read side (`cross_boundary_softmax_select`) now iterates per global head,
derives `(copy, local_head)` from the assignment, and reads that head's row from its
own copy; the write side (the softmax weight broadcast, multiplied against
fully-replicated value ciphertexts) is unchanged. This raises the packing ceiling from
`HEADS*SCORE_SLOT_STRIDE<=PACK_WIDTH-D` (`T<=21`) to
`HEADS_PER_COPY*SCORE_SLOT_STRIDE<=PACK_WIDTH-D` (`T<=85`).

**Numpy-proven before any GPU work** (28 static/numpy tests in
`test_general_attention_t32_contract.py`, all passing locally, no FIDESlib/CUDA build
required): the head->copy/local-slot assignment is a bijection covering all 12 heads
exactly once with every local index `<HEADS_PER_COPY`; the sparse per-copy read
reproduces the old all-copies-redundant read's exact softmax weights, both on
synthetic random scores and on every non-trivial causal row of the real T=32 oracle
fixture (`oracle.attention_scores`); the packing-ceiling arithmetic itself confirms
`T<=85` (`21` for the old scheme, matching the T=8 entry's derivation).

Exported the T=32 fixture (`--tokens 32`, same checkpoint/GSR FASTA — weight-file
hashes bit-identical to T=2/T=3/T=8's, only `input__embeddings.bin` and the three
oracle arrays differ). Built on Brev (`build_in_fideslib.sh`, new
`real_dnagpt_fides_scheme_b_general_attention_t32` target alongside the existing
ones), deployed via `brev copy` into the same `gpu_real_scheme_b_general_attention_v1`
tree used by the T=3/T=8 gates (additive only, no existing file touched), then
launched via the capacity-aware detached orchestrator
(`wait_and_run_scheme_b_general_attention_t32.sh`, same `LOAD_THRESHOLD`/
`POLL_SECONDS` polling and nohup+`.done`-sentinel launch pattern as the T=8 orchestrator
— the host had all 8 GPUs occupied by another tenant for the first ~8 minutes of
polling before GPU 0 freed up).

**Result: PASS.**
`fhe_fides_real_d768_t32_attention_general_attention_t32_scheme_b_a100_20260729.json`
— `global_rel_inf=2.135e-10`, `worst_token_rel_inf=2.646e-10` (same `~1e-10` band as
every other Scheme B gate, well inside the unchanged `4e-2` tolerance),
`round_trips=528` (`=496` attention softmax crossings, `T*(T-1)/2`, `+32` LN1
invsqrts — matching the design prediction exactly, and confirming the T=8 entry's
growth formula holds at a third data point), `final_decrypt_calls=8` (`=ceil(32/4)`,
confirming the carried-forward output-packing fix's arithmetic at a new `T`).
`3823.09s` total (`3703.50s` server + `119.59s` client boundary, `~3.1%` client
share).

`[U]` No speed claim: this run is not compared against the T=8 anchor (`523.65s`) — T
differs 4x (32 vs 8 tokens, 4x the QKV/MLP matmul work) and host load was not
controlled for (the shared host had all 8 GPUs occupied by another tenant for the
first several minutes of this session's polling window). `[U]` Only the `attention`
gate is confirmed at T=32 in this entry; `full` (LN2+GELU+MLP) is a separate,
subsequent run. `[U]` T=32 still sits comfortably under the raised `T<=85` ceiling —
this run does not test the ceiling itself, only the growth trend at a value inside
it. Task-representative `T~103` remains blocked on the chunked/streaming softmax
design scoped in the entry immediately above.

Reproduction:
```bash
python -m fhe.realweights.export_fixture --tokens 32
python3 fhe/gpu_real_scheme_b/test_general_attention_t32_contract.py -v
# build via fhe/gpu_real_scheme_b/build_in_fideslib.sh (target
# real_dnagpt_fides_scheme_b_general_attention_t32), then
fhe/gpu_real_scheme_b/run_scheme_b_general_attention_t32.sh 0 \
  checkpoints/fhe_exports/gsr_pos0_block0_t32_d63353abdc1a_52d046d1fcf0 \
  <output.json> attention
```

### T=32 full gate: complete block-0 (2026-07-29)

Same session, same binary, `--gate full`, launched via the same detached
orchestrator once GPU 4 freed up (~17 minutes after the `attention` gate above
completed — the shared host had all 8 GPUs occupied by another tenant across that
window). Extends the `attention`-only gate with the unchanged LN2/GELU/MLP/residual
path already proven at T=2/T=3/T=8 — no new design, no new fix, just the first
complete-block confirmation that the spread-across-copies packing fix and the
carried-forward output-packing fix compose correctly through the whole graph.

**Result: PASS.**
`fhe_fides_real_d768_t32_full_general_attention_t32_scheme_b_a100_20260729.json` —
`global_rel_inf=2.478e-10`, `worst_token_rel_inf=2.981e-10` (same `~1e-10` band as
every other Scheme B gate), `round_trips=592` (`=528` from the attention gate `+32`
LN2 invsqrts `+32` GELU 4-chunk-batched crossings, one per token),
`final_decrypt_calls=8` (unchanged from the attention-only gate, confirming the
output-packing fix composes across the full block). `8228.95s` total (`8090.27s`
server + `138.68s` client boundary, `~1.7%` client share) — consistent with the
server-GPU-dominated split seen at every prior gate.

`[U]` No speed claim: not compared against any smaller-`T` anchor (T differs, host
load uncontrolled — this run itself waited on GPU capacity). `[U]` This closes the
then-active T=32 growth-trend/packing-fix milestone; T=32
still sits comfortably under the raised `T<=85` ceiling, and task-representative
`T~103` remains blocked on the chunked/streaming softmax design scoped above. `[U]`
Single block-0 gate only — no multi-block composition attempted, consistent with
every other Scheme B evidence file to date.

Reproduction: same as the `attention` gate above, with `--gate full` (last
positional argument to `run_scheme_b_general_attention_t32.sh`).

### First Scheme B multi-block composition gate: blocks 0 -> 1 at T=2 (2026-07-29)

Forked `real_dnagpt_fides_scheme_b_two_block_refresh.cpp` from the frozen T=2
single-block source, pinning the frozen source hash before any change. The new
fixture contract binds released blocks 0 and 1 and proves that the saved block-0
oracle output is byte-identical to block 1's plaintext input. The client refresh
protocol decrypts block 0's packed hidden state, unpacks both token vectors, and
re-encrypts them as fresh level-0 replicated ciphertexts under the same crypto
context and key lineage. The evaluator never receives the private key.

The local contract suite passes `12/12`; the full Scheme B test suite passes
`174/174`. The real A100 gate also passes:

- `[V]` Block 0 refresh-boundary `global_rel_inf=9.245e-11`; block 1 final
  `global_rel_inf=8.485e-11`, both far inside the unchanged `4e-2` tolerance.
- `[V]` Block 0 exits at packed level 6 and block 1 starts at level 0 after the
  refresh. Both blocks independently reproduce the same level trace, including
  `LN2 max=8`; no depth accumulates across the block boundary.
- `[V]` Protocol counts are 14 nonlinearity crossings plus one full-hidden-state
  refresh crossing (`15` declared crossings, `50` logical instances), with no
  bootstraps and no undeclared intermediate decrypt.
- `[V]` PID-specific `nvidia-smi` telemetry (193 samples, nominally every 5 seconds)
  records a peak of `10872 MiB`. The process reached that peak during block 0 and
  never exceeded it during block 1, so T=2 sequential composition does not
  accumulate another block-sized allocation.
- `[U]` No speed claim. GPU 0 passed the preflight at 543 MiB/0% utilization, but a
  second tenant appeared after launch. The measured wall time is valid as an event
  record, not as an uncontended performance comparison.
- `[U]` This proves blocks 0 -> 1 at T=2 only. It does not yet prove all 12 blocks,
  the final task-output head, or memory sufficiency at T~100.

Immutable evidence:

- `results/runs/fhe_fides_real_d768_t2_blocks0_1_scheme_b_two_block_refresh_a100_20260729_v2.json`
  (`sha256=e8c0ceb281815a85476bb6197cb1493b301045247cecb01f31e5f434ac58b65b`)
- `results/runs/fhe_fides_real_d768_t2_blocks0_1_scheme_b_two_block_refresh_vram_a100_20260729_v2.json`
  (`sha256=fd21f2fc29d5357bcbf17f5f5d65fd91531a297dd1d6c8d14db7f7a07f3e844f`;
  raw telemetry log `sha256=d568c86f815c55f086978fed74137d95d0cc496c7bd9edce93ebfe9f7ca2f614`)

Exact correctness reproduction:

```bash
.venv/bin/python -m unittest fhe.gpu_real_scheme_b.test_two_block_refresh_contract -v
# Build target real_dnagpt_fides_scheme_b_two_block_refresh via
# fhe/gpu_real_scheme_b/build_in_fideslib.sh in the pinned FIDESlib container, then:
fhe/gpu_real_scheme_b/run_scheme_b_two_block_refresh.sh 0 \
  checkpoints/fhe_exports/gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0 \
  results/runs/<new_tag_containing_scheme_b_and_two_block_refresh>.json
```

Exact telemetry sampler (run concurrently with the correctness command, directing
stdout to a new immutable log):

```bash
while true; do
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
    --format=csv,noheader,nounits
  sleep 5
done
```

### T=103 chunked-softmax attention gate (2026-07-29/30)

Implemented the two-phase chunked causal softmax scoped above in
`real_dnagpt_fides_scheme_b_general_attention_t103.cpp`. The fixed
`CHUNK_WIDTH=85` is independent of `T`: phase A sends each row's one or two score
chunks through the declared client boundary and reassembles the complete causal row
in client float64 memory; the client then computes one stable full-row softmax.
Phase B performs no further decrypt for those scores and emits the cached
per-column weights used by the encrypted value accumulation. The T=103 source
carries forward the already-proven output grouping
`OUTPUT_GROUPS=ceil(T/COPIES)=26`.

`[V]` The local source/fixture contract passes all 27 tests, including exact
partition coverage, the 85/86 chunk boundary, the 103-token partial final chunk,
stable-softmax agreement on real and synthetic rows, the predicted protocol count,
frozen-source hash checks, and build/run wiring. The pinned hashes are:

- source:
  `70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f`
- fixture manifest:
  `d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6`
- fixture contract:
  `061d53bd25bbcaf75d4c12067ec77f032c3ba0e35300a0a6168c2ce15f3672db`

`[V]` The detached real-A100 attention gate completed successfully even though the
outer Brev/orchestrator session later lost its connection. The run's `.done`
sentinel is `0`, its log ends in
`REAL_DNAGPT_FIDES_SCHEME_B_GATE_PASS`, and the immutable result is:

`results/runs/fhe_fides_real_d768_t103_attention_general_attention_t103_scheme_b_a100_20260729.json`

Result SHA-256:
`8f22ccf7c34125710197b84ef0b571f2095f8fbb800f7eb42afbfdbf38b8daee`
(remote run-log SHA-256:
`aabb5f8df636d2c7f1a8eb32b18d0628c69c9b30dd0c7c61ca02a8a60d82dcc9`).

- `[V]` `global_rel_inf=3.6608055924837055e-10`,
  `worst_token_rel_inf=4.390401958450227e-10`, and
  `max_abs_error=6.096444238323784e-09`; all values are finite and the global
  error is `1.09e8` times inside the unchanged `4e-2` tolerance.
- `[V]` `round_trips=5476`: 103 LN1 crossings plus 5,373 chunked attention
  crossings. `logical_boundary_instances=127399` and
  `final_decrypt_calls=26=ceil(103/4)` also match the declared schedule.
- `[V]` The measured attention-gate event took `14082.36s` (`3.91h`):
  `13162.35s` server linear algebra plus `920.01s` client boundaries. This is an
  execution record on the shared host, not a clean throughput comparison.
- `[U]` This is block 0 through attention projection only. It does not validate
  LN2/GELU/MLP at T=103, a complete T=103 block, T=103 GPU-memory headroom,
  all 12 blocks, or the task-output head. No T=103 `full` gate was launched by
  this run.

Exact reproduction:

```bash
python -m fhe.realweights.export_fixture --tokens 103
python3 fhe/gpu_real_scheme_b/test_general_attention_t103_contract.py -v
# Build target real_dnagpt_fides_scheme_b_general_attention_t103 via
# fhe/gpu_real_scheme_b/build_in_fideslib.sh in the pinned FIDESlib container, then:
fhe/gpu_real_scheme_b/run_scheme_b_general_attention_t103.sh 0 \
  checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0 \
  results/runs/<new_tag_containing_scheme_b_and_general_attention_t103>.json \
  attention
```

### T=103 Token-SIMD matched linear micro-gate (2026-07-30)

The additive `real_dnagpt_fides_scheme_b_simd_linear_t103.cpp` micro-gate tests
the first structural Token-SIMD step without modifying any frozen Scheme B
implementation. Both modes use the same source, real T=103 fixture, CKKS
context configuration (`ring_dim=65536`, `batch_slots=32768`, depth 16), exact client LN1
boundary, encrypted query projection, and evidence schema:

- `packed`: 13 ciphertext groups, up to 8 logical tokens per ciphertext;
- `serial_control`: 103 ciphertexts, one logical token per ciphertext.

The source SHA-256 is
`926be6b276006dd6435713efa168d189af02e02589d143684991380bae7e9233`;
the static source contract SHA-256 is
`730ab0df519a857d618a264e9d401a67074402f56b7dac5d215d7e828d19bb5c`
and passes `12/12` locally and on Brev. The complete local Scheme B suite passed
`254/254` before deployment. The first compile-only attempt allocated no GPU
and exposed four FIDESlib API lvalue/const mismatches; the minimal correction
produced the pinned source above and the v2 build passed.

Both real-A100 correctness gates pass:

- `[V]` packed:
  `global_rel_inf=3.4048182778014454e-9`,
  `worst_token_rel_inf=6.712087988183497e-9`,
  13 matrix products and 13 client crossings,
  `encrypted_evaluation=306.542390898s`
  (`server_linear_algebra=302.858285571s`);
- `[V]` serial control:
  `global_rel_inf=8.836401749619686e-9`,
  `worst_token_rel_inf=1.86670267294098e-8`,
  103 matrix products and 103 client crossings,
  `encrypted_evaluation=2260.075644194s`
  (`server_linear_algebra=2227.005309607s`);
- `[V]` the exact dense-call reduction is `103/13=7.923076923x`. The observed
  encrypted-evaluation ratio is `2260.075644194/306.542390898=7.372799689x`;
  server-only ratio is `7.353291674x`; complete measured-stage ratio (fixture
  load + setup + encryption + evaluation + final decrypt) is `7.243345438x`.
- `[V]` PID-specific telemetry records packed/serial peaks of
  `9848/15992 MiB`; the packed micro-gate peak is `6144 MiB` lower.
- `[U]` This is not a clean dedicated-A100 timing comparison. Packed and serial
  ran at different times on physical GPUs 5 and 4, respectively, and both
  acquired co-tenants. The ratios above are observed event ratios only; the
  uncontended speedup remains unmeasured.
- `[U]` This proves exact LN1 plus one encrypted dense transform. It does not
  yet prove packed causal attention, a complete T=103 block, full-block T=103
  VRAM, all 12 blocks, or the task-output head.

Immutable evidence:

- `results/runs/fhe_fides_real_d768_t103_packed_simd_linear_t103_scheme_b_a100_20260730_v2.json`
  (`sha256=bb99734d51f94671ff419ff9bad3edcf2eb685eb33dbd0a0a0798c92e2dc8917`);
- `results/runs/fhe_fides_real_d768_t103_serial_control_simd_linear_t103_scheme_b_a100_20260730_v2.json`
  (`sha256=5217aa33025f96eb1ad6190ed141c5fceaa4f09f39f4d05a35bfd33e23b94eb9`);
- `results/runs/fhe_fides_real_d768_t103_packed_simd_linear_t103_scheme_b_vram_a100_20260730_v2.json`
  (`sha256=4f8499b702e1b5971fc0dfc2c217813652ebb964296a1ee76de2b61824342e30`;
  raw telemetry `sha256=8bf215dd206669732428260c53f55b4d512ef22c15ce24a7f4229faf21cffa67`);
- `results/runs/fhe_fides_real_d768_t103_serial_control_simd_linear_t103_scheme_b_vram_a100_20260730_v2.json`
  (`sha256=ba71798d0fa5a5e82ed1871ea2330b32dbcc5d32f0dac0def9720ba575d30467`;
  raw telemetry `sha256=3219bb2dcefc2c461fa81a913b60f117741a1b7ac215c0166048b33d6e7a0e53`).

Exact correctness reproduction (use new output paths; the scripts refuse to
overwrite evidence):

```bash
python3 fhe/gpu_real_scheme_b/test_simd_linear_source_contract.py -v
# Build target real_dnagpt_fides_scheme_b_simd_linear_t103 via
# fhe/gpu_real_scheme_b/build_in_fideslib.sh in the pinned FIDESlib container.
fhe/gpu_real_scheme_b/run_scheme_b_simd_linear_t103.sh 0 \
  checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0 \
  results/runs/<new_tag_containing_scheme_b_simd_linear_t103_packed_a100>.json \
  packed
fhe/gpu_real_scheme_b/run_scheme_b_simd_linear_t103.sh 0 \
  checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0 \
  results/runs/<new_tag_containing_scheme_b_simd_linear_t103_serial_control_a100>.json \
  serial_control
```

### Complete T=103 Token-SIMD block-0 gate (2026-07-30)

The additive
`real_dnagpt_fides_scheme_b_simd_full_t103.cpp` extends the proved B=8 layout
through the complete released-weight block without changing any frozen Scheme B
source. It evaluates LN1, Q/K/V, packed causal scores, exact stable softmax at
the declared client boundary, encrypted weight/value accumulation, attention
projection, the first residual, LN2, exact client GELU, all four MLP chunks,
the MLP projection, and the final residual. The evaluator owns no secret key
and makes no decrypt call.

Fail-closed provenance:

- source SHA-256:
  `97727a5bb2145b32749b746c3fc1b576811dfcc3222088278962b029fc1aea4b`;
- direct Token-SIMD linear parent:
  `926be6b276006dd6435713efa168d189af02e02589d143684991380bae7e9233`;
- frozen semantic T=103 anchor:
  `70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f`;
- compiled schedule:
  `c6b221f365ba6326f615c5554458d7bd092990d23c4ba0d5106ca7577cb7c3aa`;
- static contract SHA-256:
  `f7677e7a5bf4e68910d9e7a21af7f448ac14a3c886db5e5f38c7964a89f28a78`,
  `11/11` pass;
- complete local Scheme B discovery: `265/265` pass;
- the target compiled cleanly under pinned FIDESlib commit
  `786c7600fb2f16b724e0acf73df367b27b8afed6` before GPU allocation.

`[V]` The real A100 full gate passes:

- `global_rel_inf=4.777473702899847e-9`;
- `worst_token_rel_inf=5.07508500941268e-9`;
- `max_abs_error=7.659995376191331e-8`;
- all values finite and the unchanged `4e-2` gate passes;
- `156` encrypted dense products versus `1236` serial-equivalent
  (`7.923076923x` exact reduction);
- operation-count guards pass at `1506` ciphertext-ciphertext
  multiplications, `177734` ciphertext-plaintext multiplications, `8173`
  explicit rotations, and `8776` `AccumulateSum` calls;
- protocol guards pass at `91` score-tile decryptions, `727` weight-tile
  encryptions, `857` crossings, `129162` logical boundary instances, zero
  intermediate decrypt attempts, and 13 final measurement decrypts;
- `packed_output_level=6` under the conservative depth-16 context.

Observed timing is `6281.748445232s` encrypted evaluation:
`6140.143438612s` server linear algebra plus `141.605006620s` client
boundaries. `[U]` This is correctness timing, not a dedicated-A100 benchmark.
The run passed a clean preflight, but later acquired multiple co-tenants; one
cotenant reached `65868 MiB` and whole-GPU allocation reached `76757 MiB`.

`[V]` PID-specific telemetry contains 1196 five-second samples. The target
process starts at `4716 MiB` and peaks/finalizes at `12920 MiB`. This resolves
single-block T=103 capacity positively for an 80GB A100. `[U]` It does not
prove that 12-block composition preserves the same bounded live set.

Immutable evidence:

- correctness:
  `results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_scheme_b_a100_20260730.json`
  (`sha256=333461fbd83aad670d57f11d418cdb525d33d09e9d90d0f0390aaba2e8b6d6f5`);
- process-memory telemetry:
  `results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_scheme_b_vram_a100_20260730.json`
  (`sha256=9157f2e1188e68859ac9f2cb8ebe75a83f76952be7fdb84e4b3f87350c1fa752`);
- raw remote run log:
  `sha256=84480144fdb63f23e98223480485bb62276bfe87ba1303df06aad9fea1f9d59b`;
- raw remote VRAM log:
  `sha256=dbb0ee603c445adcd79dad8117cbc0fdaa144c2e84c9edaa53c265c25fa4fc6c`.

Exact correctness reproduction (use a new immutable output/tag):

```bash
python3 fhe/gpu_real_scheme_b/test_simd_full_t103_contract.py -v
# Build target real_dnagpt_fides_scheme_b_simd_full_t103 via
# fhe/gpu_real_scheme_b/build_in_fideslib.sh in the pinned FIDESlib container.
fhe/gpu_real_scheme_b/run_scheme_b_simd_full_t103.sh 0 \
  checkpoints/fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0 \
  results/runs/<new_tag_containing_scheme_b_simd_full_t103_a100>.json
```

`[A]` A naive 12-times multiplication of this contaminated block observation
is `20.94h`; it is not a measured end-to-end result. `[U]` The next work is a
minimum-depth/precision sweep, a clean complete-block benchmark, T=103
multi-block refresh composition, and real multi-GPU sharding.

`[V]` (2026-07-30) The plain depth-8 fork (`LARGE_DIGITS=4`, source SHA-256
`58c257522eefe7a928e6efaa486af6dd3ac53c289fc07878bb41d5d5f9350882`, differing
from the passing depth-16 source only in ten contract-enumerated
identity/depth substitutions) **failed at OpenFHE context construction,
before any encrypted evaluation**: nine towers could not be validly
distributed over four HYBRID large digits. Its fork contract had passed
`7/7`, full local discovery `272/272`, and the remote contract/compile had
passed — the failure is a context-generation rejection, not an accuracy or
runtime result, and produced no evidence JSON.

`[V]` (2026-07-30) An additive `LARGE_DIGITS=3` fork
(`real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3.cpp`, source
SHA-256 `c53e47123614bcddf6a577ca5cc1e22db786d234e953e5c31b6a488e3408c089`,
parent = the depth-8/4-digit source above) passed the digit-distribution
check but OpenFHE auto-selected a ring dimension below the 32,768-slot
capacity the B=8 layout requires, and **also failed at context construction**
("batch size cannot be larger than ring dimension / 2"). No evidence JSON.

`[V]` (2026-07-30) A third additive fork
(`real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536.cpp`,
source SHA-256
`65cc818cbb299af984cccdbe6a7df891cac5b7071f663edfac8c3f13be161365`, parent =
the digits-3 source above) adds an explicit `SetRingDim(65536)`, which the
B=8/32,768-slot layout requires independently of depth. Its local contract,
full local discovery, remote contract, and CUDA target compile all pass. A
detached capacity watcher (tag
`fhe_fides_real_d768_t103_block0_simd_full_t103_depth8_digits3_ring65536_scheme_b_a100_20260730`)
queued at host-load capacity; as of `2026-07-30T12:34:39Z` the gate cleared
(`load1=170.49` vs threshold `250`) and the run launched on GPU 0.

`[V]` (2026-07-30) This run **failed at `2026-07-30T13:04:09Z`, exit status
`2`, this time during actual encrypted evaluation** (not context
construction): the last logged stage is the final causal score tile
`q=12 k=12` (the 13th/last partial token group), immediately followed by
`error: vector::_M_range_check: __n (which is 18446744073709551615) >=
this->size() (which is 9)` — an unsigned-underflow index (`__n` wrapped from
`-1`) into a 9-element vector. `9` matches the depth-8 tower count (the same
count the plain depth-8/4-digit fork's context construction rejected as
undistributable over 4 digits). `[V]` The fork's own additive source has
exactly one `std::vector::at()` call site (`baby_rotations`, loop `small`
runs `1..BSGS_N1-1`, so `small - 1` never underflows) — the crash is **not**
in this fork's own new code. `[U]` It therefore originates inside the linked
FIDESlib/OpenFHE library's internal level/digit/tower bookkeeping, in a code
path exercised specifically by this `depth=8`/`3-digit`/`ring=65536`
combination that the passing depth-16 baseline never exercises; not yet
isolated to a specific library function. This is a library-internal crash,
not an accuracy, runtime, or memory result, and not fixable by editing only
the additive `.cpp` fork. No evidence JSON was produced.
Raw logs preserved and hashed: `run.log`
`sha256=eee7b4c911ce33841acbd15bac8ba5cd6d9a7338cd3c3f9012ad7e34ff96880f`,
`orchestrator.log`
`sha256=26c5a5318e28eb85b1363fef5c7b25de51b36355791a9a6d869e5aba7986ccac`
(both verified byte-identical to the remote host copies). Per the
handoff prompt's evidence rules, any corrected fork must use a new source
filename and run tag — this one must not be reused or reattempted in place.

`[V]` (2026-07-30) Root cause narrowed further by reading the vendored
FIDESlib source (local audit clone, commit `786c7600fb2f16b724e0acf73df367b27b8afed6`,
matching `docker/README.md`'s pin): `Ciphertext::rescale()`
(`src/CKKS/Ciphertext.cpp:435`) computes
`cc.param.ModReduceFactor.at(c0.getLevel())`, where `ModReduceFactor` is
sized `L+1` (9 for `MULT_DEPTH=8`) — an exact match for the crash's
`size() == 9`. `RNSPoly::rescale()` (`RNSPoly.cpp:333-339,363`) decrements
`level` unconditionally (`level -= 1;`) with no lower-bound guard (unlike
`RNSPoly::setLevel()`, which does assert `level >= -1`). So one redundant
rescale drives `level` to `-1`, and the *next* `rescale()` call reads
`getLevel() == -1`, indexing `ModReduceFactor.at(size_t(-1))` — the observed
underflow. This is consistent with the final tile's extra partial-group
mask multiply (masking 7 real lanes out of 8 in the last, 13th token group)
consuming one more level than the 12 full-group tiles need; at `MULT_DEPTH=16`
there was enough headroom to absorb it, at `MULT_DEPTH=8` there is not.
`[U]` Not confirmed against depth-16's own level trace; this is the
most-consistent hypothesis from reading the code, not a step-by-step
instrumented trace of the actual failing run.

`[V]` (2026-07-30) Corrected fork: `real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536.cpp`
(source SHA-256 `738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235`,
parent = the crashed depth8_digits3_ring65536 source above, parent SHA-256
`65cc818cbb299af984cccdbe6a7df891cac5b7071f663edfac8c3f13be161365`). The
fork changes only `MULT_DEPTH` (`8` → `9`, one extra level of headroom) and
identity/log strings — verified by a diff-equivalence contract identical in
method to the prior forks'. It does **not** patch the frozen FIDESlib
library. Local CPU OpenFHE-Python probe (`docker run ... dnagpt-openfhe`,
scratch-only, not committed) confirms `depth=9`/`3`-digit/`ring=65536`
constructs without the OpenFHE-level rejection that killed the plain
depth-8/4-digit and depth-8/3-digit-pre-ring-fix forks. Evidence chain before
touching the capacity gate: local contract `7/7` pass; remote contract `7/7`
pass (identical source hash); remote CUDA target compile succeeds cleanly
(`real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536`, 27s
build). A detached capacity watcher (tag
`fhe_fides_real_d768_t103_block0_simd_full_t103_depth9_digits3_ring65536_scheme_b_a100_20260730`,
watcher PID `1024611`) is now queued as of `2026-07-30T18:20:26Z`, waiting on
host-load/idle-GPU capacity. `[U]` No accuracy, runtime, or memory claim
exists until an immutable `results/runs` JSON lands.

`[V]` (2026-07-30) The gate cleared and this run launched at
`2026-07-30T18:33:27Z` on GPU 0 (preflight noted 4 pre-existing compute
processes already on that physical GPU — `[U]` co-tenant-contaminated
timing). It **made real progress past the depth-8 crash point**: all `91`
causal score tiles (`13` token groups, upper-triangular schedule) completed
successfully, including the final `q=12 k=12` tile that previously crashed
the depth-8 fork. It then **failed differently** at
`2026-07-30T19:19:28Z`, exit status `2` — not the vector-underflow crash
again, but an OpenFHE-level decode error:
`Decode(): The decryption failed because the approximation error is too
high. Check the parameters.` (`ckkspackedencoding.cpp:453`). Reading the
source pins this to `Client::reduce_score_tile`
(`real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536.cpp:526`),
the client-boundary decrypt of the score tile — called immediately after
each tile, so it is specifically the decrypt of the last/deepest tile
(`q=12 k=12`) that fails; the other `90` decrypted fine. `[U]` Read as
depth `9` (10 towers) still leaving too little remaining CKKS precision at
that one worst-case ciphertext for a trustworthy decode — one level short
again, not proof of a distinct bug. No evidence JSON was produced. Raw logs
preserved and hashed: `run.log`
`sha256=d0c54ed03942c4fde1eb9772a0c3423508e657df35cc348d50f16a377bc39f11`,
`orchestrator.log`
`sha256=523535275e30a69748e9f54e139713458dd87e8f9435acd2b30f4dc0fd763da2`
(both verified byte-identical to the remote host copies).

`[V]` (2026-07-30) Fork 5: `real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536.cpp`
(source SHA-256 `703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2`,
parent = the depth9 fork above, parent SHA-256
`738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235`). Changes
only `MULT_DEPTH` (`9` → `10`, one more level, targeting the last score
tile's decode-precision shortfall) and identity/log strings — same
diff-equivalence contract method as prior forks; no FIDESlib patch. Local
CPU OpenFHE-Python probe confirms `depth=10`/`3`-digit/`ring=65536`
constructs. Evidence chain before touching the capacity gate: local
contract `7/7`; remote contract `7/7` (identical source hash); remote CUDA
compile clean (27s). Queued and **launched immediately**
(`2026-07-30T19:59:15Z`, tag
`fhe_fides_real_d768_t103_block0_simd_full_t103_depth10_digits3_ring65536_scheme_b_a100_20260730`,
watcher PID `1453974`) — preflight was fully clean this time
(`compute_processes=0`), unlike the contaminated depth-9 launch. `[U]` No
accuracy, runtime, or memory claim exists until an immutable
`results/runs` JSON lands.

`[V]` (2026-07-30) This run made further real progress, then **failed a
third, distinct way**: it got past all `91` score tiles (as depth-9 did)
*and* past the previous decode-precision failure at the last tile, all the
way through attention projection, residual, and the LN2 client boundary
(`[boundary] ln2 logical=8`). It then failed at
`2026-07-30T20:20:30Z`, exit status `2`, with a clean, self-diagnosing error
from the fork's own code (not a library crash):
`error: insufficient depth before Token-SIMD MLP group 0`. This comes from
a pre-existing guard `require_remaining_depth(normalized2->GetLevel(), 4,
"Token-SIMD MLP group 0")`
(`real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536.cpp:811-813`,
unchanged across all forks, part of the frozen schedule), which throws
whenever `normalized2`'s current CKKS level leaves fewer than `4` levels of
remaining budget under `MULT_DEPTH=10`. `[U]` The exact consumed level at
that point was not printed (only QKV-stage levels are logged, at `level=3`
— confirmed empirically in this run's log — LN2's output level is not
logged anywhere in the current source). Hand-deriving it from the schedule
is unreliable: the client-boundary "refresh" pattern (decrypt, plaintext
compute, fresh re-encrypt) does not appear to zero the level of every
downstream operand — only one branch of each subsequent multiply is fresh,
and there is no explicit level-alignment call in the source, so the actual
behavior depends on how FIDESlib's `EvalMult` handles mismatched operand
levels internally. Rather than continue blindly incrementing `MULT_DEPTH`
(costly at ~20–45 min per cycle), the next step is a diagnostic-only fork
that prints `normalized2->GetLevel()` before this check, to read the exact
number directly instead of guessing. No evidence JSON was produced. Raw
logs preserved and hashed: `run.log`
`sha256=0ab3e863cf22f0723107e984981f9fa48854da436143171911b5207d9364dcf9`,
`orchestrator.log`
`sha256=2fe76537ca9d56f870beb26704e4b010ca2348d0f9d6cd2a0fd5e7f91169fecf`
(both verified byte-identical to the remote host copies).

`[V]` (2026-07-30) Diagnostic-only fork:
`real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.cpp`
(source SHA-256 `258e0da1ece28ef8f42652097ed8c7b61218b2a76c2d297f879a860410c72e82`,
parent = the depth10 fork above, parent SHA-256
`703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2`). `[V]`
Changes **only** identity/log strings plus one added line —
`std::cout << "[diag] ln2 group=" << group << " level=" << normalized2->GetLevel() << '\n';`
— immediately before the existing `require_remaining_depth` guard.
`MULT_DEPTH` stays `10`, unchanged; no crypto parameter or schedule
operation is touched; no FIDESlib patch. Diff-equivalence contract `7/7`
local, `7/7` remote (identical source hash), remote CUDA compile clean
(32s). Queued (tag
`fhe_fides_real_d768_t103_block0_simd_full_t103_depth10_diag_digits3_ring65536_scheme_b_a100_20260730`,
watcher PID `1772313`) at `2026-07-30T20:48:23Z`. `[U]` This run is expected
to fail identically to the depth-10 parent (same guard, same `MULT_DEPTH`)
— its only purpose is to reveal the exact `normalized2` level via the added
print, so the next real fork can pick the correct `MULT_DEPTH` in one step
instead of continuing +1 increments.

`[V]` (2026-07-31) Launched `2026-07-30T20:59:24Z` (mildly contaminated,
1 pre-existing compute process at preflight), failed exactly as expected at
`2026-07-30T21:22:39Z` with the same
`insufficient depth before Token-SIMD MLP group 0` guard. The added line
printed `[diag] ln2 group=0 level=9` immediately before the throw. `[V]`
Since `require_remaining_depth` needs `4 <= MULT_DEPTH - current_level`,
**the minimum passing `MULT_DEPTH` for this graph is `9 + 4 = 13`** — this
answers the handoff prompt's main question 1 (pending confirmation by an
actual passing run). Raw logs preserved and hashed: `run.log`
`sha256=41d84f7a8fe535cc293413e419a9ca41b77dba7d188a1421027b46bed8c6d845`,
`orchestrator.log`
`sha256=f74135c16a2cf867266c306f5cb3a59ee62ee15460e7b7e6166524649fdc6986`
(both verified byte-identical to the remote host copies). Local CPU OpenFHE
probe confirms `depth=13`/`3`-digit/`ring=65536` constructs without
rejection — proceeding directly to a depth-13 fork rather than any further
intermediate increment.

`[V]` (2026-07-31) Fork 7:
`real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp`
(source SHA-256 `6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e`,
parent = the depth-10 fork, parent SHA-256
`703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2`). Changes
only `MULT_DEPTH` (`10` → `13`, per the diagnostic finding above) and
identity/log strings; no FIDESlib patch. Local contract `7/7`; remote
contract `7/7` (identical source hash); remote CUDA compile clean (27s).
Queued and launched immediately (tag
`fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_scheme_b_a100_20260731`,
watcher PID `3511176`) at `2026-07-31T07:55:49Z`. `[U]` No accuracy,
runtime, or memory claim exists until an immutable `results/runs` JSON
lands.

`[V]` (2026-07-31) **This run passed — the first passing depth-8-family
result.** Launched `2026-07-31T08:09:51Z` on physical GPU 4 (not GPU 0;
preflight showed 4 pre-existing compute processes — `[U]`
co-tenant-contaminated). It ran far longer than any prior fork before
producing stdout (no `[stage]` lines visible for over an hour of checks);
`docker top` confirmed the process was actively burning CPU throughout
(rising from ~77 to ~110 core-equivalents), and this turned out to be
ordinary fully-buffered stdio rather than a hang — the ~205-line run.log
was written all at once near completion, and the process's own
`context_keygen_load` timing field (`4.15s`) proves keygen itself was fast;
the apparent "silence" was simply genuine, slow, contaminated compute
(`server_linear_algebra_seconds=4448.87s`) with nothing to flush.

Completed (`.done=0`) at `2026-07-31T09:27:41Z` with
`REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_GATE_PASS`. `[V]`
`global_rel_inf=4.35054978665743e-9`,
`worst_token_rel_inf=4.35054978665743e-9`,
`max_abs_error=6.975483973770125e-8` — tighter than the depth-16 baseline's
`4.777e-9`, both far inside the unchanged `4e-2` block gate. `[V]`
`packed_output_level=6`, identical to the depth-16 baseline — confirms the
final output level is graph-invariant regardless of the depth budget,
exactly as the handoff prompt's caveat warned ("evidence of headroom, not
proof"): the deepest *mid-graph* point (LN2 output, level `9`) needed far
more budget than the final level alone would suggest. `[V]` Encrypted
evaluation: `4662.221675408s` total (`4448.868146699s` server +
`213.353528709s` client) versus the depth-16 baseline's `6281.748445232s`
(`6140.143438612s` server + `141.605006620s` client) — about `25.8%`
faster overall, `27.5%` faster server-only. `[U]` **Directional only**: both
runs were co-tenant-contaminated single samples; `docs/roadmap.md` requires
two-to-three repeats for a reported stable speedup, which this is not.

Peak PID-specific VRAM: `9834 MiB` (below the depth-16 baseline's
`12920 MiB`, consistent with 14 towers versus 17 — `884` five-second
samples, `2026-07-31T08:09:56Z`–`2026-07-31T09:27:41Z`).

Immutable evidence:

- correctness:
  `results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_scheme_b_a100_20260731.json`
  (`sha256=fe1016b5f6c690c04e794e3db8dfe86b69109c8d90dca68229849714784b11cb`);
- process-memory telemetry:
  `results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_scheme_b_vram_a100_20260731.json`
  (`sha256=c865500297124b690d70cda3d95e550004b2e67ec8694a92d3fb6b76251b5e73`);
- raw remote run log: `sha256=4e1da4ccae24d098fb3a62aa3b1527c1ad0963e279ea4ba13301aad154380fb1`;
- raw remote orchestrator log: `sha256=c2a7fdeb02b65181db5cb6981dae332615793ffeeeed653b9cfeab57846b820f`;
- raw remote VRAM log: `sha256=8a1e6c1e7fae2f40e09f9ab93fc2d5b1fa0d77ae84eb350166cea8396c4af279`.

All four artifacts were pulled from the remote host and verified
byte-identical (matching remote `sha256sum`) before being written into the
repo. Manifest rows appended to `results/hybrid/manifest.yaml`
(correctness + linked VRAM).

**This answers the handoff prompt's main question 1**: the minimum passing
CKKS modulus chain for the complete B=8 T=103 block at unchanged security
is `MULT_DEPTH=13` (3 HYBRID digits, ring `65536`, scale `50`). `[U]`
Main questions 2–7 (uncontended-A100 speed/memory delta, further
error-budget tradeoffs, B=4 comparison, multi-GPU sharding, multi-block
composition, and the final `<=3h` verdict) remain open.

### B=4 vs B=8 Token-SIMD: closed by local operation-count evidence, no GPU time spent (2026-07-31)

`[V]` Added `fhe/gpu_real_scheme_b/test_simd_layout_b4.py`, mirroring
`test_simd_layout.py`'s B=8 contract suite at `batch_width=4` against the same
real T=103 fixture. No new implementation was needed: `simd_layout.py` was
already generic over `batch_width`. 20/20 tests pass, including every
real-oracle-matched case (Q/K/V, attention projection, all 4 MLP FC chunks,
all 4 MLP projection chunks, score tiles, causal masking, tiled softmax) at
the same `atol=1e-9`/`1e-12` bands as B=8. `[V]` B=4 also exactly saturates a
`ring_dim=32768` context (`4*1024*4=16384=32768/2` complex slots), mirroring
B=8's exact saturation of `ring_dim=65536`; this confirms the fixed-four-copy
comparison precisely rather than approximately.

`[V]` The honest full operation-count comparison (via `simd_layout.py`'s own
`protocol_counts`/`server_operation_counts`, the same formulas whose B=8
output already matches real T=32/T=103 GPU evidence) is decisive and goes
beyond the earlier dense-product-only projection (`7.92x` vs `3.96x`).
Counting every operation category at T=103, one block:

| metric | B=8 | B=4 | B=8 advantage |
|---|---|---|---|
| token groups | 13 | 26 | — |
| dense matrix products | 156 | 312 | 2.00x |
| ciphertext-ciphertext multiplications | 1,506 | 2,910 | 1.93x |
| ciphertext-plaintext multiplications | 177,734 | 353,832 | 1.99x |
| explicit rotations | 8,173 | 15,934 | 1.95x |
| full-block client round trips | 857 | 1,832 | 2.14x |
| score-tile decryptions alone | 91 | 351 | 3.86x |

B=4 is worse on *every* measured axis, not only the dense-product count. The
attention path's tile-pair count grows roughly with `G*(G+1)/2` (G=token
groups), so doubling G from 13 to 26 nearly quadruples score-tile round
trips, the single most expensive-to-avoid cost (each is a real client
decrypt/exact-softmax/re-encrypt boundary).

`[V]` **Go/no-go: NO-GO for a complete B=4 block, decided from local evidence
alone.** Per `docs/hybrid/roadmap.md` phase 2, a matched GPU micro-gate is
normally required before this decision, but here every operation count --
not just the previously-known dense-product ratio -- points the same
direction using the identical BSGS/COPIES machinery already validated
real-GPU-accurate for B=8, so there is no structural mechanism by which a
GPU timing run would reverse it (same pack_width, same COPIES=4, same
per-operation cost model, strictly more of every operation at B=4). Spending
GPU-hours on a B=4 packed/serial micro-gate is not justified given this
signal. `[U]` This is a local-evidence-only conclusion, not a GPU
measurement; if a future need arises to double-check it (e.g. if per-operation
cost at ring=32768 turns out non-uniformly cheaper than at ring=65536 for
some FIDESlib-internal reason), the micro-gate remains cheap to run and
would settle it definitively. No `results/runs/` entry exists for this
finding since it produced no GPU/timing evidence. The executable local evidence
is `simd_layout.py` plus `test_simd_layout_b4.py`.

**Practical implication at the time of this experiment**: B=8 remains the
right width under the fixed four-copy layout. This comparison does not close
the later design space in which B=16 uses two copies or B=32 uses one copy;
those alternatives require a different downstream packing schedule and are
tracked in the current roadmap.

### Depth-13 clean-timing repeats: queued via one-shot host cron (2026-07-31)

`[V]` Preflight at session start found every physical GPU on the shared host
already carrying a resident compute process (`nvidia-smi --query-compute-apps`
showed one PID per GPU, indices 0-7; GPUs 4/5/6/7 additionally showed
`memory.used<10000MiB`/`util<10%`, i.e. they would read as "idle" by
`gpu_common/capacity_lib.sh`'s own thresholds despite the resident process).
No genuinely clean (zero-compute-process) GPU existed at launch time,
consistent with every prior timing run in this project's history. Proceeding
anyway per `docs/roadmap.md`'s own guidance to label honestly rather than
wait indefinitely for a machine state that has never yet occurred.

`[V]` **Operational finding, reusable going forward**: a plain
`nohup ... & disown` launch of `wait_and_run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh`
issued via `brev exec` did not reliably detach -- the invoking shell call
never returned even after the polling/launch steps it wraps should have
completed in seconds, consistent with the shared host's known SSH-connection
flakiness (repeated "Connection failed, checking instance status..."
reconnect messages were observed on unrelated `brev exec` calls in the same
session) rather than any fault in the wrapped script. Switched to the
pattern already proven in this project's own history (see the 2026-07-27
context/key-caching entry above: "driven by a one-shot host `cron` entry so
the launch survives local session/SSH loss"): appended a one-shot
`crontab` line (`MM HH DD MM *`, computed from the host's own `date -u -d
"+N minutes"`, never edited/removed the pre-existing stale one-shot entry
from 2026-07-28) that runs
`SCHEME_B_SIMD_FULL_DEPTH13_DIGITS3_RING65536_RUN_TAG=<tag> /bin/bash
wait_and_run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh` with
output redirected to a per-tag `.cron_launch.log`. Confirmed via the
orchestrator log that the resulting process is cron-parented, not tied to
any interactive session. Two repeats queued this way:

- `fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_scheme_b_a100_20260731_rep2`
  -- confirmed launched and running (`run_pid=4084860`).
- `fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_scheme_b_a100_20260731_rep3`
  -- queued 4 minutes after rep2 so `gpucap_pick_idle_gpu` would see rep2's
  GPU already occupied and select a different physical GPU rather than
  racing for the same one.

`[U]` Both repeats are expected to be co-tenant-contaminated like every
prior run on this host (no clean GPU existed at queue time); they are being
banked as directional samples toward the `docs/roadmap.md`-required
2-3-repeat count for a *stable* depth-13 speedup claim, not as clean
benchmarks. Results pending; each run takes on the order of the original
fork-7 timing (~78 minutes encrypted evaluation) plus queueing wait.

### Depth-13 clean-timing repeats: both landed, the 25.8% speedup does NOT survive repetition (2026-07-31)

`[V]` **Both repeats PASSED correctness** and are dramatically SLOWER than
the original sample, not faster:

| sample | `encrypted_evaluation_seconds` | vs original (`4662.22s`) | vs depth-16 baseline (`6281.75s`) | `global_rel_inf` |
|---|---|---|---|---|
| original (2026-07-31, fork 7) | `4662.221675408` | 1.00x | `0.742x` (faster) | `4.3505e-9` |
| rep2 | `13658.267331302` | `2.930x` slower | `2.174x` slower | `4.3890e-9` |
| rep3 | `13202.354867729` | `2.832x` slower | `2.102x` slower | `4.9596e-9` |

Evidence: `results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_scheme_b_a100_20260731_{rep2,rep3}.json`
(`sha256=977a9da5...916111` / `sha256=9890d50d...916111`, both pulled and
verified byte-identical to the remote host copies before writing), paired
VRAM telemetry `..._vram_a100_20260731_{rep2,rep3}.json`. All four artifacts
were fetched with `brev copy` rather than `brev exec ... cat` for the two
large (~7.3MB, ~82k-line) `.vram.log` files -- `brev exec`'s `cat` pipe
stalled/timed out on files this size in this session, while `brev copy`
(a dedicated transfer, not a piped `cat`) completed each in under 2s. Worth
remembering: use `brev copy` for any file pull beyond a few hundred KB,
not `brev exec ... cat`.

`[V]` **This converts the previous `[U]` directional "25.8% faster" claim
into an explicit non-claim, per `docs/roadmap.md`'s own two-to-three-repeat
requirement for a *stable* speedup.** Not merely "not yet confirmed" --
actively contradicted: both repeats are ~2.1-2.9x SLOWER than both the
original depth-13 sample and the depth-16 baseline it was being compared
against. The honest reading of all three samples together is that
encrypted-evaluation wall time for this workload swings over roughly a
`2.83x` range (`4662s` to `13658s`) purely as a function of ambient host
contention, with no depth-13-vs-depth-16 speed advantage demonstrated by
this evidence. `docs/roadmap.md`'s "Required deliverable #1" (a clean or
honestly-labeled repeat count converting the directional figure into a
stable claim or an honest non-claim) is satisfied by the non-claim above,
not by a confirmed 25.8%.

`[V]` **Root-caused, not merely observed: the slowdown correlates with
host-wide CPU contention, not GPU-memory contention.** Per-GPU telemetry
for both repeats:

| sample | target-PID peak (MiB) | whole-GPU peak (MiB) | cotenant peak (MiB) | preflight compute_processes | host load average during run |
|---|---|---|---|---|---|
| original | `9834` | `76757` | `65868` | `4` | not recorded |
| rep2 | `9834` | `34052` | `18700` | `1` | `500`-`1042` (live-observed) |
| rep3 | `9834` | `30380` | `18700` | `1` | `500`-`1042` (live-observed) |

Both repeats had *lighter* GPU-memory contamination than the original
pass (lower whole-GPU peak, lower cotenant peak, fewer preflight
compute processes, identical target-PID peak of `9834` MiB confirming
memory footprint is graph/depth-determined, not contention-determined)
yet ran `2.1`-`2.9x` slower. Host load average of `500`-`1042` (this
project's worst observed, exceeding even the 2026-07-28 warmup run's
`800`-`1218` peak) on the shared 255-core host is the better-correlated
explanation: this workload's CPU-side work (diagonal/plaintext encoding,
per the 2026-07-27 profiling finding that ct-plaintext multiply-and-
encode dominates a matmul call's cost) competes for cores against
whatever else is running, independent of which physical GPU or how much
GPU memory is free. `[U]` Correlation, not a controlled experiment --
no repeat has run on a host with load below `250` (the project's own
capacity-gate threshold), so a genuinely clean sample still does not
exist for either depth.

`[A]` **Revised planning implication**: a defensible "clean" depth-13 (or
depth-16) timing sample requires the host to actually clear to low load,
not just an individual GPU reading low memory/utilization -- the existing
capacity gate's per-GPU idle check is insufficient by itself to predict
clean timing on this specific shared host, since CPU contention can be
severe while GPU memory is nearly idle. Future timing repeats should
additionally check `uptime`'s load average against a much stricter bar
(e.g. `<50`, not the `250` capacity-launch threshold) before trusting a
result as clean, or accept and label every sample as contaminated like
this entry does.

### Three parallel tracks toward speed and the T=103 end-to-end estimate (2026-07-31)

Following the CPU-contention finding above, three independent tracks were
built in parallel (each additive, none editing another's files, none
editing this doc/roadmap.md/results/hybrid/manifest.yaml directly --
folded in here after the fact):

**Track 1 -- CPU-side diagonal-plaintext cache retry.** New source
`fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp`
(forked from the passing depth-13 Token-SIMD source), caching only the
CPU-side `packed_values` vector per `(call-type, giant, small)` key across
the 13 token groups (a fresh `Plaintext`/GPU object is still constructed
every call) -- deliberately avoiding the exact mechanism that killed the
original 2026-07-27 diagcache attempt (reusing a GPU-resident `Plaintext`
across `multPt` calls). Local NumPy contract
(`test_diagonal_cache_reuse.py`) proves the cached diagonal is
byte-identical across all 13 groups for the real fixture before any C++.
Local + remote static contract passes, compiled cleanly under the pinned
FIDESlib commit. Queued via one-shot host cron (tag
`fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_scheme_b_a100_20260731`)
and launched on GPU 0 at 2026-07-31T18:58Z. `[U]` Result pending as of this
writing.

**Track 2 -- 2-GPU process-per-GPU sharding, Stage 1 (Q/K/V split).** New
Token-SIMD-parameter-matched writer fork
(`real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13.cpp`, matching
the depth-13 source's `SLOTS=32768`/`MULT_DEPTH=13`/rotation-key set --
the existing writer/reader pair was for the incompatible T=2/4096-slot
layout, per the design-sketch entry above) and a sharded reader
(`..._simd_shard_reader_t103_depth13.cpp`, `--part query,key,value`
flag). 40/40 local static contract pass
(`test_shard_writer_reader_t103_depth13_contract.py`); the pre-existing
math contract (`test_shard_layout.py`, 7/7) needed no changes. Both
binaries compiled cleanly remotely. Queued via a combined orchestrator
(`wait_and_run_scheme_b_simd_shard_qkv_t103_depth13.sh`, tags
`fhe_fides_real_d768_t103_qkvshard_{writer,reader_querykey,reader_value}_scheme_b_a100_20260731`);
writer completed at 2026-07-31T~19:47Z, both shard-reader workers
launching as of this writing. `[U]` Results pending.

**Track 3 -- 12-block + task-head T=103 driver.** Two real findings
during fixture generation: (1) `fhe/multiblock/export_fixture.py` is
hard-coded to reject any `--tokens` value other than 2
(`ValueError: intentionally specialized to T=2`) -- a new sibling script,
`fhe/multiblock/export_fixture_t103.py`, reuses its T-general helpers
additively; (2) at T=103 the manual float64 reimplementation diverges from
the released model's float32 upstream forward pass past the frozen `2e-5`
gate by block 6 (up to `~8e-4`) -- diagnosed as float32 rounding
accumulation over 103-token attention (not a bug) by cross-checking
against a float64-cast shadow copy of the same released weights, which
matched to `~5e-7`-`5e-6` at every block; the exporter now gates against
that shadow with both readings recorded honestly. Fixture:
`checkpoints/fhe_exports/gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0/`
(392 arrays, `is_full_prompt=true` since T=103 is GSR's complete prompt
unlike T=2's truncated graph gate, margin `12.27`, label N).

Local contract `test_all_blocks_head_t103_simd_contract.py`: initially
13/13 passed in isolation but one assertion
(`test_all_twelve_blocks_recompute_through_the_refresh_lineage`) used
`atol=1e-12` for a value chained through 12 independently-recomputed
blocks -- too tight for float64 accumulation noise between two
independent float64 forward-pass implementations (observed
`~1.7e-12` absolute, `~3e-14` relative at magnitudes `~50`-`57`, i.e.
ordinary floating-point noise, not a bug). Fixed to `atol=1e-9`, matching
every other real-fixture multi-step comparison in this codebase (e.g.
`test_simd_layout.py`). 13/13 pass after the fix.

New C++ driver
`real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head.cpp` composes the
depth-13 per-block logic 12x with a Token-SIMD-adapted refresh boundary
(13-ciphertext decrypt/unpack/re-pack/re-encrypt at level 0 between every
pair of blocks, generalizing `two_block_refresh.cpp`'s single-ciphertext
T=2 mechanism) and a new final-head evaluator. 19/19 local static contract
pass; compiled cleanly remotely; `--help` and a fail-closed wrong-hash
rejection both ran successfully. **The actual 12-block run was
deliberately NOT launched** -- per the explicit stopping point in this
track's scope, a 15-45 hour single GPU job is a real resource commitment
needing an explicit go, not an autonomous one.

A stricter quiet-window gate was written for this specific job
(`fhe/gpu_real_scheme_b/gpucap_strict_quiet_window_12blocks_head.sh`, does
not edit the shared `gpu_common/capacity_lib.sh` used by every other
gate): requires host load average `<50` (vs the default `250`) **and**
zero compute processes across **all 8 GPUs** (vs. the default checking one
device's memory/util thresholds), both sustained for 5 consecutive polls,
failing closed on any `nvidia-smi` glitch given the size of the
commitment. Live-validated against the actual host: correctly reports "not
quiet" at load average `~1020` with 21 active compute processes across
`>=6` physical GPUs, a condition the existing lenient gate would have
called launch-ready (it would have picked GPU 1 as "idle").

**Revised 12-block time estimate**: 12 x the two known single-block
samples (`4662s` best case, `13202`-`13658s` under heavy CPU contention)
gives `15.5`-`45.5` hours of block compute; refresh/head overhead adds
well under half an hour. Since CPU contention (not GPU memory) was the
documented driver of that `2.1`-`2.9x` spread, a genuinely quiet start
should push the real result toward `16`-`20` hours -- but the strict gate
only guarantees a clean *start*, not sustained quiet for `15`-`45` hours
straight, so the wide range is real risk, not just caution. `[A]` Central
estimate: `16`-`24` hours if the gate's bar holds through the run,
degrading toward `45+` if contention returns mid-run. `[U]` The wait to
even reach the strict gate's bar and start is itself unknown given today's
host behavior (load swinging `22` to `1123` within single-digit minutes).

**Mechanical note**: three concurrent agents editing the same shared
`CMakeLists.txt`/`build_in_fideslib.sh` (each additive, none touching the
others' targets) produced one small, real drift: an intermediate manual
sync of `build_in_fideslib.sh` from a remote snapshot (to reconcile
Track 1/Track 2's simultaneous edits) was taken *before* Track 1's own
`cpudiagcache` build-target line had landed on that file remotely, so the
synced copy silently dropped it (the target itself remained correctly
defined in `CMakeLists.txt` throughout -- only the convenience
build-everything script's reference to it was briefly missing). Caught by
running the full local test suite (`test_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_contract.py`'s
`test_cmake_and_build_script_reference_new_binary`) after all three tracks
reported back; fixed by re-adding the one missing `cmake --build`/`echo`
line pair to both the local and remote copies. A second, unrelated bug was
also caught the same way: the new 12-block contract's
`test_all_twelve_blocks_recompute_through_the_refresh_lineage` used
`atol=1e-12` for a value chained through 12 independently-recomputed
blocks -- too tight for ordinary float64 accumulation noise between two
independent implementations (observed `~1.7e-12` absolute at magnitudes
`~50`-`57`); fixed to `atol=1e-9`, matching every other real-fixture
multi-step comparison in this codebase. All 442 local tests pass after
both fixes.

### Tracks 1 and 2 landed: CPU-side diagonal cache passes, real 2-GPU concurrency proven (2026-07-31/08-01)

**Track 1 result.** `fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_scheme_b_a100_20260731.json`
(`sha256=436754d2...b20b19`, pulled via `brev copy` and verified
byte-identical to remote -- `brev exec ... cat` timed out again on this
run's evidence pull, same pitfall as the rep2/rep3 vram logs; `brev copy`
is now the standing rule for any file over a few hundred KB *and* any file
pull that seems to hang). **PASS**: `global_rel_inf=4.4972e-9`, same band
as every depth-13 sample. Cache hit rate `159732/159744=99.99%`,
confirming the diagonal really was being redundantly recomputed 13x as
hypothesized. Target-PID peak memory `9834 MiB`, identical to every
uncached depth-13 sample -- the CPU-side-only cache adds no GPU memory
cost, as designed. `encrypted_evaluation=7324.91s`
(`7201.57s` server + `123.34s` client), landing between the best uncached
sample (`4662.22s`) and the two heavily-contaminated repeats
(`13202`-`13658s`). `[U]` This run also carries the heaviest whole-GPU
memory contamination of any depth-13 sample this session (`80037 MiB`
peak, `>20` co-tenant PIDs, one alone at `70186 MiB`) -- consistent with
real cache benefit given that severity, but not a clean, matched-conditions
speed claim. The cache mechanism itself (hit rate, unchanged memory) is
decisive, uncontaminated evidence independent of the wall-clock question;
1-2 repeats are needed before calling any speedup stable. VRAM evidence:
`..._vram_a100_20260731.json`.

**Track 2 result -- the first measured 2-GPU concurrency benefit in this
project.** Writer (`fhe_fides_real_d768_t103_qkvshard_writer_scheme_b_a100_20260731.json`,
`sha256=4ede27be...d07d8fd0`) built the Token-SIMD-parameter-matched
context/keys in `3.26s` and serialized them. Two shard-reader workers then
launched **concurrently on two distinct, genuinely clean physical GPUs**
(GPU 4 and GPU 5, both `preflight mem=0MiB util=0% compute_processes=0` --
the first fully clean preflight of this entire session on either GPU):
worker A (`query,key`,
`fhe_fides_real_d768_t103_qkvshard_reader_querykey_scheme_b_a100_20260731.json`,
`sha256=cf6fc18f...64fa422d5`) computed 26 matrix products in `915.09s`;
worker B (`value`,
`..._reader_value_scheme_b_a100_20260731.json`,
`sha256=a1a9b2fa...d6425516`) computed 13 matrix products in `591.54s`.
Both **PASS** against the real T=103 oracle at the unchanged `~1e-9` band,
deserializing the writer's state with zero
`GenCryptoContext`/`KeyGen`/`EvalMultKeyGen`/`EvalRotateKeyGen` calls of
their own. Secret-key hygiene preserved (state directory removed after
both readers completed).

**The concurrency arithmetic**: wall-clock for both workers to complete is
`max(915.09, 591.54) = 915.09s`, versus `915.09 + 591.54 = 1506.63s` if
the same 3 parts had been computed serially on one GPU -- an observed
`~1.65x` speedup from splitting Q/K/V across 2 physical GPUs. `[U]` One
sample, an uneven split (2 parts vs 1), and the two workers' per-part cost
differs (`35.2s`/product for A vs `45.5s`/product for B) in a way not yet
separated from ambient per-GPU contention differences at that moment --
not yet a controlled, repeated measurement, but the core infra claim (two
independent FIDESlib/CUDA processes sharing one deserialized context/key
lineage, running concurrently on two physical GPUs, both correct) is now
directly demonstrated, not just designed.

**Both landing together also resolved the outstanding shared-file drift**:
after both tracks' reports, the local test suite surfaced the
`cpudiagcache` build-script gap described above -- fixed on both local and
remote copies, all 442 tests pass.

### Diagcache rep2: fastest of all samples, but does NOT cleanly isolate cache benefit (2026-08-01)

`[V]` **A repeat of the Track-1 CPU-side diagcache run (identical binary)
PASSED and is the fastest of all completed depth-13 T=103 samples**, but
the measurement is confounded and does not isolate the cache from ambient
host contention. Evidence:
`results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_scheme_b_a100_20260801_rep2.json`
(`sha256=6407e132...d272fb`). Identical cache behavior to the original
cached run (`159732` hits / `12` misses = `99.99%`), so the two cached runs
are byte-for-byte the same computation.

All five completed depth-13 T=103 samples (a sixth, a cached rep1, was
killed after launching into a load-961 spike):

| sample | `encrypted_evaluation_seconds` | `server_linear_algebra_seconds` | cache | contamination / host load1 | `global_rel_inf` |
|---|---|---|---|---|---|
| **rep2 (cached, NEW 2026-08-01)** | `2986.26` | `2871.67` | yes (159732 hits) | **light**: whole-GPU `30241` MiB, load `~150`-`240` | `4.099e-9` |
| original cached (2026-07-31) | `7324.91` | `7201.57` | yes (159732 hits) | heavy: whole-GPU `80037` MiB, `>20` cotenant, load `500`+ | `4.497e-9` |
| uncached best / fork 7 (2026-07-31) | `4662.22` | `4448.87` | no | moderate (whole-GPU `76757` MiB); host load not recorded | `4.351e-9` |
| uncached rep2 (2026-07-31) | `13658.27` | `13523.84` | no | severe: load `500`-`1042` | `4.389e-9` |
| uncached rep3 (2026-07-31) | `13202.35` | `13060.99` | no | severe: load `500`-`1042` | `4.960e-9` |

`[V]` **Light contamination, live-observed load trajectory.** GPU 4, clean
preflight (`mem=0MiB util=0%`), whole-GPU peaked at only `30241` MiB --
essentially just this job, no 80GB co-tenant. Host load1 trajectory as
observed by the monitor: gate launched at load1 `~6.2` (`07:41Z`, after 5
sustained sub-50 polls), then `155.16` at `07:46Z`, `240.79` at `08:06Z`,
`159.63` at `08:26Z`; run ended `~08:31Z`. So rep2 ran mostly under load
`~150`-`240` -- much lighter than the original cached run's sustained
`500`+.

`[U-leaning-positive]` **Verdict.** rep2 is the fastest of all samples at
the lightest contamination, but it does NOT cleanly isolate cache benefit:
(1) both cached runs have identical cache behavior (`159732` hits), so the
`7324`->`2986s` gap between the two cached runs is PURE contamination, not
cache; (2) rep2 (cached) is faster than the best uncached (`4662s`) but ran
under lighter contamination, so still not a matched-conditions comparison.
A clean cache-vs-nocache isolation needs both to run under identical light
load, which this shared host will not reliably provide (this session: quiet
windows collapse from load `~6` back to `150`+ within `~5` min; rep1 was
killed after launching into a load-961 spike). `[V]` Correctness is
unaffected: PASS at `4.099e-9`, same band as every depth-13 sample. This
leaves open-question #1 (does the diagcache give a real repeatable speedup?)
in the same `[U]` state as after Track 1: the cache mechanism (hit rate,
unchanged GPU memory) is decisive, but the wall-clock speedup cannot be
separated from host contention on this shared host.

### 2-GPU process-per-GPU sharding: design sketch, not yet built (2026-07-31)

Per `docs/hybrid/roadmap.md` phase 3, FIDESlib's native multi-device path is
closed (SIGSEGV in `SetupConstants`/`ContextData`,
`fhe_fides_gpu_multiblock_multigpu_2gpu_sigsegv_setupconstants_FAIL_20260725`).
The cross-process context/key serialization writer/reader pair
(`fhe_fides_real_d768_t2_full_serialized_writer/reader_scheme_b_a100_20260727`)
already proves one process can deserialize another's `CryptoContext`/keys and
evaluate the full graph correctly, but only sequentially, one reader at a
time -- not yet two processes computing concurrently on two physical GPUs
from one shared lineage.

Proposed staged plan, ordered by merge complexity (simplest infra proof
first, per `docs/roadmap.md`'s "test one variable at a time"):

1. **Stage 1 -- Q/K/V split, zero-merge infra proof.** Extend the writer to
   serialize state once; two independent reader processes, each pinned to
   its own physical GPU (`--gpu 0` / `--gpu 1` on the existing binaries'
   flag), each deserialize the *same* context/key state and independently
   compute one disjoint subset of {Q, K, V} for one token group (e.g. GPU A:
   Q+K, GPU B: V). Nothing is added between the two ciphertexts -- Q, K, V
   are only ever consumed together downstream (attention), never summed --
   so this stage tests purely whether two concurrent FIDESlib/CUDA processes
   sharing one deserialized context on two physical GPUs produce results
   bit-identical to the existing sequential one-GPU computation, with no
   ciphertext-arithmetic-correctness question at all. Go/no-go: matching
   `global_rel_inf` against a same-source 1-GPU control, plus wall-clock
   and per-process VRAM for both processes.
2. **Stage 2 -- MLP chunk split, exact-merge proof.** The four MLP FC and
   four MLP projection chunks (`chunk in range(COPIES)` in the existing
   schedule) are independent dense transforms; the four projection-chunk
   results are summed only via ciphertext `EvalAdd` (native CKKS addition,
   exact, no approximation) after all four are computed, matching
   `simd_layout.py`'s `context_from_weight_tiles`-style accumulation
   pattern already proven for Token-SIMD. Split 2-and-2 across the two
   GPUs from stage 1's infra; the merge step deserializes one process's
   partial-sum ciphertext into the other's context and does one `EvalAdd`.
   Contract-test locally first (NumPy-level: does splitting the existing
   `bsgs_matmul_tokens` chunk loop 2-and-2 and summing partial results match
   the single-process sum, trivially true in plaintext but establishes the
   split points before any C++) before any GPU work.
3. **Stage 3 -- independent attention token/query groups.** Per
   `docs/roadmap.md`, the harder case: different token groups' causal score
   tiles are independent of each other (a `(query_group, key_group)` tile
   only depends on those two groups' Q/K, per `active_weight_shifts`), so
   groups could be sharded by `(query_group mod N_GPUs)`. Needs the
   `context_from_weight_tiles` accumulation (currently per-query-group,
   summed across key groups) checked for cross-GPU accumulation order
   sensitivity before any implementation -- CKKS addition is exact and
   order-independent in exact arithmetic, but only after confirming no
   query group's key-group loop is split *across* the sharding boundary in
   a way that would need a partial ciphertext to cross GPUs before its own
   group finishes.

`[U]` None of these three stages has been implemented or run. This is a
design sketch to unblock a scoped first micro-gate, not evidence. Per
`docs/roadmap.md`'s "no multi-GPU work until single-GPU throughput is
measured" (already satisfied for B=8) and "start with a two-GPU micro-gate
with a same-source one-GPU control," stage 1 (Q/K/V split) is the
recommended next concrete step: it isolates the infra question (can two
FIDESlib/CUDA processes share one deserialized lineage and run
concurrently on two physical GPUs) from the merge-correctness question
(deferred to stage 2), so a failure or success is unambiguous about which
mechanism it's testing.

`[V]` Added `fhe/gpu_real_scheme_b/shard_layout.py` +
`test_shard_layout.py` (7/7 pass): the Q/K/V partition assignment
(`{query,key}` on worker 0, `{value}` on worker 1) is exact and exhaustive,
each shard's independently-computed values match the real T=103 oracle,
and gathering both shards equals the unsharded single-process path. Also
makes explicit and quantifies (via the existing 2026-07-27 profiling
result, rotation/keyswitch `<0.1%` of one matmul call) that each worker
must independently recompute `baby_rotations` for every group it touches
(no cross-process ciphertext-compute sharing exists), while the LN1
*client* round trip itself is not duplicated (the trusted client can hand
the same re-encrypted ciphertext to both workers).

`[V]` **Two systems findings from reading the vendored FIDESlib source
(`/private/tmp/dnagpt-fideslib-audit-20260729`, same audit clone used for
the depth-13 investigation) and the existing writer/reader/launch scripts,
found before writing any new C++ -- avoided what would otherwise have been
a broken remote build attempt:**

1. `CryptoContextImpl<DCRTPoly>::LoadContext` (`api/CryptoContext.cpp`)
   silently **no-ops** (`if (this->loaded || this->devices.empty()) return;`)
   if `SetDevices` was never called on that exact in-process `cc` object --
   it does not throw, it just skips GPU loading. This looked like it could
   force per-reader `SetDevices` calls to pick a physical GPU different
   from whatever the writer used. It doesn't matter in practice: every
   `launch_brev_scheme_b*.sh` already launches each gate inside
   `docker run --gpus "device=${PHYSICAL_GPU}"`, so exactly one physical
   GPU is visible per container, always as device index 0 -- the
   binary's own `--gpu 0` (or whatever default `SetDevices` value survives
   deserialization) is correct regardless of which physical GPU Docker
   mapped in. Physical-GPU selection for a 2-GPU micro-gate is therefore
   entirely a Docker/launch-script concern, not something the C++ fork
   itself needs to handle.
2. **The existing `real_dnagpt_fides_scheme_b_serialize_writer/reader`
   pair is parameter-incompatible with the Token-SIMD B=8 T=103 layout**:
   the writer's `SLOTS = COPIES * PACK_WIDTH` (`4096`, the original
   single-token-per-ciphertext T=2 layout) versus Token-SIMD's
   `SLOTS = COPIES * PACK_WIDTH * TOKEN_BATCH` (`32768`), and its
   `required_rotation_keys_for_full()` generates the T=2 gate's rotation
   set, not Token-SIMD's `TOKEN_BATCH`-scaled one. A Stage-1 shard worker
   cannot deserialize the existing writer's state -- **a new,
   Token-SIMD-parameter-matched writer fork is a prerequisite**, forked
   from `real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp`'s
   own context construction (`SLOTS=32768`, `MULT_DEPTH=13`,
   `required_rotation_keys()`) with the writer's serialize-and-exit
   structure, before any sharded reader can be built. `[U]` This new
   writer fork, its matching sharded-reader fork (with a `--part
   query|key|value` selector replacing the existing reader's fixed
   single-query computation), and their contract tests are scoped but not
   yet written -- this is real additional engineering scope discovered
   during design, not a small addendum to the existing reader.

### 2-GPU Stage-2 MLP-chunk sharding: NOT buildable on the pinned FIDESlib -- dropped as infeasible (2026-08-01)

`[V]` **The staged plan's Stage 2 (split the four MLP down-projection
chunks 2-and-2 across two GPU processes and merge the partial-sum
ciphertexts with `EvalAdd`, per the "2-GPU process-per-GPU sharding"
design sketch above) is NOT buildable on the pinned FIDESlib** (commit
`786c7600`, container `dnagpt-fideslib:786c-asymfix2`, CUDA 13.0). A
2026-08-01 remote compile of both Stage-2 sources fails `nvcc` on exactly
one thing: **FIDESlib exposes no ciphertext serialization.**
`/usr/local/include/fideslib/Serialize.hpp` declares
`SerializeToFile`/`DeserializeFromFile` only for `CryptoContext`,
`PublicKey`, and `PrivateKey` -- never `Ciphertext`. This is consistent
with the library's `SetCiphertextAutoload(true)` design: ciphertexts stay
GPU-resident and have no host serialization path. The reader source fails
with `error: no instance of overloaded function
"fideslib::Serial::SerializeToFile" matches ... (std::string, const Ct,
fideslib::SerType)`, and the merge source fails with the
`DeserializeFromFile` analogue. **Everything else in both sources compiled
clean** -- the block is solely the cross-process ciphertext transport that
an `EvalAdd` merge between two GPU processes fundamentally requires.

`[V]` **Why Stage 1 (Q/K/V split) is unaffected**: Stage 1 serializes only
`context` + keys and never a ciphertext, so it hits none of this. Stage 1
built, ran concurrently on two clean physical GPUs, and passed
(2026-07-31 "Tracks 1 and 2 landed" entry). It remains valid.

`[V]` **The merge MATH is proven; only the transport is missing.** The
`EvalAdd` partial-sum merge is exact by construction (CKKS addition, no
approximation); its local NumPy contract passes at `max abs error
2.84e-14`. What is unavailable on this pinned FIDESlib is the cross-GPU,
2-process ciphertext transport that the sharded merge needs -- not the
correctness of the merge itself. The idea is not wrong; the library it
would run on cannot move a ciphertext between two processes.

`[U]` **Two forward directions exist, neither being pursued (each needs an
explicit go-ahead):** (1) a single-binary, in-process merge that avoids
serialization -- but this only re-associates `EvalAdd`s that the existing
complete-block gate already exercises and passes (`global_rel_inf`
`4.099e-9`), so it yields ~no new accuracy signal and no cross-GPU speed
proof; (2) extend FIDESlib itself with a `Ciphertext` serialize/deserialize
overload (library surgery on the pinned commit).

`[V]` **Decision: 2-GPU Stage-2 MLP-chunk sharding is DROPPED as infeasible
on the current pinned FIDESlib**, kept as a documented limitation.
Stage-1 Q/K/V sharding is unaffected. The two Stage-2 orphan sources
(`src/real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13.cpp`,
`sha256=68ac9e6b...`, and the `..._mlp_merge_...` sibling,
`sha256=5ef44118...`) exist inert on the remote and are **not** wired into
the shared build -- `CMakeLists.txt` and `build_in_fideslib.sh` were
restored pristine. The local contract
(`test_shard_mlp_reader_merge_t103_depth13_contract.py`, 38 tests) passes,
capturing the proven merge math for the record. No run JSON: this was a
compile-time infeasibility, not an executed gate.

### Combined-lever correctness micro-gate: CPU-diagonal-cache + Q/K/V Stage-1 sharding, run together for the first time (2026-08-01)

`[V]` **Scope**: the two previously-independently-passing Scheme B speed levers --
the CPU-side diagonal-plaintext cache (`..._cpudiagcache.cpp`, `global_rel_inf
4.4972e-9`, 99.99% hit rate) and the 2-GPU process-per-GPU Q/K/V sharding
(Stage 1, `..._simd_shard_writer/reader_t103_depth13.cpp`) -- had each passed
correctness on their own but had never been combined with each other. This is
a **correctness-only** micro-gate: does the combined recipe still pass, not a
speed run, not the 12-block end-to-end run.

`[V]` **Design**: forked the shard reader into
`src/real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.cpp`
(three pinned parents: the shard-reader, its own structural
`serialize_reader.cpp` grandparent, the Token-SIMD schedule source, and the
cpudiagcache source), porting `cached_packed_values()`/the cache-gated
`matmul()` in verbatim from the cpudiagcache parent. The writer needed **no
fork at all** -- the cache lever only touches the reader's `matmul()`, so
`wait_and_run_scheme_b_simd_shard_qkv_cpudiagcache_t103_depth13.sh` reuses the
existing (unchanged) shard-writer launch script unmodified. Per-shard cache
keying (by weight-vector stable address) needed no change: each of the two
shard workers is its own OS process with its own `std::map`, and each only
ever calls `matmul()` with the 1-2 weight addresses in its own `--part`
selection, so there is no cross-part or cross-process cache-sharing question
-- confirmed before coding, not just asserted after.

`[V]` **Local contract first**: `test_shard_reader_cpudiagcache_contract.py`
(27 tests) -- static/structural contract (parent hashes pinned, cache gates
`matmul()` not an inline rebuild, a fresh `Plaintext` still built every call,
per-shard hit/miss invariant asserted in `main()`, build/run/launch/
orchestrator scripts wired) plus a from-scratch Python simulation of the
cache's own hit/miss bookkeeping (`SimulatedCacheHitMissFormulaTests`) that
independently re-derives the exact counts the C++ invariant asserts. Did not
re-derive math already proved elsewhere: the per-shard Q/K/V split's exactness
against the real oracle is `test_shard_layout.py`'s job; the BSGS diagonal
construction being a pure function of `(weight, giant, small)` for each real
QKV matrix individually is `test_diagonal_cache_reuse.py`'s job. Full local
suite: 507/507 pass.

`[V]` **Remote build + fail-closed smoke checks**: built clean on the pinned
FIDESlib commit (`786c7600`, `dnagpt-fideslib:786c-asymfix2`). `--help`
exits 0; a deliberately wrong `--shard-reader-parent-sha256` is refused
(`[FATAL] refusing changed shard-reader parent`); the `run_scheme_b_*.sh`
wrapper refuses a missing `--state-dir` before ever invoking the binary.

`[U]` **First attempt (`_rep1`) failed closed on a bug in the NEW invariant
itself, not on the cache or the sharding.** Both workers ran all 13 token
groups to completion, then `main()`'s own
`diagonal_cache_hits`/`diagonal_cache_misses` assertion threw
`"diagonal-cache hit/miss count mismatch"` right before the evidence JSON
would have been written -- so `_rep1` produced no evidence file and is not
recorded as a manifest row. Root cause: the assertion assumed one
`cached_packed_values()` lookup per `matmul()` call; it is actually called
`BSGS_N1*BSGS_N2=1024` times per `matmul()` call (once per BSGS diagonal, all
built and cached together on the first call for a weight) -- exactly the
counting the frozen cpudiagcache gate's own
`EXPECTED_DIAGONAL_CACHE_HITS`/`_MISSES` constants already used, which this
fork's first draft failed to carry over correctly. Fixed
(`expected_hits = (TOKEN_GROUPS * BSGS_N1 * BSGS_N2 - 1) * requested.size()`),
re-verified against the from-scratch Python simulation above, rebuilt, and
re-run as `_rep2`.

`[V]` **`_rep2` PASS, both workers, combined levers confirmed compatible.**
Launched concurrently: worker A (`query,key`) on physical GPU 0, worker B
(`value`) on physical GPU 1.

| worker | parts | matrix products | diagonal cache hits/misses | global_rel_inf | encrypted_evaluation_seconds |
|---|---|---|---|---|---|
| A | query, key | 26 | 26622 / 2 | query `3.743e-9`, key `2.873e-9` | 643.06 (informational) |
| B | value | 13 | 13311 / 1 | `3.246e-9` | 439.63 (informational) |

Both at the unchanged `~1e-9` band, both cache hit/miss counts exactly match
the corrected formula (`(TOKEN_GROUPS*BSGS_N1*BSGS_N2 - 1) * requested.size()`
hits, `requested.size()` misses -- i.e. 2 and 1 distinct weights respectively,
each cached once and reused for the other 12 token groups' worth of BSGS
lookups). No cross-process cache leakage: each worker's map only ever holds
the weight address(es) it requested.

`[U]` **Timing is explicitly not a speed claim, per this gate's own scope.**
This host showed load average `180-212` during the `_rep2` run from the
concurrent Stage-2 MLP workstream's own activity documented immediately above
in this file (a separate, longer session on this shared Brev host) -- the
same class of host-wide CPU contention that invalidated the depth-13
clean-timing repeat samples (see the 2026-07-31 retraction entry). GPU0 was
also not genuinely clean at preflight (1 pre-existing compute process, `543
MiB`); GPU1 was genuinely clean (`0 MiB`/`0%`). VRAM: worker B's device-wide
peak (`14033 MiB`) is a clean reading on an uncontaminated GPU; worker A's
device-wide peak (`74351 MiB`, erratic `8335`-`74351 MiB` samples) sits on a
GPU shared with another process and is not a clean isolated reading -- its
own compute-app samples (host PID `3675909`) show a `14320`->`66548 MiB`
ramp, reported as the more trustworthy per-process figure.

`[V]` Evidence: 3 immutable run JSONs under `results/runs/`
(`fhe_fides_real_d768_t103_qkvshard_writer_cpudiagcache_scheme_b_a100_20260801_rep2.json`,
`..._reader_querykey_..._rep2.json`, `..._reader_value_..._rep2.json`) + 3
new rows in `results/hybrid/manifest.yaml`. New source
(`real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.cpp`)
+ run/launch/orchestrator scripts + contract test, all additive; the
concurrent Stage-2 MLP workstream's own in-flight `CMakeLists.txt`/
`build_in_fideslib.sh` edits on the shared remote host were detected and
preserved (idempotent single-round-trip patch applied on top of whatever was
live there, rather than a blind overwrite, after an earlier overwrite/
clobber collision was caught and corrected).
