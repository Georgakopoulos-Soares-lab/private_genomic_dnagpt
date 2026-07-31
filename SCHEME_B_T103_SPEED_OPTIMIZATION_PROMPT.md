# Scheme B T=103 speed-optimization handoff prompt

Use the prompt below in a new session. It is written to continue the existing
work without duplicating the live depth experiment or reopening measured-negative
speed levers.

## How to watch the queued depth experiment

The detached watcher waits until both host load and GPU occupancy satisfy the
repository capacity gate. Its driver log shows capacity polling; its
orchestrator log records launch, preflight, and exit; the `.done` file contains
the process exit code once the run terminates. A successful run also creates the
evidence JSON under `evidence/`. Do not stop the watcher, weaken the capacity
gate, or reuse this tag.

```bash
brev exec awesome-gpu-name -- '
BASE=/data/christos/private_genomic_ml/dnagpt_fides_asymfix2_20260724/gpu_real_scheme_b_general_attention_v1/gpu_real_scheme_b
TAG=fhe_fides_real_d768_t103_block0_simd_full_t103_depth8_digits3_ring65536_scheme_b_a100_20260730
cd "$BASE"
if test -f "$TAG.done"; then
  printf "done="
  cat "$TAG.done"
else
  echo "queued-or-running"
fi
tail -n 20 "$TAG.driver.log" 2>/dev/null || true
tail -n 20 "$TAG.orchestrator.log" 2>/dev/null || true
docker ps --filter "name=dnagpt-$TAG" \
  --format "{{.Names}} {{.Status}}" 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader
test -f "evidence/$TAG.json" && \
  python3 -m json.tool "evidence/$TAG.json"
'
```

Interpretation:

- no `.done`, no matching container: queued by the capacity gate;
- no `.done`, matching container: GPU computation is running;
- `.done` contains `0` and the evidence JSON exists: terminal success;
- nonzero `.done`: terminal failure; preserve every log and use a new tag for
  any corrected fork.

## Copy-ready prompt

You are continuing this repository's Scheme B DNAGPT-under-FHE performance
work. The concrete goal is a correct T=103 encrypted
embedded-vector-to-task-output run through all 12 released 0.1b backbone blocks
and the released task head, with the lowest defensible wall time. A few-hour POC
is acceptable; `<=3h` is the stretch target. Accuracy may be traded only when
the trade produces a real FHE parameter or operation-count reduction and the
effect is measured at the final downstream task output. Do not weaken the
privacy model, HE security level, evaluator/key separation, or declared client
boundary contract.

This is both an evaluation and an evidence-driven optimization task. Continue
implementation where the existing contracts justify it, but do not claim a
speedup, accuracy tolerance, memory bound, or end-to-end result without an
immutable run JSON.

### Read first, in order

1. `AGENTS.md`
2. `CLAUDE.md`
3. `docs/roadmap.md`, especially **Scale and optimize** and
   **Compose only after one block scales**
4. `docs/hybrid/roadmap.md` in full
5. `docs/hybrid/tasks.md`, including every 2026-07-27 through 2026-07-30 entry
6. `docs/hybrid/simd_current_state.txt`
7. `docs/hybrid/scheme_b_status_summary.md`
8. `results/hybrid/manifest.yaml`
9. `results/README.md`
10. The evidence JSONs named below and their pinned source/run contracts

Re-derive important numbers from the JSONs and source. Do not merely repeat the
documentation.

### Verified starting point

Use `[V]` for verified facts, `[U]` for open unknowns, and `[A]` for
assumptions.

- `[V]` The complete B=8 Token-SIMD T=103 released-weight block-0 graph passes:
  LN1, Q/K/V, chunked causal attention and exact client softmax, attention
  projection, residual, LN2, exact client GELU, full MLP, and final residual.
- `[V]` Correctness evidence:
  `results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_scheme_b_a100_20260730.json`,
  SHA-256
  `333461fbd83aad670d57f11d418cdb525d33d09e9d90d0f0390aaba2e8b6d6f5`.
- `[V]` Its `global_rel_inf=4.777473702899847e-9`,
  `worst_token_rel_inf=5.07508500941268e-9`, and
  `max_abs_error=7.659995376191331e-8`, far inside the unchanged `4e-2`
  block gate.
- `[V]` It uses 13 token groups, 156 encrypted dense products versus 1,236
  serial-equivalent products, an exact `7.923076923x` reduction.
- `[V]` It records 857 declared crossings, 129,162 logical boundary instances,
  zero intermediate evaluator decryptions, and no evaluator private key.
- `[V]` Observed encrypted evaluation was `6281.748445232s`:
  `6140.143438612s` server linear algebra plus `141.605006620s` client
  boundaries.
- `[U]` That timing is heavily co-tenant-contaminated. It is correctness
  evidence, not a dedicated-A100 throughput benchmark.
- `[V]` PID-specific memory evidence:
  `results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_scheme_b_vram_a100_20260730.json`,
  SHA-256
  `9157f2e1188e68859ac9f2cb8ebe75a83f76952be7fdb84e4b3f87350c1fa752`.
  The target process peaked at `12920 MiB`; single-block T=103 capacity fits
  one 80GB A100.
- `[A]` Multiplying the contaminated block observation by 12 gives `20.94h`.
  This is a planning projection, not measured end-to-end evidence.
- `[U]` No T=103 run has composed two blocks, all 12 blocks, or the task head.

The matched Token-SIMD micro-gate is also important:

- packed evidence:
  `results/runs/fhe_fides_real_d768_t103_packed_simd_linear_t103_scheme_b_a100_20260730_v2.json`;
- serial control:
  `results/runs/fhe_fides_real_d768_t103_serial_control_simd_linear_t103_scheme_b_a100_20260730_v2.json`;
- `[V]` packed: 13 products, `306.542390898s`, global `3.4048e-9`,
  target-PID peak `9848 MiB`;
- `[V]` serial: 103 products, `2260.075644194s`, global `8.8364e-9`,
  target-PID peak `15992 MiB`;
- `[V]` exact product-count reduction is `7.923x`;
- `[U]` observed timing ratio `7.3728x` is co-tenant-contaminated and the two
  modes ran in different windows.

Composition and task-length anchors:

- `[V]` T=2 blocks 0 -> 1 with a declared full-hidden-state client refresh
  passes under one context/key lineage:
  `results/runs/fhe_fides_real_d768_t2_blocks0_1_scheme_b_two_block_refresh_a100_20260729_v2.json`.
- `[V]` Its PID-specific peak is `10872 MiB` and does not grow in block 1:
  `results/runs/fhe_fides_real_d768_t2_blocks0_1_scheme_b_two_block_refresh_vram_a100_20260729_v2.json`.
- `[V]` T=103 chunked causal attention passes and removed the old T<=85
  packing ceiling:
  `results/runs/fhe_fides_real_d768_t103_attention_general_attention_t103_scheme_b_a100_20260729.json`.
- `[U]` The T=2 refresh result does not establish T=103 composition, 12-block
  runtime, or the released task output.

### Live depth experiment: reconcile before starting anything else

At handoff creation (`2026-07-30T12:25Z`), the corrected experiment is
capacity-queued on Brev; it has not allocated a GPU. Do not launch a duplicate.

- watcher PID: `857860`;
- run tag:
  `fhe_fides_real_d768_t103_block0_simd_full_t103_depth8_digits3_ring65536_scheme_b_a100_20260730`;
- source:
  `fhe/gpu_real_scheme_b/src/real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536.cpp`;
- source SHA-256:
  `65cc818cbb299af984cccdbe6a7df891cac5b7071f663edfac8c3f13be161365`;
- direct parent SHA-256:
  `c53e47123614bcddf6a577ca5cc1e22db786d234e953e5c31b6a488e3408c089`;
- parameters: B=8, ring 65536, depth 8, scale 50, three HYBRID large
  digits, unchanged security level and graph;
- local full Scheme B suite: `286/286` pass;
- remote contract and CUDA target build: pass.

Two earlier immutable remote attempts failed before FHE evaluation and produced
no evidence JSON:

1. Depth 8 with four HYBRID digits was rejected by OpenFHE because nine towers
   could not be validly distributed over four digits.
2. Depth 8 with three digits passed that check but OpenFHE auto-selected a ring
   whose capacity was below the requested 32,768 complex slots. It failed with
   “batch size cannot be larger than ring dimension / 2.”

The live fork changes only the declared identity/provenance fields and
explicitly fixes `ring_dim=65536`, which the B=8/32,768-slot layout requires
independently of depth.

First inspect the watcher using the command at the top of this file.

- If it passes, verify hashes, gate fields, accuracy, output level, timing,
  context parameters, and PID-specific VRAM. Copy via a temporary path, bank a
  new immutable correctness JSON and a separate derived VRAM JSON, then append
  the manifest and documentation. Never overwrite.
- If it fails, preserve and hash the raw logs, root-cause it, and use a new
  source filename and run tag for any correction. Do not mislabel a
  context-generation failure as an accuracy or runtime result.
- Compare depth 8 with depth 16 only after labeling co-tenant conditions. The
  graph and operation counts are matched, but contaminated wall times do not
  prove an isolated-A100 speed ratio.

### Main questions

1. What is the minimum passing CKKS modulus chain for the complete B=8 T=103
   block at unchanged security?
2. How much wall-time and memory reduction does that chain produce per real
   operation on an uncontended A100?
3. Can accepting more numerical error remove additional RNS primes or permit a
   different Token-SIMD/ring configuration without changing the downstream
   task decision unacceptably?
4. Is B=8/ring-65536 actually the fastest packing point, or can
   B=4/ring-32768 beat it despite doubling token groups?
5. How should independent work be sharded across 2, then 4, then at most 8
   GPUs, and what is the measured parallel efficiency?
6. Does the block-boundary refresh keep T=103 memory bounded and accuracy
   acceptable across two blocks, all 12 blocks, and the task head?
7. With all proven improvements combined, is a few-hour POC or `<=3h`
   end-to-end run measured, plausible but unproven, or ruled out?

### Work plan

#### 1. Finish the depth and precision Pareto sweep

- Start from the live depth-8 result.
- Derive candidate parameter sets from OpenFHE/FIDESlib modulus-chain and
  HYBRID digit constraints before consuming GPU time. Do not brute-force
  malformed configurations remotely.
- Keep `HEStd_128_classic` and explicit ring capacity valid.
- Record actual ring dimension, batch slots, modulus/tower count, large-digit
  count, output level, source hash, accuracy, timing, and PID memory in every
  valid experiment.
- If depth 8 passes, test only the smallest justified next depth candidates.
  `packed_output_level=6` is evidence of headroom, not proof that a
  depth-6 context will construct or pass.
- Sweep scale bits only when the change can remove or shrink real modulus
  limbs. Lower scale with the same tower structure is not automatically a
  useful speed optimization.
- Keep the existing `4e-2` block gate as the reference. Relaxed-error
  candidates may be exploratory, but they are acceptable only after the final
  downstream task output is measured. Do not invent a task-accuracy threshold;
  report a Pareto curve and label any acceptance choice `[A]`.

#### 2. Optimize Token-SIMD as a ring/packing trade-off

- Treat the proven B=8 schedule as the correctness anchor.
- Compare B=8/ring-65536 against a locally proved B=4/ring-32768 design at
  matched depth and scale. B=4 doubles dense groups but halves the ring; measure
  the trade rather than assuming either wins.
- Consider B>8 only after operation-count, ring, memory, and expected NTT-cost
  math shows a credible benefit. B=16 would require a larger ring and must not
  be attempted speculatively on the shared host.
- Contract-test pack/unpack, partial final groups, scaled rotations, masks,
  score tiles, causal boundaries, client crossing counts, and exact merge
  semantics before CUDA.
- Preserve the already-proven fixed-width chunked softmax. Do not redesign it
  without evidence of a new blocker.

#### 3. Implement process-per-GPU sharding

- Q/K/V projections, independent attention query/token groups where the merge
  is algebraically exact, and the four MLP chunks are candidate parallel
  branches. Blocks themselves remain sequential.
- Do not rely first on FIDESlib's native multi-device context path: a prior
  real 2-GPU attempt failed with a SIGSEGV in
  `SetupConstants`/`ContextData`.
- Prefer one process per physical GPU with the same serialized context/key
  lineage and exact encrypted merge semantics. Cross-process serialization was
  measured negative as a standalone speed lever, but it may still be necessary
  sharding infrastructure; count its overhead honestly.
- Begin with a two-GPU micro-gate that has a same-source one-GPU control.
  Require matching outputs and provenance. Then test 4 GPUs and only test
  8 GPUs if scaling remains useful.
- Record end-to-end wall time, critical-path server time, serialization and
  synchronization overhead, aggregate GPU-seconds, per-process VRAM, physical
  GPU mapping, and co-tenant telemetry.
- Never multiply an ideal `1/N` estimate and report it as measured scaling.

#### 4. Compose before making an end-to-end speed claim

- Build a T=103 Token-SIMD two-block gate using the already-proven declared
  block-boundary client refresh.
- Verify one context/key lineage, fresh level-0 input to block 1, no evaluator
  secret key, no undeclared decryptions, accuracy against the two-block
  plaintext oracle, and PID-specific memory across both blocks.
- Then extend to all 12 released blocks and the released task head.
- Measure final task logits/output, task decision, and the repository's actual
  downstream metric. Block-0 relative error alone is insufficient once
  precision is deliberately reduced.
- Report total elapsed wall time, server time, client time, setup, refresh,
  final decrypt, GPU-hours, peak memory, and per-block drift.

#### 5. Profile only remaining critical-path work

- After Token-SIMD, reduced parameters, and sharding are combined, profile the
  new critical path.
- Kernel fusion, CUDA streams, and keeping sequential block call sites
  GPU-resident remain open only if profiling shows they dominate.
- Do not reopen the closed levers below without new evidence that changes their
  mechanism.

### Closed or already-tried levers

- round-trip batching: adopted;
- in-process context caching: adopted, about 1.4% measured improvement;
- cross-process serialization as a standalone speed optimization: measured
  negative;
- BSGS N1/N2 retuning: ct-pt multiplication is approximately all of a profiled
  matmul call while rotation/keyswitch is below 0.1%, so this is not a current
  lever;
- diagonal plaintext caching: abandoned after 6/6 real-GPU OOM crashes, with no
  passing evidence;
- warm-up prelude: inconclusive because contention swamped the signal;
- native FIDESlib `LinearTransform` at one call site: measured 1.6-1.9x slower.

Do not reattempt these without new source or profiling evidence that changes the
calculus.

### Evidence and coordination rules

- Read before editing and preserve unrelated dirty-worktree changes.
- Do not edit frozen passing sources. Fork additively and pin the exact parent
  hash.
- Run local NumPy/static contracts before CUDA, then compile remotely before
  GPU allocation.
- Use only the repository host-load and idle-GPU gates. Never stop, preempt, or
  share a busy GPU with another tenant.
- Never reuse a run tag, output JSON, log, or container name.
- One valid run produces one immutable `results/runs/<tag>.json` and one
  manifest row. Memory telemetry is a separate immutable JSON.
- A number without a supporting run JSON does not exist. Raw failure logs may
  explain a failure but do not become passing performance evidence.
- Label co-tenant-contaminated timing `[U]`.
- Never commit unless the user explicitly requests it.
- Keep `[V]`, `[U]`, and `[A]` on every load-bearing claim.

### Required deliverables

1. A reconciled and banked result for the live depth-8/ring-65536 experiment.
2. A depth/scale/packing Pareto table with exact context structure, accuracy,
   time, memory, and evidence paths.
3. A measured 1-vs-2-vs-4-vs-8-GPU sharding curve, stopping when marginal
   scaling ceases to justify complexity or GPU-hours.
4. A passing T=103 two-block composition result with memory and drift.
5. A passing all-12-block-plus-task-head result, or a precise evidenced blocker.
6. The best measured end-to-end configuration and its total wall time,
   GPU-hours, peak memory, final task output/metric, and privacy-contract audit.
7. A verdict:
   - `[V] <=3h achieved`,
   - `[A] few-hour POC remains plausible but unmeasured`, or
   - `[V] current design cannot reach the target`, with the measured bottleneck.

Lead with measured outcomes. Do not tune the conclusion toward optimism or
pessimism.
