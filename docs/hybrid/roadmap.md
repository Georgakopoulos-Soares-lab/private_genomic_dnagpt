# Client-assisted CKKS execution roadmap

This is the current decision document for the active encrypted-inference path. The legacy source and
result namespace contains `scheme_b`; use **client-assisted CKKS** in research prose.

Detailed experiment history belongs in [tasks.md](tasks.md). The concise account of optimizations that
worked, failed, or remain open is
[optimizations_and_combinations_report.md](optimizations_and_combinations_report.md).

## Decision

The current implementation is correct at the single-block task length, but it is not
optimization-complete. A dedicated host is necessary for defensible timing; it is not sufficient to
justify running the existing 12-block driver as the final performance experiment.

- Run the existing 12-block driver now only if the immediate goal is full-model arithmetic
  correctness. Label it a baseline feasibility run.
- For a performance result, first profile the current 103-token block, close the high-value exact-model
  gates below, rebuild the full driver, and then run on a dedicated host.
- A dedicated arithmetic run still does not measure networked end-to-end latency. That requires a real
  client/server transport experiment.

## Current evidence

| Claim | State |
|---|---|
| Complete released-weight block at 103 tokens | `[V]` passes at approximately `4e-9` relative error |
| Eight-token SIMD layout | `[V]` 156 dense products versus 1,236 serial-equivalent products |
| Current operation schedule | `[V]` 177,734 ciphertext–plaintext multiplications, 1,506 ciphertext–ciphertext multiplications, 8,173 explicit rotations, 8,776 reduction calls |
| Minimum demonstrated depth | `[V]` 13 at the retained scale, digit count, ring, and packing |
| Process peak GPU memory | `[V]` approximately `9.8 GiB` for the retained task-length block |
| Short composition | `[V]` blocks 0 and 1 pass at two tokens through a full-state client refresh |
| Full task-length driver | `[V]` builds and passes local contracts; `[U]` never run on a GPU |
| Clean block latency | `[U]` samples span roughly 2,986–13,658 seconds under different host load |
| Integrated multi-GPU block | `[U]` only Q/K/V process sharding has run; its outputs cannot feed the full process with the current backend |
| Networked protocol latency | `[U]` not implemented; current crossings are in-process cryptographic boundaries, not RPCs |

The older two-token synchronized profile found dense multiplication dominant. It does not prove where
time goes in the current graph, whose attention schedule and operation mix are substantially different.

## Stage 0 — dedicated baseline and current profile

Before changing the circuit:

1. run the retained task-length block with and without the CPU diagonal-vector cache;
2. use identical hardware, context parameters, fixture, and process placement;
3. warm up once and record at least three measured repetitions;
4. record host load, CPU utilization, GPU utilization, target-process RAM/VRAM, and stage times; and
5. capture one Nsight Systems trace with NVTX ranges for packing/encoding, plaintext upload, dense
   projections, score construction, client boundaries, attention context, MLP, synchronization, and
   output handling.

The trace must distinguish CPU vector construction, CKKS encoding, host-to-device transfer, CUDA API
and launch gaps, kernel execution, and explicit synchronization. Use CUDA Graphs, stream overlap,
pinned transfers, or kernel fusion only if that trace shows the corresponding cost. See the official
[Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) and
[CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

**Exit condition:** a stage-attributed current-block baseline with variance small enough to compare
micro-gates. If the dedicated host still varies materially, report the variance and fix measurement
control before claiming speedups.

## Stage 1 — local exact-model contracts

These gates require no GPU allocation. Each must reconstruct the real 103-token oracle and publish a
source-derived operation schedule before C++ work.

### 1. Distinct transforms across the four packed copies

The retained layout carries four copies of each activation but applies the same public matrix to all
four. Encode different diagonals per copy so one encrypted transform can:

- produce Q, K, and V in three copies;
- produce the four 768-wide MLP expansion chunks in four copies; and
- produce four MLP down-projection partials, followed by copy rotation and addition.

`[A]` Before mask and copy-repair overhead, this reduces dense products from 156 to approximately 52.
Dense ciphertext–plaintext multiplications fall from 159,744 to approximately 53,248, reducing the
complete current ciphertext–plaintext count by roughly 60%.

**Gate:** exact NumPy reconstruction for every full and tail token group, exact downstream copy
placement, a complete key/mask schedule, and a net operation-count reduction after repair overhead.

### 2. Complete LayerNorm at the existing client boundary

The client already participates in LayerNorm to evaluate the exact inverse square root. Test a variant
where it decrypts the packed hidden state, computes the complete LayerNorm including affine terms, and
returns a fresh normalized ciphertext.

Also fuse each inter-block full-state refresh with the next block's first LayerNorm, and fuse the final
refresh with the normalization required by the task head.

This preserves the released model and the server's lack of plaintext access, but moves more work to the
client. Treat work allocation as an explicit protocol choice. Re-derive depth from the resulting level
trace rather than assuming the current minimum of 13 remains necessary.

**Gate:** exact oracle equivalence, fixed non-adaptive boundaries, a documented client/server work
change, fewer encrypted operations, and a new minimum-depth proof.

### 3. Attention reduction and packing

The current schedule performs separate head reductions for every alignment. Test a segmented 64-slot
reduction that accumulates all 12 heads concurrently while masking head boundaries.

In parallel, model compact PC-MM and CC-MM layouts for the exact DNAGPT projection and attention shapes.
THOR evaluates `768×768×768×128` plaintext–ciphertext multiplication and twelve
`64×128 × 128×128` ciphertext products—shapes close to this workload—and reports gains from
diagonal-major encoding, compact packing, and specialized matrix algorithms. This is a broader family
than the single FIDESlib `LinearTransform` call site already measured negative. See the
[THOR primary publication](https://scholarworks.bwise.kr/hanyang/handle/2021.sw.hanyang/209896?mode=full)
and [fast homomorphic linear algebra with BLAS](https://arxiv.org/abs/2503.16080).

**Gate:** exact causal coverage and tail handling, exact attention-context oracle, complete counts, and
fewer reductions or encrypted multiplications than the retained schedule.

### 4. Wider and stage-specific layouts

B=8 beats B=4 under the fixed four-copy layout. That result does not close:

- B=16 with two copies;
- B=32 with one copy;
- packing two real batches in real and imaginary components;
- different layouts before and after client boundaries; or
- a smaller context for the final single-token head.

Client re-encryption can directly produce the layout and context required by the next stage, avoiding
homomorphic format conversion. Compare complete operation and ciphertext counts before building a GPU
gate.

### 5. Safe encoded-plaintext reuse

The retained cache reuses CPU `vector<double>` values but still creates a fresh CKKS `Plaintext` for
every multiplication. The six GPU-object reuse crashes identify a backend lifecycle/level-management
problem; they do not close the optimization.

Test, in order:

1. pre-encoding by required level without object reuse across incompatible levels;
2. batched encoding and upload;
3. a corrected FIDESlib plaintext-level path; and
4. a representative alternative matrix kernel/backend if the current API cannot retain encoded
   weights safely.

**Gate:** repeated use across the real level schedule, no object growth or memory regression, exact
oracle output, and a same-run encoding/upload or wall-time improvement.

## Stage 2 — isolated GPU micro-gates

Implement one representative call site for each locally passing Stage-1 idea. Use the Stage-0 baseline
and one-variable-at-a-time comparisons.

Retain a change only when:

- global and worst-token error remain within `4e-2` and in the expected numerical band;
- the full operation and boundary counts match the contract;
- target-process memory does not regress without an explicit benefit; and
- either a same-run sibling comparison or at least two dedicated paired repetitions improve the
  targeted cost.

A correct but slower primitive remains a documented negative result and is not propagated.

## Stage 3 — integrate one optimized block

Combine only retained micro-gates into one task-length block. Re-run all local contracts and then at
least three dedicated-host repetitions. Measure interaction, server/client split, level trace, memory,
operation counts, and the full numerical oracle.

Do not carry the current two-process Q/K/V result into this stage unless its ciphertext outputs can feed
the full evaluation. The current FIDESlib API lacks the ciphertext transport needed by the attempted
process-separated MLP merge. Recent multi-GPU systems instead combine placement with
communication–computation overlap; [AEGIS](https://arxiv.org/abs/2604.03425) is relevant design
evidence, not a transferable speedup claim for DNAGPT.

## Stage 4 — rebuild and run the complete classifier

Port the optimized block into the 12-block plus released GSR-head driver. Fuse inter-block refreshes
with following boundaries where Stage 1 proves it correct.

Run on a dedicated host and require:

- all twelve block outputs finite;
- final logits and label matching the frozen GSR oracle;
- per-block error and level traces;
- bounded live memory across block composition;
- three or more repetitions for stable latency; and
- no extrapolated block time presented as measured end-to-end latency.

## Stage 5 — networked end-to-end measurement

Implement ciphertext transport before calling the system end to end. Batch objects by dependency phase
rather than treating every ciphertext as a serial RPC. Measure:

- physical RPC count and payload bytes;
- serialization and encryption/decryption time;
- latency and bandwidth sensitivity;
- total client work and server work; and
- complete wall time from encrypted input submission to client-decrypted prediction.

Transport, authentication, key custody, malicious-server behavior, traffic analysis, and side channels
remain outside the arithmetic result until explicitly implemented and evaluated.

## Explicit protocol and model variants

These may be valuable but must not silently replace the main exact-model protocol:

- **Client-computed attention context:** the client receives V at the existing softmax boundary,
  computes `softmax(QKᵀ)V`, and returns encrypted context. This could remove hundreds of encrypted
  weight tiles and accumulations while moving substantial linear work to the client.
- **Model-changing compression:** low-rank factorization, structured or block-circulant weights,
  pruning, quantization, distillation, and token dropping require task-metric validation and a separate
  claim from exact released-model reproduction.
- **Backend or custom multi-GPU work:** porting newer matrix algorithms or adding ciphertext transport
  is a systems research branch, not a parameter tweak.

## Stop conditions

The optimization phase is complete only when all high-priority local contracts either:

1. fail exactness or produce no structural reduction;
2. pass locally but fail a controlled GPU micro-gate; or
3. are retained and integrated into the full driver.

Only after that closure may a clean dedicated-host full-model run be described as the optimized result.
