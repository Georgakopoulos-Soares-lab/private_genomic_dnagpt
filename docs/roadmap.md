# Execution plan

## Acceptance rule

Every experiment must add a verified claim. Retain an optimization only when it passes the unchanged
plaintext oracle and improves at least one measured quantity: wall time under comparable conditions,
peak memory, encrypted operation count, interaction count, or required cryptographic depth.

All encrypted correctness gates use 128-bit-class security, finite-output checks, and global plus
worst-token relative-infinity error no greater than `4e-2`. The active client-assisted protocol also
records every declared client boundary and separates server-encrypted from client-plaintext time.

Accepted results receive a new immutable run file and manifest entry. Structural reductions are not
latency claims; contaminated observations are not benchmarks.

## Verified foundation

- `[V]` Three plaintext task families pass and supply frozen numerical oracles.
- `[V]` Pure non-interactive CKKS closes one real-weight block and then hits a root-caused GPU memory
  wall during chained-composition setup. That path is frozen as the non-interactive baseline.
- `[V]` Client-assisted CKKS closes a complete real-weight block at 103 tokens with eight-token SIMD
  packing, depth 13, approximately `4e-9` relative error, and approximately `9.8 GiB` peak
  device-wide GPU memory used during the run.
- `[V]` Two released blocks compose at two tokens through a declared client refresh.
- `[V]` The 12-block plus GSR-head task-length driver executes and passes at 103 tokens in one
  whole-node sample with no contention detected in recorded telemetry (`6683 s`, correct label,
  no homomorphic bootstrap).
- `[U]` Independent repetition, encrypted task-set accuracy, a current causal profile, and a real
  networked protocol remain unmeasured.

Historical pure-CKKS decisions are in [pure/roadmap.md](pure/roadmap.md). The active detailed plan is
[hybrid/roadmap.md](hybrid/roadmap.md).

## Ordered execution

### 1. Establish a clean current-block baseline

Run paired cached and uncached task-length blocks on a dedicated host, with warm-up, at least three
repetitions, process-specific CPU/GPU memory, GPU utilization, host load, and exact stage timing.
Capture one current T=103 Nsight Systems/NVTX trace. The older two-token profile does not identify the
bottleneck of the current quadratic-attention graph.

### 2. Close exact-model optimization gates

Use local slot/NumPy contracts before GPU work, then test one change at a time:

1. distinct dense transforms across the four existing packed copies;
2. complete LayerNorm at the existing client boundary and fusion with inter-block refresh;
3. segmented per-head attention reductions and specialized PC-MM/CC-MM packing;
4. wider B=16/B=32 layouts and stage-specific contexts;
5. safe pre-encoded plaintext reuse or batched encoding/upload;
6. launch, stream, graph, or kernel changes only when the current profile supports them.

The detailed hypotheses and go/no-go conditions are in [hybrid/roadmap.md](hybrid/roadmap.md).

### 3. Rebuild composition after retained changes

Integrate retained optimizations into one task-length block. Reconfirm the reference, level schedule,
memory, and operation counts. Then update and repeat the 12-block plus released GSR-head driver. Do
not assume individually passing micro-gates compose correctly.

### 4. Reproduce arithmetic end to end

Arithmetic correctness is closed for one selected prompt: all twelve blocks and the task head pass at
103 tokens on a dedicated host. Repeat the same case at least once to confirm reproducibility, and use
three or more repetitions before reporting a mean, variance, service rate, or stable latency target.

### 5. Measure the protocol, not only the arithmetic

Build a real client/server harness after arithmetic closure. Batch ciphertexts into dependency phases
and measure bytes, serialization, encryption/decryption, network latency, bandwidth sensitivity, and
end-to-end wall time. The current in-process “round trips” count cryptographic crossings, not network
RPCs.

## Separate research branches

The following may reduce latency but change the research question and must not be mixed into the
exact released-model claim:

- moving projections or attention-context multiplication to the client beyond the declared
  nonlinearity boundary;
- low-rank or structured weights, pruning, distillation, quantization, or token dropping;
- a backend replacement or custom multi-GPU transport layer.

Evaluate these as named protocol or model variants with their own threat-model and task-accuracy
checks.

## Explicit skips

- no new pure non-interactive CKKS run without a mechanism that addresses its measured memory wall;
- no full Python/OpenFHE CPU backbone run after the GPU arithmetic path has closed;
- no performance claim from shared-host long runs;
- no multiplication of a contaminated one-block time into a claimed full-model latency;
- no production-security or private-token-lookup claim without implementing and evaluating it.
