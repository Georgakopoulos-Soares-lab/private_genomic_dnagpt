# Execution plan

## One rule

Every step must add a verified claim. Skip any run that only repeats established
arithmetic on a slower or irrelevant path. An optimization is retained only when it
passes the unchanged oracle and improves measured wall time, peak memory, ciphertext
count, rotation count, or multiplicative depth.

Every encrypted correctness run uses `HEStd_128_classic`, one uninterrupted ciphertext
lineage, no intermediate decryption feeding evaluation, finite output, and global plus
worst-token `rel_inf <= 4e-2`. Every retained result gets a new immutable run JSON and
manifest entry.

## Verified foundation

- `[V]` The three plaintext tasks pass and provide frozen task oracles.
- `[V]` Every required CKKS primitive and bootstrap refresh passes independently.
- `[V]` One complete `D=8`, `T=4`, two-head DNAGPT-shaped block passes with global
  rel-inf `1.49e-3`, zero intermediate decrypt attempts, and one final decrypt.
- `[V]` The complete C++/CUDA block passes on one A100 with global rel-inf
  `1.8423e-4`, worst-token rel-inf `4.8125e-4`, and `93.405 s [gpu]` evaluation.
- `[V]` Released block-0 weights pass the encrypted `D=768`, `T=2` LayerNorm gate at
  global rel-inf `3.8193e-10`.
- `[V]` The released block-0 `D=768`, `T=2`, 12-head encrypted attention/projection
  gate passes at global and worst-token rel-inf `7.0183e-10` in `1,347.062 s [gpu]`.
- `[V]` The independent plaintext/export contract matches all 12 released blocks and
  the classifier head; `[V]` block-0 domains fail beginning at block 1.
- `[V/A]` A fixed public T=2 range-control schedule passes all 12 block outputs and
  the classifier with worst-token rel-inf `8.1174e-3` and zero domain violations.
- `[V]` The required FIDESlib asymmetric-Chebyshev correction is measured before and
  after patching (`0.292889` fail to `1.5282e-5` pass).
- `[V]` Local optimization probes preserve correctness:
  - BSGS reduces `D=16` rotations from 15 to 6.
  - hoisted baby rotations preserve the same encrypted result.
  - numerator-first attention saves two levels.
  - GELU degree 5 uses four levels and has `4.26e-3` sample-grid rel-inf.

The complete block is the CPU arithmetic anchor. The optimization probes are screening
evidence; each selected change still has to pass the complete-block and real-weight
oracles.

## Completed: GPU backend parity

The existing block schedule was implemented in C++/CUDA with current
[FIDESlib](https://github.com/CAPS-UMU/FIDESlib) and its patched OpenFHE interop. FIDESlib
2.1.3 exposes CKKS multiplication, hoisted rotation, bootstrap, CUDA acceleration, and
multi-GPU support; the completed parity implementation kept server-side evaluation GPU
resident.

The small block verified:

- ciphertext layout and rotations match the Python anchor;
- scale/level bookkeeping survives OpenFHE↔FIDESlib conversion;
- no secret key or intermediate decrypt enters server evaluation; and
- wall time, level trace, fixed polynomial domains, image digest, and operation
  schedule are recorded.

Use CPU fallback only for an operator that fails the gate in the current GPU library.
A mixed CPU/GPU path is not a milestone by itself because transfers can dominate.
Do not implement cryptographic CUDA kernels from scratch.

## Current: real-width block

Close one released-weight DNAGPT block at `D=768`, 12 heads, and `T=2` before sequence
scaling. `T=2` is the smallest non-degenerate causal-attention graph and exercises every
real block-0 matrix while containing memory. It uses public, query-independent
polynomial domains and never feeds a decrypted diagnostic into evaluation.

LayerNorm and QKV/12-head attention/projection have passed. The full 3072-wide GELU MLP
is the remaining block-0 gate. After the complete block passes, run the native
block-boundary refresh and two-block same-lineage gate before increasing sequence length.

The first full attempt is measured and retained as a failure: exp-plus-reciprocal
attention creates a 13-level causal-branch gap, so token 1 needs packed level 46
against a depth-43 chain. The active `T=2` version replaces that path with the exact
sigmoid score-difference identity. Raising depth to 49 without changing the graph is
the fallback/control, not the preferred performance path.

The twelve-block plaintext preflight rejected blind block-0 interval reuse and selected
public per-block domains, T=2 sigmoid attention, public-scaled LayerNorm, and full-domain
GELU. The intervals are `[A]` until calibrated on a broader public set; they are never
adapted from a decrypted private query.

## Scale and optimize

Increase sequence length only after the real-width block passes. `T=2` is not an
application milestone. Validate the packed layout at `T=8/16`, then use capacity gates
`T=32, 64, 103`; `T=103` is the current GSR task-representative target. Skip other
sizes unless a correctness, memory, or throughput boundary needs resolution. Test one
change at a time:

1. token/head packing and batched nonlinear evaluation;
2. parallel BSGS plus hoisted/double-hoisted rotations;
3. pre-encoded weight reuse and public-mask fusion;
4. numerator-first attention;
5. the lowest polynomial degrees that pass the full oracle;
6. kernel fusion, streams, and full GPU residency;
7. multi-GPU sharding only when one A100 cannot hold the live set.

Two timing repeats are required for a directional speedup; three or more are required
for a reported stable speedup. Always report wall time, operation split, bootstrap
count, host RAM, per-GPU VRAM, and ciphertext layout.

## Compose only after one block scales

Test two blocks first to validate refresh placement and accumulated CKKS error. If it
passes, advance directly to the twelve-block 0.1b backbone unless a four-block run is
needed to locate an error or memory boundary. Then add final LayerNorm and the GSR
readout head. The end gate is the encrypted embedded-vector-to-task-output graph at
`T=103`, not a T=2 block-only demonstration.

Input scope remains explicit:

- `[V]` The transformer path accepts encrypted embedded numeric vectors.
- `[U]` Encrypted token-index embedding lookup is separate.
- `[U]` Production key custody, transport, and client-only final decryption need a
  deployment harness after arithmetic closure.

## Explicit skips

- no full twelve-layer Python/OpenFHE CPU run;
- no duplicate “CPU container” full pass after one complete block has closed;
- no custom CUDA cryptography while current FIDESlib supplies the operation;
- no CPU/GPU hybrid milestone unless a measured GPU correctness failure forces it;
- no multi-GPU work until single-GPU memory or throughput is measured;
- no task-scale twelve-block run before a real-width block and a two-block refresh gate pass.
