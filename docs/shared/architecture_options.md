# Architecture options and decision

## Trigger

Pure non-interactive CKKS closed one complete real-weight DNAGPT block. Chaining a second block then
failed in three parameter configurations while loading rotation-key and bootstrap material. A
symbolized trace placed the failure in GPU allocation, before useful composition arithmetic. The
tested context and evaluation material exceeded one A100's memory envelope.

The depth requirement came primarily from repeated polynomial approximations of LayerNorm inverse
square root, attention nonlinearity, and GELU. This made the nonlinear schedule—not the model's dense
maps—the immediate composition blocker in the evaluated stack.

This is a measured backend/configuration boundary. It is not a proof that pure CKKS or non-interactive
encrypted transformers are impossible.

## Evaluated options

| | Pure non-interactive CKKS | Client-assisted CKKS | CKKS/FHEW scheme switching |
|---|---|---|---|
| Mechanism | Linear and nonlinear work stays in one CKKS lineage; nonlinearities use polynomial approximations and bootstrap refresh | Server evaluates encrypted model algebra; the key-owning data client evaluates exact functions at fixed boundaries and re-encrypts | CKKS values switch to an exact LUT-oriented scheme for nonlinearities, then return to CKKS |
| Server sees plaintext or key | No | No | No |
| Online client | Only input/output | Required at fixed boundaries | Only input/output |
| Nonlinearity error/depth | Approximation and substantial depth | Exact client arithmetic and fresh ciphertexts | Exact LUT arithmetic with switching cost |
| GPU path in evaluated stack | Implemented | Implemented for encrypted work | No suitable integrated GPU path found |
| Current state | Frozen baseline after measured composition memory wall | Active | Deferred research option |

Legacy implementation and result namespaces call these paths `Scheme A`, `Scheme B`, and `Scheme C`.
External prose should use the descriptive names above.

## Decision

Use **client-assisted CKKS** as the active architecture.

- It removes the measured polynomial-depth/bootstrap pressure while reusing the proven CKKS GPU
  linear algebra and plaintext oracles.
- The client is the data owner and secret-key holder. It decrypts only values derived from its own
  query at fixed, predeclared boundaries. The untrusted server does not perform or observe decryption.
- Exact client nonlinearities avoid a new approximation-domain calibration problem.
- The pure path remains a useful non-interactive baseline and negative composition result.
- Scheme switching remains deferred because adopting it would require a new backend integration and a
  fresh correctness/performance program.

The decision protects data from the server under the prototype boundary. It does not establish
malicious-server security, traffic-analysis resistance, side-channel resistance, private model
weights, production key custody, or network authentication.

## Current client-assisted protocol

The server performs encrypted projections, attention algebra, residual additions, and MLP operations.
The client participates at:

- LayerNorm nonlinear statistics;
- exact causal softmax;
- GELU; and
- declared full-state refreshes between blocks.

The task-length block currently records 857 ciphertext crossings and 129,162 logical boundary
instances. These occur in one process and are not 857 WAN RPCs. A network implementation can batch
independent ciphertexts into a small number of dependency phases, but must measure payload volume,
serialization, latency, and bandwidth rather than inferring them from the local counter.

## Acceptance contract

Every accepted client-assisted correctness gate requires:

- 128-bit-class OpenFHE security parameters;
- one declared context/key lineage per stage unless a client boundary explicitly starts a new one;
- decryption only by the data-owning client at boundaries fixed before execution;
- no adaptive boundary chosen from decrypted content;
- finite output and global plus worst-token relative-infinity error no greater than `4e-2` against the
  frozen oracle;
- recorded encrypted operation counts, client boundary counts, and server/client wall-time split; and
- explicit scope: operator, subgraph, one block, short composition, or complete model.

## Permitted optimization variants

The active roadmap may change packing, context parameters, client-boundary batching, or the amount of
LayerNorm performed at an already-declared client boundary while preserving the model and the server
privacy objective. Any change that moves substantial projection or attention-context work to the
client must be named as a protocol variant and report the new work allocation.

Model compression, pruning, quantization, distillation, or structured weights are separate
model-changing branches and cannot support the exact released-model claim without task-level
revalidation.

Current execution order: [../hybrid/roadmap.md](../hybrid/roadmap.md). Detailed evidence:
[../pure/tasks.md](../pure/tasks.md) and [../hybrid/tasks.md](../hybrid/tasks.md).
