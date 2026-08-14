# Methods and protocol

> **Status supersession, 2026-08-14.** The described full-hidden-state refresh now composes all
> twelve released blocks and the task head once at the 103-token prompt. The accepted run is one
> selected input and one execution; network transport and private token lookup remain outside the
> evaluated boundary. Historical statements below saying full composition has not executed are
> superseded by `evidence/measurements.yaml`.

The method combines CKKS arithmetic on an untrusted GPU server with exact nonlinear computation by
the data owner. It is designed to protect the genomic input from the compute provider, not to remove
the client from the inference loop.

## Research target

The target is the released `dna_gpt0.1b_m` classifier backbone:

- 12 transformer blocks;
- hidden width 768;
- 12 attention heads of width 64;
- MLP width 3,072; and
- the released GSR classification head.

The task-representative encrypted input is the full GSR prompt after DNAGPT's dynamic 6-mer
tokenization: 103 tokens derived from the 600-base-pair sequence used by the classifier.

The current encrypted graph begins after embedding. The client creates token and position embeddings
in plaintext, adds them, encodes the resulting numeric vectors, and encrypts them. Private lookup of
an encrypted token index is a separate problem and is not implemented here.

## Threat model

The parties are:

- **Data owner/client:** holds the genomic sequence and CKKS secret key. It can see its own input,
  intermediate values returned at declared boundaries, and final output.
- **Compute provider/server:** holds the public/evaluation material and model weights, and performs
  encrypted arithmetic on GPUs. It must not receive the secret key, a partial decryption, or plaintext
  query data.

The server evaluates fixed public model structure. Client boundaries are declared before inference
and do not depend on the decrypted value. The same protocol shape therefore applies to every input
of a given length.

The current evidence is an arithmetic and systems prototype, not a deployed security protocol. It
does not establish malicious-server security, authenticated transport, key rotation, access-pattern
privacy, traffic-analysis resistance, side-channel resistance, or production key custody. The
server's view of message sizes and boundary timing has not been analyzed as a leakage channel.

## Why CKKS

DNAGPT's expensive operations are dense maps, dot products, scalar-vector products, and additions on
real-valued activations. CKKS supports approximate real/complex arithmetic and packs many values into
one ciphertext, allowing rotations and component-wise operations to implement these maps.

The selected software separates correctness from performance:

- **OpenFHE** supplies CKKS context construction, security-parameter selection, encoding, rotations,
  polynomial evaluation, and approximate bootstrap semantics.
- **FIDESlib** supplies CUDA CKKS operations and interoperability with the OpenFHE representation.
- **Python/NumPy/PyTorch** supply independent plaintext oracles and test-input construction; they are not
  the encrypted performance path.

CKKS is the selected approach among the software paths evaluated in this project, not the only
cryptographic scheme that could express the graph.

## Plaintext oracle construction

Each plaintext task harness follows the released DNAGPT forward path and records per-example outputs.
For encrypted block experiments, an independent NumPy implementation reconstructs the selected
transformer boundary from released weights and compares it with the upstream PyTorch graph.

Encrypted correctness is measured against this frozen numerical output. The main criterion is
relative infinity error:

```text
relative infinity error = max(|encrypted - oracle|) / max(|oracle|)
```

Both global and worst-token values must be finite and no greater than `4e-2`. This tolerance is held
constant across pure and client-assisted experiments. The observed client-assisted errors are much
smaller, but the gate is not tightened after seeing a result.

## Pure non-interactive CKKS baseline

The baseline keeps one uninterrupted ciphertext lineage from encrypted embeddings to block output.
It evaluates nonlinear functions with Chebyshev polynomials:

- inverse square root for LayerNorm;
- exponential and reciprocal, or a two-token sigmoid identity, for attention; and
- GELU for the MLP activation.

Approximation domains are fixed from public data before private inference. An out-of-domain value is
supposed to fail closed; the evaluator must not choose a new interval after decrypting a private
query. Approximate bootstrapping restores multiplicative depth between deep circuits.

This protocol is non-interactive but expensive in depth and memory. It provides the comparison that
motivates client assistance.

## Client-assisted hybrid CKKS

The main protocol preserves server-side encryption for operations that CKKS handles efficiently and
moves exact nonlinear functions to the key-owning client.

```text
client encrypts embedded token vectors
  -> server: encrypted LayerNorm statistics and linear maps
  -> client: exact normalization boundary, re-encrypt
  -> server: encrypted Q/K/V and attention-score algebra
  -> client: exact causal softmax boundary, re-encrypt weights
  -> server: encrypted value aggregation, projection, and residual
  -> client: exact second normalization and GELU boundaries
  -> server: encrypted MLP projection, residual, and output packing
  -> client: decrypt result or refresh hidden state for the next block
```

The client computes only on values derived from its own request. The server never performs or
observes a decryption. Exact client nonlinearities eliminate Chebyshev approximation error and public
input-range calibration for those functions, but they introduce communication, client availability,
and exposure of intermediate activations to the data owner.

## Linear algebra

Weights remain plaintext model parameters; activations remain encrypted. Dense matrix multiplication
uses diagonal encoding with baby-step/giant-step scheduling:

1. rotate the ciphertext to expose groups of vector diagonals;
2. multiply each rotated ciphertext by an encoded plaintext diagonal;
3. accumulate the products; and
4. repeat across matrix tiles where the full weight does not fit one packing region.

At width 768, the implementation uses a 32-by-32 decomposition. Hoisted rotations and fixed packing
layouts reduce repeated key-switch work, while the CUDA path keeps encrypted arithmetic on the GPU.
Profiling showed that plaintext-diagonal multiplication and accumulation dominate the measured server
time; rotations and key switching were a small fraction of one matrix call in the tested build.

## Client-boundary batching

A client call may carry packed data for multiple independent evaluations in disjoint ciphertext
regions. At 103 tokens, the complete block makes 857 in-process client calls. Attention accounts
for 818 of them (95.4%) and for 818 of the 896 boundary ciphertext objects (91.3%). The separate
129,162 logical-instance assertion mixes token, token-copy, and head/query/key-scalar units and is
therefore a schedule check, not a scalar-work or traffic measure. Packing changes the physical
schedule rather than the model mathematics.

## General causal attention

For query position `i`, exact softmax is evaluated only over keys `0..i`. The server computes
encrypted dot products and packs score data for the client. The client applies stable softmax and
returns encrypted weights used by the server's value aggregation.

Short rows fit directly in the packing layout. Long rows use a two-phase chunked algorithm:

1. fixed-width score chunks cross the boundary and contribute to the row maximum and denominator;
2. the client combines those statistics and emits weights for each column without requiring the
   entire row to coexist in one ciphertext.

The chunk width is independent of total sequence length. This design removed the earlier packing
ceiling and supports the complete 103-token GSR prompt.

## Token-SIMD packing

The retained layout places eight logical tokens in separate lanes of one ciphertext. Matrix rotations
use strides that preserve those lanes, allowing one dense-map call to process up to eight tokens.

At 103 tokens this produces 13 token groups. For one complete block, the schedule performs 156 dense
products instead of 1,236 serial-equivalent products, a structural reduction of 7.92 times. A
four-token alternative was rejected from operation counts because it increased every measured work
category, including products, rotations, multiplications, attention decryptions, and client crossings.

The retained task-length configuration uses:

| Parameter | Value |
|---|---:|
| CKKS security profile | 128-bit classic security setting |
| Ring dimension | 65,536 |
| Multiplicative depth | 13 |
| Scale size | 50 bits |
| Logical tokens per ciphertext group | 8 |
| Complex packing slots | 32,768 |

Depth 13 is the minimum demonstrated for this graph, scale, and packing. It is not a universal
minimum for DNAGPT or CKKS.

## Block composition

The demonstrated block-to-block mechanism uses a client refresh rather than an encrypted bootstrap.
After block `n`, the client decrypts the packed hidden state, unpacks its token vectors, and re-encrypts
fresh level-zero inputs for block `n+1` under the same context and key lineage.

This mechanism first passed for released blocks 0 and 1 at two tokens. It then composed all twelve
released blocks and the task head at 103 tokens through eleven inter-block refreshes and one refresh
into the head. Device-wide GPU memory used during the complete run reached 9839 MiB after about
400 s and did not grow with block index.

## Measurement discipline

Each retained correctness experiment uses released weights, a frozen plaintext oracle, a declared
security configuration, fail-closed structural checks, and recorded client/server timing splits.

Historical block experiments ran on a shared eight-GPU host. An idle target GPU did not imply an
idle host, so their wall times remain excluded. The accepted complete-model execution instead used
a whole-node allocation with no contention detected in recorded telemetry, although scheduler
exclusivity was not requested. Consequently:

- correctness and exact operation counts remain usable;
- host RSS is reported per process, while retained GPU memory samples are device-wide;
- within-one-run call-site comparisons are stronger than comparisons across windows;
- cross-run wall time is labeled contaminated unless both host and GPUs stayed quiet; and
- projected all-block latency is superseded by the single measured `6683 s` complete execution.
