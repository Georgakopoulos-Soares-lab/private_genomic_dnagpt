# Claims for outside readers

> **Status supersession, 2026-08-14.** The paper may now claim one measured complete-model
> execution: twelve released blocks plus the GSR head at 103 tokens, one selected prompt, one
> whole-node run with no contention detected in recorded telemetry, final label agreement, and
> 6683 s wall time. Older prohibitions below against any
> complete-model or final-label claim are historical. It may not claim encrypted task accuracy,
> run-to-run variance, networked protocol cost, private lookup, or input-domain numerical
> generality.

This note translates repository evidence into language suitable for an abstract, introduction,
results section, talk, or review response. It is intentionally conservative.

## Strongest defensible headline

> Client-assisted CKKS executes all twelve released DNAGPT blocks and the task head at the full
> 103-token GSR prompt from encrypted embedded vectors. The selected prompt matches the reference
> label; the measured single-process cost is 6683 s, which rules out interactive use in the
> evaluated form.

This is stronger than an operator demonstration and weaker than end-to-end private inference.

## Short abstract language

The following paragraph is defensible with the current evidence:

> We evaluate privacy-preserving inference for the 0.1-billion-parameter DNAGPT model. After
> reproducing its performance on genomic signal recognition, promoter and splice-site classification,
> and mRNA abundance regression, we compare two CKKS protocols. A pure non-interactive baseline
> completes one real-weight two-token transformer block but exceeds the tested A100 memory envelope
> when configured for chained composition. A client-assisted protocol keeps linear algebra encrypted
> on the GPU server while the data owner evaluates exact nonlinearities at fixed boundaries. With
> chunked causal softmax and eight-token SIMD packing, this protocol executes all twelve released
> blocks and the task head at the full 103-token GSR prompt. For the one selected prompt, the label
> matches the reference and the margin has `8.56e-9` relative error. One in-process execution takes
> `6683 s`; private embedding lookup, network transport, repetition, and encrypted task accuracy
> remain open.

## Claims that can be made

### DNAGPT baseline

- The released DNAGPT classification and regression heads reproduce useful results on full or
  reconstructed canonical test sets.
- A locally fine-tuned DNAGPT backbone produces promoter and splice-site results near approximate
  DNABERT-2 references on three selected GUE datasets.
- Per-example plaintext outputs provide an explicit numerical oracle for encrypted evaluation.

Required qualifier: GUE uses a locally trained head and covers three of 28 datasets; the recovered
Xpresso split may not be gene-for-gene identical to the historical split.

### CKKS selection

- CKKS is a practical match for DNAGPT's real-valued, matrix-heavy arithmetic because it supports
  approximate SIMD arithmetic and rotations.
- OpenFHE supplied the correctness semantics and FIDESlib supplied the evaluated GPU path.
- This was the best fit among the software paths evaluated in the project.

Required qualifier: this is not proof that CKKS is the only suitable encryption scheme.

### Pure non-interactive CKKS

- Isolated nonlinear operators, approximate refresh, a complete toy block, and one complete
  real-weight two-token block pass without intermediate decryption.
- Chained composition failed because the tested deep context and evaluation material did not fit the
  available 80 GB GPU memory.
- The failure motivated the client-assisted architecture by identifying nonlinear approximation depth
  as the memory driver.

Required qualifier: the failure is scoped to the chosen library, parameters, packing, and hardware;
it is not an impossibility result for pure CKKS.

### Client-assisted hybrid CKKS

- The compute server performs dense linear algebra on encrypted activations and never receives the
  secret key or a plaintext intermediate.
- The data owner evaluates the LayerNorm inverse square root, causal softmax, GELU, and declared
  refreshes at fixed boundaries.
- A complete released-weight block passes at the complete 103-token GSR prompt length.
- Eight-token SIMD packing reduces dense products from 1,236 serial-equivalent products to 156.
- The minimum demonstrated depth for the retained task-length graph is 13.
- Two released blocks compose correctly at two tokens using a client full-hidden-state refresh.
- All twelve released blocks and the task head compose at 103 tokens for one selected prompt.
- Device-wide GPU memory used during the complete run reaches 9839 MiB and does not grow with block
  index after loading.

Required qualifier: the protocol is interactive; the numerical result covers one selected prompt
and one complete execution, not encrypted task accuracy or a service rate.

### Optimization evidence

- The CPU-side diagonal-vector cache works structurally and adds no GPU allocation.
- Q/K/V projections can be split across two GPU processes while preserving correctness.
- Some intuitive optimizations failed: GPU plaintext reuse crashed, a native linear primitive was
  slower, and process-separated MLP merging was blocked by missing ciphertext transport.

Required qualifier: no clean full-block cache speedup or complete multi-GPU block speedup has been
measured.

## Claims that require careful wording

| Tempting wording | Defensible replacement |
|---|---|
| “DNAGPT runs under FHE.” | “All twelve released DNAGPT blocks and the task head execute from encrypted embedded vectors under client-assisted CKKS for one selected prompt.” |
| “The full GSR input is encrypted end to end.” | “The full 103-token embedded prompt is encrypted; token-index embedding lookup occurs before the encrypted boundary.” |
| “The hybrid method is 6.6 times faster.” | “The first two-token hybrid observation used a much smaller circuit and completed faster than the pure baseline; it was not a controlled paired benchmark.” |
| “Token packing gives a 7.4-times speedup.” | “Token packing reduces dense products by 7.92 times; a contaminated micro-gate showed a similar directional wall-time ratio.” |
| “Depth 13 is faster.” | “Depth 13 is the smallest demonstrated passing depth; shared-host repeats do not establish a speed advantage.” |
| “One A100 is enough for DNAGPT.” | “One complete 103-token execution ran on one 40 GB A100; device-wide memory used reached 9839 MiB, but latency, throughput, and other inputs remain uncharacterized.” |
| “Two GPUs accelerate the model.” | “Two GPUs reduced one Q/K/V projection substage in a single unrepeated sample.” |
| “The server learns nothing.” | “The server does not receive plaintext activations or the secret key under the stated prototype boundary; network metadata and side channels were not analyzed.” |

## Claims to avoid

Do not state or imply that the project has demonstrated:

- encrypted GSR task accuracy or input-domain numerical stability from the one evaluated prompt;
- run-to-run latency variance, throughput, or a service rate;
- encrypted tokenization or embedding lookup;
- pure, non-interactive execution of the primary protocol;
- a clean latency benchmark, production throughput, or stable speedup;
- an optimization-complete implementation;
- a general impossibility result for pure CKKS;
- security against a malicious server, side channels, traffic analysis, or compromised clients;
- production key management or network deployment;
- privacy from the data owner itself;
- clinical privacy compliance, clinical efficacy, or biological discovery; or
- state-of-the-art GUE performance across the full benchmark.

## Reviewer questions and direct answers

### Why validate DNAGPT before encrypting it?

Encrypted arithmetic needs a meaningful, reproducible target. The plaintext stage verifies the model
path and freezes exact outputs. Otherwise an encrypted block could agree with a faulty reimplementation
and still appear correct.

### Why is this not ordinary fully homomorphic inference?

The primary protocol is deliberately interactive. CKKS protects every server-side linear operation,
but the key-owning client decrypts its own intermediates at fixed nonlinear boundaries. The pure
non-interactive baseline is reported separately.

### Does the client see sensitive intermediates?

Yes. The client is the data owner and already knows the query and secret key. The privacy objective is
to prevent the untrusted compute provider from seeing the input or derived activations.

### Why not approximate the nonlinearities under CKKS?

That path was implemented and completed one real-weight block. Repeated polynomial approximations
required a deeper context and bootstrap schedule whose evaluation material exceeded the tested GPU
memory envelope at composition. Client evaluation removes that measured bottleneck at the cost of
interaction.

### Why should the `1e-9` block error be trusted if timing is contaminated?

Correctness compares the decrypted output with a frozen plaintext oracle and was stable across runs.
Host contention changes wall time, not the mathematical comparison. Timing and correctness therefore
carry different evidentiary strength.

### Is 103 tokens a toy sequence?

It is the complete tokenized prompt length for the selected GSR task, not an arbitrary short prefix.
All twelve blocks and the task head have executed at that length, but only for one selected prompt.

### What evidence would most strengthen the current claim?

Repeat the same complete execution and evaluate a small, predeclared panel of prompts spanning both
labels and plaintext margins. For performance, obtain a current task-length CPU/CUDA profile and a
controlled pre-cache/current comparison before integrating further exact-model changes. Networked
latency and serialized byte volume require a separate transport implementation.

## Paper hygiene checklist

Before moving a statement into manuscript prose:

- identify whether it is measured, derived, observed under contamination, or still unmeasured;
- name the exact model boundary: operator, attention subgraph, one block, two blocks, or full model;
- name the sequence length and whether it represents a real task prompt;
- say whether nonlinearities were encrypted or evaluated by the client;
- distinguish homogeneous client-call and ciphertext-object counts from the heterogeneous
  logical-instance schedule counter;
- identify whether GPU memory is a per-process counter or the sampler's device-wide value;
- avoid converting a structural operation reduction into a latency claim;
- avoid multiplying a contaminated one-block time into a full-model result;
- distinguish a baseline full-model correctness run from an optimized performance run;
- preserve the embedding-lookup limitation;
- preserve the GUE and Xpresso dataset caveats; and
- recheck every quantitative value against its canonical source immediately before submission.
