# Overview

## Purpose

Evaluate whether the released 0.1-billion-parameter DNAGPT model can process encrypted embedded DNA
tokens while an untrusted compute provider sees neither plaintext inputs nor plaintext activations.
The project measures numerical fidelity, memory, latency, interaction, and the boundary at which a
complete deployment stops being practical.

The encrypted boundary currently begins after token embedding. Private token-index lookup remains
unresolved.

## Plaintext oracle

Phase A establishes that the model path is meaningful before encrypting it. The frozen outputs from
three genomic task families are the numerical acceptance oracles for encrypted evaluation.

| Task | Local result | Verdict |
|---|---:|---|
| Human AATAAA signal recognition | accuracy `0.9124`, F1 `0.916`, `n=22,604` | `[V]` pass |
| Human mRNA abundance | r² `0.562`, Pearson `0.753`, `n=1,000` | `[V]` pass |
| GUE human promoter/splice subsets | MCC `0.680`, `0.897`, `0.831` | `[V]` pass |

These validate the target; they are not claims of state-of-the-art performance. Methods, commands,
and dataset caveats live in [tasks.md](tasks.md) and [data_provenance.md](data_provenance.md).

## Encrypted architectures

Two implemented paths matter to the paper:

- **Pure non-interactive CKKS** is the frozen baseline. It closes isolated operators, the complete toy
  block, and one complete real-weight block without intermediate decryption. Three chained-composition
  attempts then failed during GPU setup because rotation-key and bootstrap material exceeded the
  tested A100 memory envelope. This is a measured backend/configuration boundary, not an impossibility
  result for pure CKKS.
- **Client-assisted CKKS** is the active path. The GPU server evaluates encrypted linear algebra. At
  fixed LayerNorm, attention-softmax, GELU, and refresh boundaries, the data-owning client decrypts its
  own intermediate, computes the exact operation, and re-encrypts it. The server does not receive the
  secret key or plaintext activations under the stated prototype boundary.

See [shared/architecture_options.md](shared/architecture_options.md) and
[shared/backend_selection.md](shared/backend_selection.md) for the decision rationale.

## Strongest current encrypted result

- `[V]` One complete released-weight transformer block passes at the full 103-token GSR prompt length.
- `[V]` Eight-token SIMD packing reduces dense products from 1,236 serial-equivalent products to 156.
- `[V]` The minimum demonstrated depth for the retained graph is 13.
- `[V]` Relative-infinity error is approximately `4e-9`, far inside the `4e-2` gate.
- `[V]` Target-process peak GPU memory is approximately `9.8 GiB`.
- `[V]` Two released blocks compose correctly at two tokens using a client full-state refresh.
- `[V]` A two-GPU Q/K/V substage passes; one unrepeated sample observed about `1.65x` substage
  improvement. It is not integrated into the full block.
- `[U]` The built 12-block plus released GSR-head driver has not run at 103 tokens.
- `[U]` Existing long timings are contaminated by host load. Completed task-length block samples range
  from roughly 50 minutes to 3.8 hours for materially identical computation.
- `[U]` No production network, transport, authentication, key-custody, or side-channel evaluation has
  been performed.

Correctness is therefore established for a task-length block, not for the complete classifier.
Practical latency remains unestablished.

## Current decision

A dedicated host is required but does not by itself close the performance question. Before treating
the 12-block driver as a final latency experiment, the project will:

1. obtain a clean one-block baseline and current T=103 CPU/CUDA profile;
2. evaluate copy-fused dense projections, complete client-boundary LayerNorm, attention reduction and
   packing changes, wider layouts, and true encoded-plaintext reuse;
3. integrate only changes that preserve the oracle and improve a measured cost;
4. run the resulting 12-block plus head graph on a dedicated host; and
5. measure a real client/server transport separately.

The current driver may still be run immediately for arithmetic correctness closure, but that run is
a baseline feasibility result rather than an optimized end-to-end benchmark. The ordered gates are in
[hybrid/roadmap.md](hybrid/roadmap.md).

## Environment

Phase A uses Python 3.12, PyTorch float32, and MPS/CPU. Phase B uses pinned OpenFHE/FIDESlib C++/CUDA
environments and A100-class GPUs. Weights and datasets are not committed.
