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

- `[V]` **The complete released-weight model (all 12 transformer blocks plus the encrypted GSR head)
  passes at the full 103-token GSR prompt length** (TACC Lonestar6, 2026-08-12; see
  [hybrid/tasks.md](hybrid/tasks.md)). Wall clock `6683 s` (~1.86 h) on a telemetry-confirmed clean
  single A100. Per-block relative-infinity error stays in the `2.3e-9`-`2.1e-8` band across all 12
  blocks and 11 inter-block client refreshes; the final decrypted head margin matches the frozen Phase-A
  plaintext oracle (label `N`) to a relative error of `8.6e-9`. Zero intermediate decrypts; the
  evaluator never holds the secret key.
- `[V]` Eight-token SIMD packing reduces dense products from 1,236 serial-equivalent products to 156.
- `[V]` The minimum demonstrated depth for the retained graph is 13.
- `[V]` Target-process peak GPU memory is approximately `9.8 GiB` per block; target-process peak host
  RAM across the complete 12-block run is `49.7 GiB` and does not accumulate block-over-block (each
  block's evaluator is freshly constructed).
- `[V]` A two-GPU Q/K/V substage passes; one unrepeated sample observed about `1.65x` substage
  improvement. It is not integrated into the full block.
- `[U]` The 2026-08-12 result is one clean sample, not yet reproduced a second time on an independent
  node -- repeat/variance confirmation is still open.
- `[U]` No production network, transport, authentication, key-custody, or side-channel evaluation has
  been performed. All boundary crossings measured so far are in-process function calls, not networked
  round trips.

Correctness is therefore now established for the complete classifier, not only a task-length block.
`1.86 h` for one encrypted example on a single A100 is a real, measured latency figure; whether that
counts as "practical" is a separate, deliberately unaddressed judgment call, not tuned toward either
answer.

## Current decision

The complete-model correctness question is closed for one clean sample. What remains before any
practicality claim:

1. reproduce the 12-block result at least once more on an independent clean node to bound variance;
2. evaluate further exact-model optimization gates (packing/layout/reuse changes) only where they
   preserve the oracle and improve a measured cost;
3. measure a real client/server transport separately -- every crossing measured so far is an in-process
   function call, not a networked RPC.

The ordered gates are in [hybrid/roadmap.md](hybrid/roadmap.md).

## Environment

Phase A uses Python 3.12, PyTorch float32, and MPS/CPU. Phase B uses pinned OpenFHE/FIDESlib C++/CUDA
environments and A100-class GPUs. Weights and datasets are not committed.
