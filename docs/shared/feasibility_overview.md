# Encrypted-inference feasibility overview

## Question and boundary

Can DNAGPT evaluate encrypted embedded genomic-token vectors while an untrusted compute provider sees
neither the input nor derived plaintext activations?

The current boundary starts after embedding and ends, in the intended complete experiment, at the
released task head. Private token-index lookup is unresolved. Production transport and key custody are
not implemented.

## Evaluated architectures

| Architecture | What passed | Why it is or is not active |
|---|---|---|
| Pure non-interactive CKKS | Required operators, complete toy block, CUDA parity, and one complete real-weight block | Frozen baseline. Three chained-composition configurations failed during GPU setup because evaluation material exceeded the tested A100 memory envelope. |
| Client-assisted CKKS | Exact client nonlinearities, general causal attention, eight-token SIMD packing, one complete 103-token real-weight block, and two-block refresh at two tokens | Active. It removes bootstrap/nonlinear depth accumulation while keeping server-side model arithmetic encrypted. |
| CKKS/FHEW scheme switching | Design-only option | Deferred because no suitable GPU-accelerated implementation was available in the evaluated stack. |

The architecture comparison and threat-model consequences are in
[architecture_options.md](architecture_options.md). Backend selection is in
[backend_selection.md](backend_selection.md).

## Strongest verified claims

- `[V]` The three plaintext task families pass and supply frozen oracles.
- `[V]` Pure non-interactive CKKS establishes that a real-weight DNAGPT block is arithmetically viable
  without intermediate decryption.
- `[V]` Client-assisted CKKS evaluates one complete real-weight block at the full 103-token GSR prompt
  length with approximately `4e-9` relative error.
- `[V]` Eight-token SIMD packing reduces dense products from 1,236 serial-equivalent products to 156.
- `[V]` The retained task-length block uses minimum demonstrated depth 13 and approximately `9.8 GiB`
  peak device-wide GPU memory used during the run.
- `[V]` Two released blocks compose at two tokens using a fixed client full-state refresh.
- `[V]` The 12-block plus GSR-head driver executes and passes at 103 tokens in one whole-node sample
  with no contention detected in recorded telemetry: `6683 s`, final label agreement, no
  homomorphic bootstrap, and device-wide GPU memory flat at `9839 MiB` after about 400 s.

## Unresolved claims

- `[U]` The complete run has not been independently repeated and only one selected prompt has been
  evaluated; encrypted task-set accuracy and run-to-run variance remain unknown.
- `[U]` A current task-length CPU/CUDA profile has not identified the causal performance bottleneck.
- `[U]` The current multi-GPU result covers only Q/K/V projection and is not connected to full-block
  execution.
- `[U]` The prototype has no real client/server ciphertext transport, so cryptographic boundary counts
  are not network round-trip measurements.
- `[U]` Private token lookup, malicious-server security, traffic leakage, and side channels are outside
  the present result.

## Current execution decision

Treat the existing complete execution as a single-sample feasibility result, not a final performance
experiment. First obtain a dedicated current-block profile, test the exact-model optimization gates in
[../hybrid/roadmap.md](../hybrid/roadmap.md), integrate retained changes, and repeat the complete
classifier on a dedicated host. A separate repetition of the current driver is useful for
reproducibility, but it does not replace profiling or a networked protocol experiment.

## Evidence ownership

- Pure non-interactive measurements: [../pure/measurements.md](../pure/measurements.md)
- Client-assisted detailed history: [../hybrid/tasks.md](../hybrid/tasks.md)
- Current client-assisted roadmap: [../hybrid/roadmap.md](../hybrid/roadmap.md)
- Paper-oriented results and limitations: [../paper/03_results_and_limits.md](../paper/03_results_and_limits.md)
- Immutable evidence: [`../../results/`](../../results/)
