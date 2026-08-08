# Paper evidence sourcebook

This directory gathers the material needed to draft a paper about privacy-preserving DNAGPT
inference with client-assisted CKKS. It is not the manuscript and it does not establish scientific
status on its own. The current audit cutoff is **2026-08-01**.

The notes deliberately use terms an outside reader can understand. They describe the two evaluated
protocols as **pure non-interactive CKKS** and **client-assisted hybrid CKKS**, not by the internal
letters used during development. They also omit result identifiers, checksums, fixture names, and
other repository bookkeeping that does not belong in a scientific narrative.

## Read in this order

1. [Research timeline](01_research_timeline.md) — why DNAGPT was validated first, why CKKS was
   selected, what the pure encrypted path established, why it stopped, and how the client-assisted
   path reached a complete task-length block.
2. [Methods and protocol](02_methods_and_protocol.md) — threat model, encrypted boundary, model
   decomposition, CKKS mapping, client nonlinearities, packing, composition, and acceptance rules.
3. [Results and limits](03_results_and_limits.md) — plaintext model results, encrypted measurements,
   useful negative results, timing contamination, and the remaining boundary to a complete model.
4. [Claims for outside readers](04_claims_for_outside_readers.md) — defensible headline language,
   required qualifiers, claims to avoid, reviewer questions, and paper hygiene.

## Canonical ownership

These notes are derived views. When a number or interpretation differs, correct the sourcebook from
the canonical repository document rather than treating this directory as a competing ledger.

| Question | Canonical source |
|---|---|
| What is the project trying to establish? | [Project overview](../overview.md) |
| How was DNAGPT evaluated? | [Evaluation approach](../eval_approach.md) |
| What are the plaintext task results? | [Task methods and results](../tasks.md) |
| Where did the datasets come from? | [Data provenance](../data_provenance.md) |
| Why CKKS and these software backends? | [Backend selection](../shared/backend_selection.md) |
| Why did the architecture become client-assisted? | [Architecture comparison](../shared/architecture_options.md) |
| What did pure non-interactive CKKS establish? | [Pure CKKS measurements](../pure/measurements.md) |
| What is the detailed client-assisted evidence? | [Client-assisted task history](../hybrid/tasks.md) |
| Which optimizations worked or failed? | [Optimization synthesis](../hybrid/optimizations_and_combinations_report.md) |
| What must run next, and in what order? | [Client-assisted execution roadmap](../hybrid/roadmap.md) |

Quantitative manuscript prose should be checked against the canonical source immediately before it
is used. The development documents contain more detail than belongs here, including failed launch
mechanics, source-integrity checks, and experimental bookkeeping.

## Defensible paper shape today

1. Establish that the released 0.1-billion-parameter DNAGPT model is meaningful on three genomic
   task families and freeze plaintext outputs as encrypted-computation oracles.
2. Explain why approximate-real CKKS is a natural arithmetic match and why OpenFHE plus FIDESlib was
   selected from the evaluated software paths.
3. Use pure non-interactive CKKS as the baseline: individual operators, a toy block, and one complete
   real-weight block pass, but chained composition exceeds the tested GPU memory envelope.
4. Present client-assisted hybrid CKKS as the main method: linear algebra remains encrypted on the
   server; the data owner evaluates exact nonlinearities at fixed boundaries.
5. Show progression from two tokens to the complete 103-token GSR prompt, including general causal
   attention, chunked softmax, token-SIMD packing, and a complete real-weight block.
6. Separate correctness from practicality. Correctness is strong at the measured scope; clean
   latency, all-block composition, the task head, and private embedding lookup remain open.
7. Report negative results that explain the retained design and constrain future work.
8. Keep the optimization audit visible: the current driver can close arithmetic correctness, but
   copy-fused projections, fuller use of client boundaries, attention packing, and encoded-weight
   reuse must be resolved before claiming optimized latency.

## Terminology and evidence rules

- Use **measured**, **observed**, **derived**, **unmeasured**, and **failed** in ordinary prose.
- A **task-length block** means one transformer block evaluated at the 103-token GSR prompt length.
  It does not mean the complete twelve-block classifier.
- A **client boundary** is a fixed decrypt/compute/re-encrypt operation performed by the data owner,
  never by the compute server.
- A **round trip** is a physical boundary crossing. One crossing may batch many logical nonlinear
  calculations.
- Do not call contaminated shared-host wall time a benchmark or stable speedup.
- Do not describe the current system as end-to-end encrypted DNAGPT: encrypted token lookup, all
  twelve blocks, and the released task head have not closed together.

## Update rule

When new accepted evidence changes the paper story:

1. update the canonical task/status documents first;
2. add the scientific event to the timeline;
3. update the methods note only if the protocol or algorithm changed;
4. revise the results table with its scope and limitations;
5. tighten or expand outside-facing claims only after the supporting measurement exists; and
6. recheck local links and the terminology rules above.

Do not place weights, sequences, activations, keys, ciphertexts, credentials, raw tensor payloads,
or development-only provenance identifiers in this directory.
