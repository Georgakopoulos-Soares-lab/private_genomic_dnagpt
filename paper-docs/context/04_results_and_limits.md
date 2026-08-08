# Results and limits

> **Superseded on timing — read this first.**
>
> This note predates the optimization campaign. Any latency, wall-clock, or server/client
> timing figure below has been replaced by the dedicated-node measurements in
> [`../evidence/measurements.yaml`](../evidence/measurements.yaml) and
> [`../evidence/optimizations.yaml`](../evidence/optimizations.yaml): one complete block at
> 103 tokens now runs in **652 s of encrypted evaluation** (663 s wall), down from
> approximately 2.1 hours, at a relative error of `4.64e-9`.
>
> The older figures here were measured on a contended shared host and **do not go in the
> manuscript in any form** — not as results and not as caveats. What remains valid in this note
> is everything contention cannot affect: the protocol, the threat model, the argument structure,
> the operation counts, the depth, the memory footprint, and the claim discipline.
>
> When this note and the evidence ledger disagree about a number, the ledger wins.

The current evidence supports a narrow but substantive conclusion: DNAGPT is a meaningful plaintext
target, and one complete real-weight transformer block can be evaluated at the full GSR prompt length
with client-assisted CKKS without exposing plaintext query activations to the compute server. It does
not yet support a claim of complete private DNAGPT inference.

## Plaintext model correctness

The plaintext stage used canonical or reconstructed test splits and the released 0.1-billion-parameter
models. The encrypted study inherits these outputs as its numerical oracles.

| Task | Local result | Comparison | Interpretation |
|---|---:|---:|---|
| Human AATAAA signal recognition | accuracy `0.9124`, F1 `0.916`, `n=22,604` | DeepGSR accuracy about `0.916` | The released DNAGPT classification head reproduces a strong full-set signal-recognition result. |
| Human mRNA abundance | r² `0.562`, Pearson `0.753`, Spearman `0.745`, `n=1,000` | DNAGPT paper r² about `0.62` | The released regression head preserves strong correlation, with a modest reference gap. |
| GUE core promoter | MCC `0.680`, accuracy `0.840` | DNABERT-2 MCC about `0.69` | Locally fine-tuned DNAGPT is near the approximate reference. |
| GUE 300-base-pair promoter | MCC `0.897`, accuracy `0.948` | DNABERT-2 MCC about `0.87` | Locally fine-tuned DNAGPT exceeds the approximate reference. |
| GUE reconstructed splice sites | MCC `0.831`, accuracy `0.895` | DNABERT-2 MCC about `0.85` | Locally fine-tuned DNAGPT is close to the approximate reference. |

These are model-validation results, not five independent claims of state-of-the-art performance.
GSR and mRNA use released fine-tuned heads. GUE required local fine-tuning because no released head
exists, and covers three of 28 benchmark datasets. The original Xpresso host was unavailable; the
data and split procedure were recovered, but historical test-gene identity is unverified and is a
plausible contributor to the mRNA r² gap.

## What pure non-interactive CKKS established

| Boundary | Result | Scope |
|---|---:|---|
| Required operators | Passed the `4e-2` relative-error criterion | Reduced deterministic inputs; isolated operations |
| Complete two-head toy block | global error `1.49e-3` on CPU and `1.84e-4` on GPU | Reduced width and four tokens; no intermediate decryption |
| Real-weight block-0 LayerNorm | global error `3.82e-10` | Width 768, two tokens |
| Real-weight block-0 attention | global error `7.02e-10` | 12 heads, two tokens |
| Complete real-weight block 0 | global error `6.41e-6` | Two-token sigmoid attention identity; depth 43 |
| Chained block composition | Failed during GPU context loading | Context and evaluation material exceeded the tested 80 GB GPU envelope |

The complete two-token block is the strongest pure encrypted arithmetic result. The composition
failure was caused by memory allocation for the deeper context, rotation keys, and bootstrap
precomputation, not by an incorrect decrypted output. Pure CKKS therefore remains a useful baseline,
but the tested implementation did not reach a second block.

## Client-assisted CKKS progression

### Complete two-token block

Moving nonlinearities to fixed client boundaries reduced the configuration from ring dimension
131,072 and depth 43 to ring dimension 65,536 and depth 16 for the first complete block. The block
passed at global relative error `3.32e-10`.

The observed encrypted-evaluation time was 372 seconds, compared with 2,462 seconds for the pure
baseline. This is an observed architectural comparison, not a controlled paired benchmark: the runs
used different parameter sets and occurred in different shared-host windows. Its reliable meaning is
that removing encrypted polynomial nonlinearities enabled a substantially smaller circuit while
preserving accuracy.

Boundary batching later reduced the same block's physical client crossings from 24 to seven.

### General attention and sequence growth

Exact causal attention at the client boundary passed at increasing sequence lengths:

| Tokens | New boundary established | Global relative error |
|---:|---|---:|
| 3 | First general causal-softmax block beyond the two-token identity | `3.65e-10` for the complete block |
| 8 | General attention plus corrected arbitrary-group output packing | `3.69e-10` for attention |
| 32 | Head-distributed score packing and a complete block | `2.48e-10` for the complete block |
| 103 | Fixed-width chunked causal softmax beyond the prior packing ceiling | `3.66e-10` through attention projection |

The 103-token row is the complete prompt length for the GSR classification task. At this stage it
covered block 0 through attention projection, not the MLP or task head.

### Complete task-length block

Token-SIMD packing then carried the entire real-weight block through all remaining operations.

| Property | Measurement |
|---|---:|
| Model boundary | released block 0, complete graph |
| Sequence length | 103 tokens |
| Hidden width / heads | 768 / 12 |
| Token lanes per ciphertext group | 8 |
| Dense products | 156 |
| Serial-equivalent dense products | 1,236 |
| Structural dense-product reduction | `7.92x` |
| Minimum demonstrated multiplicative depth | 13 |
| Relative error across completed depth-13 samples | approximately `4.1e-9` to `5.0e-9` |
| Correctness criterion | `4e-2` |
| Physical client crossings | 857 |
| Logical boundary instances | 129,162 |
| Target-process peak GPU memory | 9,834 MiB, about 9.6 GiB |

Accuracy is not the current single-block constraint. The measured errors are millions of times below
the predeclared criterion. The cost is interaction and linear algebra: many logical nonlinear values
are batched into 857 physical crossings, while dense encrypted maps dominate server work.

### Short multi-block composition

Released blocks 0 and 1 passed sequentially at two tokens using a full-hidden-state client refresh
between blocks. Relative error after block 1 was `8.48e-11`. Target-process GPU memory peaked at
10,872 MiB during block 0 and did not increase in block 1.

This shows that the client-refresh mechanism prevents the pure baseline's per-block depth and memory
accumulation at the tested shape. It does not establish the same property at 103 tokens or across all
twelve blocks.

## Optimization results

### Retained

| Change | Evidence | Qualification |
|---|---|---|
| Boundary batching | Reduced physical client crossings without changing logical functions | Used by later block gates |
| General and chunked causal softmax | Passed through full GSR prompt length | Exact at client boundary; interactive |
| Eight-token SIMD packing | `7.92x` fewer dense products for the complete 103-token block | Structural count is stronger than wall time |
| Depth 13 | Smallest demonstrated passing setting | No validated speedup over depth 16 |
| CPU-side diagonal-vector cache | More than `99.99%` cache hits, unchanged target-process GPU memory, correct output | Timing benefit unresolved |
| Cross-process context/key reuse | Correct results after reloading cryptographic state | Avoids setup, which is a small share of full-block time |
| Two-GPU Q/K/V split | Both workers matched the real oracle | One clean, unrepeated sample showed about `1.65x` improvement for this substage |
| Cache plus two-GPU Q/K/V split | Combined configuration passed | Correctness-only result under host contention |

### Rejected or unresolved

| Attempt | Outcome | Consequence |
|---|---|---|
| Reuse GPU-resident plaintext objects | Six reproducible crashes, passing unmodified controls | Abandoned in favor of a CPU-vector-only cache |
| Native batched linear transform | Correct but `1.6–1.9x` slower at the converted call site | Not propagated to the other dense maps |
| Warm-up prelude | Correctness passed; timing confounded by worsening host load | Hypothesis remains unresolved |
| Four-token SIMD width | Worse in every counted operation category | Rejected without spending GPU time |
| Two-process MLP partial-sum split | Merge math passed, but the pinned library cannot transport ciphertexts between processes | Dropped for the current backend |
| Depth-13 speedup | Initial sample looked faster; two repeats were much slower | Speed claim retracted; shared-host CPU contention dominated |

## Timing contamination

The long-running measurements were performed on a shared host. Target GPUs could pass an idle
preflight and later acquire co-tenants. More importantly, host-wide CPU load sometimes rose from
single digits to hundreds while a run was active. CPU preparation of plaintext diagonals and encoding
therefore slowed encrypted GPU work even when the target process's GPU memory footprint was stable.

Completed depth-13 task-length block variants ranged from 2,986 to 13,658 seconds. Cached runs with
identical cache behavior differed by more than a factor of two solely under different ambient load.
Uncached repeats also reversed the apparent depth-13 advantage over depth 16.

The paper may use these observations to establish that the current implementation takes tens of
minutes to hours on the shared system. It must not select the fastest sample as a benchmark, claim a
repeatable cache speedup, or multiply one contaminated block time by twelve and present the result as
end-to-end latency.

Trustworthy performance evidence is narrower:

- exact operation-count reductions;
- target-process memory when measured separately from device-wide co-tenants;
- comparisons between sibling call sites in one run; and
- the single-sample, substage-only two-GPU concurrency observation with its qualifier.

## Optimization audit and execution decision

The retained implementation is not yet optimization-complete. The current block performs 156 dense
products and 177,734 ciphertext–plaintext multiplications, but the four packed copies always carry the
same dense transform. A derived, unmeasured schedule could use distinct public diagonals per copy to
compute Q/K/V and MLP chunks concurrently, reducing dense products to roughly 52 before mask and
copy-repair overhead.

Three other exact-model opportunities remain unresolved:

- compute complete LayerNorm at its existing client boundary, return a fresh normalized ciphertext,
  and fuse inter-block refresh with the following normalization;
- replace separate per-head attention reductions with segmented reductions or compact matrix layouts;
  and
- reuse encoded plaintext weights safely rather than caching only their host vectors.

Wider B=16/B=32 layouts and stage-specific contexts also require operation-count screening. These are
research hypotheses, not measured improvements.

The execution decision separates correctness from performance. The existing 12-block driver may run
now to establish arithmetic composition. Before it supplies an optimized latency result, the project
must obtain a dedicated current-block profile, test the remaining gates individually, integrate the
retained changes, and rerun the complete classifier. A separate networked experiment is required for
end-to-end protocol latency because current client crossings occur inside one process.

## What has not been proved

- **Complete encrypted DNAGPT:** all twelve blocks and the released GSR head have not been executed
  together under the client-assisted protocol.
- **Task-level accuracy:** no encrypted 103-token prediction has reached the released head and been
  compared with the frozen GSR label or logits.
- **Task-length composition:** the two-block refresh passed only at two tokens.
- **Private token lookup:** the client currently computes embeddings before encryption.
- **Non-interactive inference:** the main protocol requires the key-owning client at every declared
  nonlinear and block-refresh boundary.
- **Clean latency:** no sustained uncontaminated full-block or full-model benchmark exists.
- **Production security:** transport, authentication, key custody, malicious behavior, traffic
  leakage, and side channels are outside the prototype.
- **Clinical or biological claims:** the work evaluates computational privacy and model fidelity; it
  does not establish clinical validity.

## Current conclusion

The work has moved the feasibility boundary from isolated encrypted operators to a complete
real-weight DNAGPT block at the full GSR prompt length. Client assistance removes the nonlinear depth
and calibration burden that stopped pure CKKS composition, while token packing makes the sequence
length representable on one A100-class GPU.

The twelve-block task-length driver makes the remaining correctness experiment concrete. Until it runs
through the released head and matches the plaintext task oracle, the correct conclusion remains
**task-length single-block feasibility**, not complete private DNAGPT inference.

Performance has a second boundary: the current driver is a baseline, not an optimization-complete
implementation. Clean profiling and the unresolved exact-model gates must precede any final optimized
latency claim.
