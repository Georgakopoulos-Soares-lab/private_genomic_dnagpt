# Full-manuscript evidence audit — 2026-09-07

Scope: the rewritten IEEE/JBHI LaTeX manuscript, its included packing figure, and the four paper
evidence ledgers, checked against the canonical repository sources after the complete twelve-block
plus genomic-signal-head result was added. No manuscript or evidence file was edited during this
audit. `python3 paper-docs/scripts/check_numbers.py` passes all four mechanical checks: banned terms,
unledgered numbers, citation keys, and structure.

## Findings

| Location | Claim as written | Problem | Fix |
|---|---|---|---|
| `07_results.tex:46`; `evidence/measurements.yaml:30-78` | The one-block row reports a measured wall time of `663 s`. | **Prohibited timing provenance.** The cited source, `docs/hybrid/t123_walkthrough.md:751-753`, identifies this exact value as one sample on a shared-partition node and says it does not satisfy the project's latency-claim standard. The paper contract excludes timing measured under shared-host contention. The newer clean complete-classifier timing does not retroactively validate this older one-block timing. | Remove `663 s` from the one-block row, or replace it only with a separately documented clean one-block measurement. Correctness, depth, operation counts, and process memory from the run may remain. |
| `07_results.tex:24-30` | “The blockwise errors show no accumulating drift across the refreshes.” | **Stronger than the ledger and the trace.** The ledger records only ranges, not a verified no-trend result. In the run record, worst-token error rises from `4.39e-9` after block 1 to `1.43e-7` after block 12, and absolute error also rises; all values pass comfortably, but one trajectory does not support a general no-drift claim. | State the supported result: every block output stayed within the predeclared tolerance, with worst-token relative error at most `1.54e-7`. If a trend claim is wanted, define and ledger a trend statistic first. |
| `evidence/scaling.yaml:5-9,116-121` | “DNAGPT tokenizes overlapping 6-mers, so a window of B base pairs becomes B/6 tokens.” | **Wrong derivation wording in the ledger.** The upstream tokenizer uses non-overlapping chunks with stride six (`tokenizer.py:67`), including a final short chunk. The general formula is `ceil(B/6)`, not `B/6`; this distinction produces the correct 12-token and 67-token rows for 70 bp and 400 bp. The manuscript's displayed task counts are correct, but its source rule is not. | Replace “overlapping” with “non-overlapping” and use `ceil(B/6) + special tokens` as the general derivation rule. |
| `06_optimization.tex:51-56,124-128,138-150`; `07_results.tex:86-90` | The cache “removes 90%”; masks are encoded afresh at `17,400` uses; the proposed layout lowers the total by `60%`; the within-run spread is `4.5%`. | **Approximate/derived values are printed as exact.** The implementation-ground-truth count gives a 92.3% cache-hit/removal fraction, the mask count is exactly 17,448, and the proposed total reduction is about 59.9%; their ledger text says “approximately” or “roughly.” The 4.5% spread is also underspecified: `(max-min)/max = 4.486%`, whereas division by the mean gives 4.583%. | Use “about 90%,” either “17,448” or “about 17,400,” and “roughly 60%.” Define the spread denominator and use matching rounding (or say “about 4.5%”). |
| `evidence/measurements.yaml:18-24`; `evidence/optimizations.yaml:14-18` | The ledger metadata calls the hardware “allocated as a dedicated node.” | **Provenance scope is broader than the canonical full-run record.** The accepted result documents a clean target A100 and low host load, but also records another job on a different GPU of the same physical node. “Telemetry-confirmed clean single-A100 execution” is supported; physical-node exclusivity is not. The manuscript already uses the supported wording. | Change the ledger metadata to the documented clean single-A100/32-core allocation and avoid “dedicated node” unless allocation evidence establishes whole-node exclusivity. |

## Independently recomputed arithmetic

- Complete encrypted evaluation: `6565.992935047 + 9.045895324 + 19.709652827 =
  6594.748483198 s`, exactly matching the ledger.
- Work split: `5638.561475474 + 956.187007724 = 6594.748483198 s`; the corresponding
  shares are 85.5008% and 14.4992%, correctly printed as 85.5% and 14.5%.
- Complete wall time: `6683 / 3600 = 1.8564 h`, correctly printed as 1.86 h.
- Within-run block timing: the twelve values sum to `6565.992935047 s`, average
  `547.1661 s`, and range from `533.8692` to `558.9433 s`; the table/body rounding is otherwise
  consistent.
- Complete crossings: `12 * 857 + 11` inter-block refreshes `+ 1` last-token refresh `+ 3`
  head nonlinearities `= 10,299` physical crossings.
- Complete logical instances: `12 * 129,162 + 11 * 103 + 1 + 3 = 1,551,081`.
- Dense products: `12 * 156 = 1,872` block products, plus one measured head product.
- Packing: `1236 / 156 = 7.9231`, supporting the printed `7.92x` structural reduction, not a
  latency speedup.
- Boundary decomposition: `128,544 / 129,162 = 99.5215%`, correctly printed as 99.5%.
- Prompt arithmetic under the actual tokenizer is `ceil(bp/6) + specials`: 70 bp maps to 13
  total tokens, 300 bp to 51, 400 bp to 68, 600 bp to 103, and 10,500 bp to 1,755. Then
  `ceil(1755/8) = 220` groups and `220 * 221 / 2 = 24,310` causal tiles.
- Memory units are now coherent: `9,839 MiB = 9.608 GiB`; `50,930 MiB = 49.736 GiB`, supporting
  the printed 49.7 GiB host peak. The single-block stage values 24.9, 12.1, 47.5, and 60.4 GiB
  also match binary conversion of their source MiB readings.

## Verified clean

- The complete twelve-block plus released-head result is presented as one measured held-out-input
  execution, not as encrypted test-set accuracy or a repeat-run benchmark.
- The `6,683 s` complete-program measurement is consistently distinguished from the
  `6,594.75 s` encrypted-evaluation subtotal and from the per-block mean.
- The final label, head-margin error (`8.56e-9`), block-error maxima (`2.07e-8` global and
  `1.54e-7` worst-token), and predeclared `4e-2` tolerance agree with the immutable full-run record.
- The complete process peaks are consistently reported as 9,839 MiB GPU and 49.7 GiB host; the
  GPU value is correctly identified as target-process memory rather than device-wide occupancy.
- One-sample language is present in the abstract, Results, Limitations, and Conclusion. Independent
  repetition and variance remain explicitly unresolved.
- Client crossings are consistently identified as in-process cryptographic operations, not network
  messages; transport, serialization, and production key custody are not inferred from them.
- The encrypted boundary is consistently scoped to embedded vectors. Private token-index lookup is
  excluded in the abstract, threat model, Results, Limitations, and Conclusion.
- Model confidentiality is not claimed; the compute-provider guarantee is limited to receiving no
  plaintext query-derived activation or secret key under the semi-honest boundary.
- The unsupported 14 GB client-memory value and 10% latest-client-share value do not appear.
- The old 11.6x factor, its 7,560-second denominator, and the 26-hour projection do not appear in the
  manuscript or included figure.
- Depth 13 is described as the smallest demonstrated passing depth and tied to level-0 client
  re-encryption, never to a measured speed advantage.
- The complete operation counts and the one-block physical/logical boundary counts are not
  conflated; 7.92x is correctly labelled an operation-count reduction.

## Figure-restoration follow-up

The independent restoration verifier confirmed that the twelve time/global-error/worst-token-error
triples in the new complete-run trajectory exactly match the accepted JSON. It also verified the
token and group counts and the recomputed clean-anchor group-linear scaling arithmetic, with
measured and assumed values distinguished. Its finding that the older shorter-task projections
inherited a contaminated standalone-block timing anchor was resolved before the final render.

The verifier checked that relocation preserves the algorithm bodies: the short routines are
unchanged, and splitting the query/key/value assignment across two lines preserves the block
operations. Algorithm numbering now follows explanation order: boundary, LayerNorm, matrix product,
and complete block. No operation, result, or citation was removed to achieve the layout.

During final integration, the manuscript owner independently recomputed the twelve trajectory
triples against the immutable result and every scaling row against the clean measurement ledger.
All comparisons passed. The figure-source mappings and final PDF were then checked by the
manuscript owner: all nine assets are present, and the captions retain one-input/one-run,
standalone-block-memory, in-process-client, and unmeasured-alternative-head distinctions.
