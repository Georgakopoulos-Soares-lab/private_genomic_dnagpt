# Paper submission TODO

This file records work that cannot be completed from the committed Markdown and source code alone.
The manuscript currently withholds the affected claims and excludes the affected generated figures.
Do not restore them until the supporting artifacts have landed and the evidence ledger has been
updated by an author.

## Submission blockers

### Commit clean timing evidence

The exact timing values currently in the evidence ledger trace to
`docs/hybrid/t123_walkthrough.md`, which identifies the run as a single shared-partition sample.
They therefore do not satisfy the manuscript rule that only clean dedicated-node timings may appear.

For both the documented pre-optimization baseline and the retained implementation, commit:

- the immutable result JSON and complete run log;
- target-process telemetry used for host and accelerator memory;
- allocation evidence showing an uncontended, dedicated node;
- the executable configuration and environment summary needed to reproduce the run;
- multiple repetitions, together with the declared summary statistic; and
- a paired comparison confirming that the circuit, packing, depth, weights, prompt, boundary
  schedule, and acceptance criterion are identical.

The same standard applies if the thread-affinity and NUMA-local allocation deltas are to be reported.
The current repository does not contain the paired artifacts for the claimed server, client, and
wall-time changes. The client-side peak-memory value also needs its cited run before it can return to
the manuscript or figures.

After these artifacts land, update the evidence-ledger source fields to the immutable files. Do not
cite the walkthrough as dedicated-node timing evidence.

### Correct ledger entries

These corrections were identified from committed Markdown and arithmetic checks. They are listed
here because the manuscript workflow prohibits this drafting pass from editing
`paper-docs/evidence/*.yaml`.

- Host-memory units: the source records stage values of `25,530`, `12,434`, `48,610`, and `61,863`
  MiB. The ledger currently treats decimal conversions as GiB. Either retain the exact MiB values or
  convert them consistently to `24.9`, `12.1`, `47.5`, and `60.4` GiB.
- Long-sequence causal tiles: `220` ciphertext groups use the lower-triangular schedule, so the
  derived count is `220 * 221 / 2 = 24,310`, not the value currently stored in the scaling ledger.
- Fixed-circuit twelve-block sums: `320 * 12 = 3,840` seconds and
  `652.545 * 12 = 7,830.54` seconds. The stored full-pass seconds do not recompute from their stated
  per-block inputs.
- Timing decomposition: the instrumented top-level phases leave `3.157` seconds between their sum
  and the recorded wall time. Add a sourced overhead row or present the phases explicitly as an
  incomplete instrumented decomposition.
- Projection status: change the pre-campaign full-pass endpoint from verified to derived. A
  twelve-block sum at fixed circuit is a projection, not a measured complete inference.
- Wall-time scope: correct the reversed wording to process start through process end once a valid
  timing source is available.

### Regenerate evidence-dependent figures

Do not edit the PDFs manually. Correct the ledger and figure generator, then regenerate and inspect
the assets.

- `fig_graphical_abstract.pdf`: remove unsupported timing and client-memory values; identify the
  numerical comparison as one block against the validated plaintext reference.
- `fig_cost_split.pdf`: wait for clean timing evidence, reconcile the uninstrumented remainder, and
  replace the provider-view overstatement.
- `fig_timeline.pdf`: wait for clean timing evidence and corrected host-memory units.
- `fig_scaling.pdf`: recompute the causal-tile count and all fixed-circuit sums; distinguish a
  per-block series from any future measured complete-model result.
- `fig_waterfall.pdf`: wait for two supported timing endpoints; remove host-key-copy release from
  the measured campaign because it was not present in the retained endpoint.
- `fig_memory.pdf`: regenerate from consistently expressed MiB or GiB values.
- `fig_architecture.pdf`: replace “sees only ciphertexts” with the supported provider view: the
  provider receives no plaintext query-derived activation and no secret key, but it does observe
  model weights, public and evaluation keys, sequence length, message sizes, schedule, and timing.
- `fig_baseline.pdf`: use the same display precision as Table 1 for both members of each comparison.

The graphical abstract, architecture, cost split, timeline, scaling, waterfall, and memory figures
remain excluded from the LaTeX source until these repairs are complete. The baseline figure remains
included because its corrected DNAGPT comparison is supported; only display precision remains.

## Restore claims after evidence lands

If the new artifacts pass the dedicated-node requirement, restore timing prose section by section
and rerun the numeric checker after each section. The abstract, Key Points, Introduction,
Optimization, Results, Practicality verdict, and Conclusion must all use the same supported endpoint
and display precision. Label every twelve-block fixed-circuit sum as a projection in the sentence
where it first appears. Do not infer networked latency, throughput, laptop suitability, or complete
classifier cost from an in-process one-block measurement.

## Complete-model and deployment evidence

- Run the built twelve-block driver and released task head at the task-defined prompt, then compare
  the encrypted prediction with the frozen plaintext reference.
- Add serialized client/server transport and measure payload size, phase dependencies, bandwidth
  sensitivity, and transport latency.
- Resolve private token-index lookup or keep it outside the claimed encrypted service boundary.
- Select and validate a noise-flooding budget for client re-encryption.

## Author and production tasks

- Replace the corresponding-author email placeholder.
- Supply the biographical note and acknowledgments.
- Decide and document the repository reuse license.
- Rebuild the manuscript when a TeX toolchain is available and inspect all pages, floats,
  references, and captions.
- Resolve the current Tectonic font error before submission. The isolated build stops at
  `preamble.tex:18` because the installed Latin Modern package has no `T1/lmr/b/sc` shape for the
  bold small-caps section style. Test the replacement against the target journal toolchain rather
  than accepting silent font substitution.
- Run `paper-docs/.venv-paper/bin/python paper-docs/scripts/check_numbers.py` after every ledger,
  figure, or prose restoration.

## Resolved in the current draft

- The baseline table now uses DNAGPT's own `0.9151` signal-recognition result and states the split
  mismatch. The incorrectly attributed DeepGSR number is no longer used.
- The manuscript reports host-memory observations in their source MiB units.
- Unsupported timing, speedup, utilization, affinity-delta, client-memory, and whole-model timing
  projections have been removed from active manuscript prose.
