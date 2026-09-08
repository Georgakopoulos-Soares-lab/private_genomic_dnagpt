# Paper-figures review — 2026-09-07

## Scope

Audited the journal-facing figure generator against the 2026-09-07 evidence ledger and regenerated
the eight figures that have admissible inputs. The review target was IEEEtran at final printed size:
3.5 inches for a column figure and 7.16 inches for a two-column figure. The block telemetry figure
was deliberately not regenerated or approved for the manuscript because its trace still belongs to
the earlier standalone-block run; the ledger does not provide a stage trace for the verified complete
classifier.

## Repairs

- `fig_graphical_abstract.pdf`: removed the open-provenance 14 GB client-memory claim. It now reports
  the measured 6,683 s complete-classifier wall time, the measured head-margin error, and label
  agreement. The provider-view sentence uses the supported no-plaintext-activation/no-secret-key
  claim.
- `fig_architecture.pdf`: replaced “sees only ciphertexts” with the supported provider view and
  corrected the second LayerNorm path: encrypted statistics remain server-side and only inverse
  square root crosses to the client. Sequence length, schedule, and timing exposure are explicit.
- `fig_packing.pdf`: regenerated at column size, retained the ledger-derived 32,768-slot layout and
  103-to-13 token grouping, and states the verified single padded lane. The feature-padding text now
  includes attention-score staging.
- `fig_waterfall.pdf`: removed the unsupported 7,560 s/11.6x/26 h baseline entirely and repurposed
  the figure as a measured complete-run decomposition. It shows the 6,683 s process wall time and
  two independent decompositions of the 6,594.748 s encrypted-evaluation interval: by phase and by
  executor. It does not infer an uninstrumented remainder.
- `fig_memory.pdf`: rebuilt directly from the measurement rows using binary MiB-to-GiB values:
  24.9, 12.1, and 47.5 GiB by stage, with 60.4 GiB target-process RSS at the failed unbounded-cache
  allocation. Labels now say target-process host RSS rather than implying machine-wide occupancy.
- `fig_cost_split.pdf`: moved from the old standalone-block timing to the verified complete
  classifier. The bar is explicitly the measured 6,594.748 s encrypted-evaluation interval, split
  into 5,639 s provider work (85.5%) and 956 s in-process client-boundary work (14.5%). The figure
  says network transport was not measured and contains no client-memory claim.
- `fig_scaling.pdf`: plots the measured 6,683 s complete classifier as a solid bar and the three
  shorter-task twelve-block sums as hatched projections. Every bar is labelled “measured” or
  “projected” in the plot itself. Removed the incorrect 48,000-tile annotation; mRNA abundance is
  shown only as outside the encrypted scope.
- `fig_baseline.pdf`: gives each comparison pair the same precision (four decimals for signal
  recognition and three decimals for the other pairs), preventing rounding from changing the
  apparent gap.

The generator now uses shared `MEASURED_KW` and `PROJECTED_KW` styling, publication-width constants,
and ledger-tag assertions for result figures. Searches of `figures.py` find no remaining 14 GB client
RAM, 11.6x, 26-hour, 48,000-tile, “sees only ciphertexts,” or host-key-release attribution.

## Visual and technical checks

Rendered review PNGs for all eight regenerated figures and inspected a contact sheet plus the
column-width cost split individually. Corrected two visual defects found during inspection: merged
memory-stage labels and overlapping executor notes. The final views have no observed label/bar or
legend collisions; measured versus projected scaling remains separable by fill, outline, hatch, and
explicit text, so the distinction survives grayscale.

The scripts pass Python byte-compilation. The generated PDFs are single-page vector assets with
embedded subset CID TrueType fonts. Their bounding boxes are close to their intended publication
widths after wrapping long subtitles.

## Manuscript handoff

The active `fig_packing` inclusion was a single-column `figure` using `width=\textwidth`; under
IEEEtran that requests the full two-column text width and can overflow. The manuscript owner was
notified to use `width=\columnwidth`, or to change the float to `figure*` if a two-column placement
is intended. No manuscript source was edited in this figure pass.

## Follow-up restoration — manuscript-owner integration

All nine assets are now included, superseding the earlier one-figure manuscript handoff. Compact
artwork was prepared for the full figure set and completed during manuscript integration. Overview
and architecture are wide, shallow two-column figures; the other seven are column-width figures.
The final review corrected crowded architecture labels, baseline metric annotations, and waterfall
axis/footer spacing. The rendered figures and their captions were inspected at manuscript scale.

The timeline is no longer the historical standalone-block utilization trace. It plots all twelve
measured block times and both error series from `telemetry.yaml:complete_run`; all triples match
the immutable complete-run JSON. The historical trace remains in the ledger but is not plotted.

Scaling is now anchored to the measured 6,683-second complete-program wall time, multiplied by
ciphertext-group count divided by thirteen. Hatched projections retain an explicit fixed-circuit,
fixed-head, group-linear assumption; separate quadratic-attention and fixed-cost models are not
claimed. Final bar labels are rounded to seconds: 1,028, 3,599, 4,627, and 6,683. The earlier
fixed-circuit block-sum wording above has been replaced in both artwork and caption.

Final inclusion map: overview p. 3; architecture and packing p. 6; memory and plaintext baseline
p. 8; block trajectory p. 9; measured time and executor split p. 10; sequence scaling p. 11.
All nine are vector PDFs with embedded fonts. The lint now reports generated-but-unincluded
figures to catch this class of omission during later restructuring.

## Figure 2 border-padding correction

Visually confirmed the reported defect in the existing architecture asset: the MLP
down-projection label crossed its box borders, and several other labels approached them too
closely. Rewrapped the long labels, including a three-line MLP down-projection label, and
increased box height from 24 to 26 canvas units. The physical figure size remains 7.16 by
2.25 inches; the twelve operation nodes, their evaluation order, executor colors, and original
6.6-point label size are preserved.

Added a rendered-text bounding-box check to `fig_architecture`: each label must have at least
3 points of clearance from every box border at the final print width. Regenerated only
`fig_architecture.pdf`; this check passes for all twelve nodes. Inspected both the generated
220-dpi PNG and an independently rasterized PDF at 1,800 pixels wide. No label overflow or
border collision remains, and all connecting arrows retain their correct direction. The
manuscript owner will rebuild and verify the full paper separately.
