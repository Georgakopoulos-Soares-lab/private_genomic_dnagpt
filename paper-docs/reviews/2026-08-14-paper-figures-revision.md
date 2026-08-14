# Paper-figures revision — 2026-08-14

## Outcome

The bounded figure revision is complete. Seven retained figures were regenerated as vector PDFs;
the heterogeneous plaintext-baseline chart was unregistered and its committed PDF removed. No
experimental job or measurement was rerun.

## Changes

- **Graphical abstract:** scopes the input to a genomic fragment, shows local tokenization and
  embedding lookup before encryption, labels the transmitted object as embedded vectors, and marks
  the result as one measured 103-token execution. The client percentage now names encrypted
  evaluation as its denominator.
- **Architecture:** mirrors encrypted LayerNorm statistics and client inverse square root at both
  LayerNorm sites. It distinguishes a validation-only output decryption from the full-hidden-state
  decrypt/re-encrypt used for composition.
- **Packing:** names the genomic-signal task and separates 768 active features from the remaining
  power-of-two padding and attention-staging region.
- **Memory:** identifies the bars as the bounded-cache one-block run and the red line as a separate
  unbounded-cache attempt killed by the operating system.
- **Complete-run trace:** uses descriptive underutilization and bounded-memory language, removes the
  causal bottleneck claim and the unrelated unbounded-cache threshold, and states that the plotted
  curve is thinned from the full 653-sample trace.
- **Cost split:** removes the clipped client annotation, uses encrypted evaluation as the bar
  denominator, and labels the client bucket as declared boundaries plus refreshes rather than as
  nonlinearities alone.
- **Sequence structure:** names the plotted metric as the causal score-tile schedule and explicitly
  says it is neither total operation count nor a timing model.

Captions were changed only where needed to carry these scopes. Figure canvases and type were reduced
to approximately their final manuscript width, with text cut before resizing.

## Verification

- Regenerated all seven retained PDFs from `scripts/figures.py`.
- Inspected review PNGs for every retained figure; corrected graphical-abstract and cost-split label
  collisions found on the first pass.
- Rebuilt the manuscript with Tectonic.
- Ran the four manuscript checks: banned terms, evidence-backed numbers, citations, and structure
  are all clean.
- Tectonic continues to emit pre-existing invalid-UTF-8 warnings from `algorithm.sty` and
  `lineno.sty`; the PDF is produced successfully.

The latest figures contain no observed clipping, overlap, or causal performance claim. Complete-PDF
float placement and page-level QA remain part of the coordinating manuscript pass.
