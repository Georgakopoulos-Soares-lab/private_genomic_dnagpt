# Paper-figures review — 2026-08-14

## Verdict

The visual argument is coherent and mostly worth keeping. The figures use a consistent visual
language, the committed PDFs are vector graphics with embedded fonts, and most plotted values are
correctly coupled to the evidence ledgers. The resource trace, memory figure, packing diagram, and
cost split add real academic value rather than decoration.

The manuscript is not submission-ready, however. Four issues should be fixed before any broader
polish: the protocol algorithms float behind the bibliography; the architecture figure assigns the
second LayerNorm to the wrong party; the cost-split label is visibly clipped; and the graphical
abstract implies that the genome itself enters the encrypted boundary. The baseline comparison
figure also needs either substantial qualification or removal. Fix those, make the figures legible
at their actual placed size, and stop. The remaining observations are improvements, not reasons to
delay the paper.

## Material inspected

- The complete 25-page `manuscript/source/main.pdf`, rendered page by page at 120 dpi.
- All eight current figure PDFs, rendered independently at 180 dpi in colour and 120 dpi in
  grayscale: graphical abstract, plaintext baseline, architecture, packing, host memory, complete
  resource trace, cost split, and sequence-length scaling.
- Figure placement and captions in `manuscript/source/sections/*.tex`.
- `scripts/figures.py`, `scripts/figstyle.py`, and the four evidence ledgers.

No figure or manuscript source was changed. The unregistered waterfall generator was not treated as
a current manuscript figure. Its omission is scientifically correct while the pre-optimization
timing endpoint lacks committed dedicated-node evidence.

## Submission blockers

### 1. Algorithms 1–4 render after the bibliography

The four algorithms are declared in Section 5 (`sections/05_protocol.tex:82–187`) but render on
pages 24–25, after the references end on page 23. Page 23 is mostly empty, Algorithm 1 and Algorithm
2 occupy page 24, and Algorithms 3 and 4 are separated by a large blank region on page 25. Readers
encounter prose that depends on the algorithms roughly fifteen pages before the algorithms
themselves.

This is a submission defect, not cosmetic whitespace. Keep the algorithms inside Section 5 and
before Section 6. A float barrier alone may still leave Algorithm 1 too tall; the likely robust fix
is a breakable algorithm presentation or a deliberate split of Algorithm 1, followed by a barrier
before leaving the protocol section. Recheck the final PDF rather than trusting float specifiers.

Float placement elsewhere also needs one quick pass: Figure 1 interrupts the Key Points list
between pages 1 and 2, and Figures 6 and 7 appear before the subsection headings that introduce
them. These are less serious than the algorithm backlog but stem from the same uncontrolled float
queue.

### 2. Figure 3 assigns the second LayerNorm to the client

`scripts/figures.py:263` places “LayerNorm, second occurrence” in the data-owner column. The actual
protocol mirrors the first LayerNorm: the compute provider evaluates the mean, centered values, and
variance on ciphertexts; only inverse square root is client-evaluated
(`sections/05_protocol.tex:156–169`). The figure therefore contradicts both Algorithm 3 and the
caption's statement that the provider evaluates every linear operation.

Mirror steps 2–3 for the second LayerNorm, or label the client step narrowly as “inverse square
root, second occurrence.” The final “Decrypt the block output” box should also distinguish the
single-block validation readout from the full-hidden-state decrypt/re-encrypt refresh used between
blocks. The level-0 refresh is central to the paper and the current diagram does not show it.

### 3. Figure 7 contains clipped text

The data-owner segment reads “14.3% of wal…” in both `fig_cost_split.pdf` and the placed figure on
page 16. The source assumes that in-segment labels cannot collide (`scripts/figures.py:848–877`),
but the client segment is too narrow. This is the only clear rendering failure in the eight figure
PDFs. Move the percentage outside the segment, shorten it, or use a callout.

### 4. Figure 1 overstates the encrypted input boundary

The graphical abstract says the data owner has “a human genome” and “sends it encrypted”
(`scripts/figures.py:148–150, 184–195`). The evaluated boundary begins after tokenization,
token-index lookup, and embedding construction. The arrow can therefore be read as encrypted
token-index lookup or end-to-end encrypted genomic inference, neither of which was evaluated.

Show the missing local step explicitly: “tokenize and embed locally,” then “send encrypted embedded
vectors.” Also replace “a human genome” with “a genomic sequence” or the measured 600-base-pair task
window. The bottom-right 14.5% needs the denominator (“of encrypted evaluation”), and the figure or
caption should say that the 1.86-hour result is one measured execution at the 103-token
genomic-signal prompt.

### 5. Figure 2 visually overstates comparability

The paired bars look like direct replications of matched published baselines, but the comparisons
are heterogeneous:

- polyadenylation-signal accuracy is the same model class on a different split;
- promoter and splice-site values are locally fine-tuned DNAGPT-backbone results compared with
  DNABERT-2;
- abundance regression compares different pretrained backbones and is outside encrypted scope.

Table 2 states these qualifications; Figure 2 and its caption do not. The figure also prints the
DNAGPT reference 0.9151 as 0.92 while printing this work as 0.912
(`scripts/figures.py:1123–1142`), an asymmetric precision choice that makes the small difference
look larger.

The efficient fix is to remove Figure 2 and keep Table 2, which is more informative and already
carries the qualifications. If the figure stays, identify each comparator in the graphic, retain
source-appropriate precision, and encode the non-comparability visibly rather than in surrounding
prose. Its orange “outside the encrypted scope” text is also low-contrast in grayscale.

## High-value revisions

### Design at final print size

The standalone PDFs are clear on screen, but most are generated at widths of roughly 10–11 inches
and then reduced to the manuscript's approximately 7-inch text width. Many 9–10 point annotations
therefore land near 6 points on the page. This is visible in Figures 1, 3, 4, 6, 7, and 8; the
timeline legend and annotations and the scaling footnote are particularly small.

Use the final placed dimensions as the design canvas, target at least 7–8 point text after scaling,
and cut annotations before enlarging the canvas. Vector output keeps the glyphs sharp but does not
make physically small type readable.

### Narrow Figure 6's claims to what the trace shows

“Twelve blocks cost what one block costs” can be read as a latency claim, although the evidence is
about memory. “The GPU is never the limit” and the caption's “not accelerator-bound” are also
stronger causal claims than an utilization trace alone supports. Low utilization is evidence of
headroom; it does not exclude memory-bandwidth, launch, synchronization, or dependency bottlenecks.

Use a title such as “GPU memory stays flat while host-memory stages repeat across twelve blocks,”
and state the utilization result descriptively. The plotted curve is a 73-point thinning of a
653-sample trace; the maximum line is correctly labelled as coming from the full trace, but this
should also be stated in the caption. Replace the hard-coded `51` and `653` at
`scripts/figures.py:700–706` with structured ledger fields.

### Name the metric in Figure 8 instead of calling it total circuit size

The bars show lower-triangular causal score-tile count, one component of the circuit. “Circuit size”
is broader and may be read as total operations, ciphertexts, or gates. Rename the title and caption
to “causal attention score schedule” or “score-tile count.” The measured-versus-not-timed visual
distinction is otherwise excellent: solid fill, hatch, labels, and the caption all agree that only
the 103-token task has a measured complete inference.

### Complete Figure 4's packing explanation

Figure 4 accurately gives 32,768 slots, four copies, 1,024 padded features, eight lanes, 13 groups,
91 causal score tiles, and one padded lane. It should also show or state that only 768 features are
active, while the remaining feature region provides power-of-two padding and per-head attention
staging. The current sentence that the four copies exist because of the 4× MLP is correct but
incomplete (`scripts/figures.py:403–409`). The caption should name the genomic-signal task when it
calls 103 tokens “the prompt.”

### Make Figure 5's two configurations explicit

The three bars are measured stage RSS values for one 103-token block with bounded caching. The
61,863 MiB line is the observed RSS in a separate unbounded-cache attempt that was killed; it is not
an operating-system threshold. State both scopes in the caption. The figure's mechanism and
placement are otherwise strong.

## Per-figure disposition

| Figure | Academic role | Disposition |
|---|---|---|
| 1. Graphical abstract | Deployment premise and headline outcome | Keep; correct the embedded-vector boundary and identify the single measured run. |
| 2. Plaintext baseline | Model-fidelity comparison | Prefer removal; Table 2 carries the evidence with necessary qualifications. |
| 3. Architecture | Party allocation and protocol order | Keep; correct the second LayerNorm and show the inter-block refresh distinctly. |
| 4. Packing | Explains why 103 tokens become 13 groups and 91 tiles | Keep; add active-width/staging detail and task scope. |
| 5. Host memory | Shows why stage flushing makes caching executable | Keep; scope the bars and failure line to their separate runs. |
| 6. Resource trace | Strongest empirical systems figure | Keep; soften the causal title, state thinning, and enlarge labels. |
| 7. Cost split | Quantifies the deployment burden on each party | Keep; fix clipping and preserve the wall-versus-evaluation denominators. |
| 8. Sequence structure | Shows task-dependent attention schedule | Keep; call the metric score-tile count rather than total circuit size. |

## Evidence, measurement status, and accessibility

- Current plotted values agree with the ledger/source data. The main coupling exceptions are the
  hard-coded complete-model margin in Figure 1 (`scripts/figures.py:224`) and the hard-coded
  51%/653-sample annotation in Figure 6. They happen to match the current evidence but violate the
  figure contract and can drift.
- Figure 8 is the only mixed-status panel and distinguishes the measured complete inference from
  derived, untimed task schedules by fill, hatch, legend, and direct annotation. No projected
  latency is plotted. This is the right pattern.
- Figures 1 and 6 mix measured values with derived interpretation without an explicit visual
  evidence marker. A self-contained “one measured execution” label is enough; the paper does not
  need internal `[V]`/`[A]` tags in the public graphic.
- All eight PDFs contain embedded TrueType fonts. No rasterization, missing glyphs, broken axes, or
  overlapping annotations were found apart from the Figure 7 clipping.
- The figures remain interpretable in grayscale because position, outline/fill, hatching, labels,
  or line style carry the distinctions. Figure 2's pale outside-scope treatment is the exception;
  its label nearly disappears in grayscale. Role colours in Figures 1, 3, and 7 should remain
  redundant with explicit role labels, as they are now.

## Minimum viable revision sequence

1. Fix algorithm float containment and recompile all pages.
2. Correct Figure 3 and Figure 1's protocol scope.
3. Fix Figure 7 clipping.
4. Remove or qualify Figure 2.
5. Enlarge final-size text while revising Figures 4, 6, and 8 captions/titles.
6. Re-render every figure in colour and grayscale, then inspect the complete PDF once more.

After this sequence, the visual layer will be solid enough for academic review. A wholesale visual
redesign is not warranted.
