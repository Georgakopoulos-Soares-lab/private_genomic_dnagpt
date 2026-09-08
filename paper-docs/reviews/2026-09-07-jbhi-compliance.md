# IEEE JBHI manuscript compliance check — 2026-09-07

Checked against the saved `Prepare and Submit Your Manuscript` page and the supplied
`paper-docs/brief.tex` template after the final build.

## Manuscript checks

| Requirement | Result |
|---|---|
| Article type | Regular research paper. |
| Layout | IEEEtran, 10-point, single-spaced, two-column, US letter. |
| Page limit | 14 pages, including references; meets the 14-page regular-paper maximum. |
| Abstract | One paragraph, 210 source words; states objective, methods, major results, conclusion, and one biomedical-significance sentence. It contains no citations, footnotes, displayed equations, or abbreviations. |
| Keywords | Four alphabetized search phrases. |
| Figures and tables | Embedded in the rendered PDF; the included packing figure is legible at journal size. Tables and algorithms remain in the relevant manuscript sections. |
| References | Separate section at the end; citations are numeric in square brackets and bibliography records use IEEE style. |
| Ethics/data statements | Ethics, data availability, and code availability statements are present. |
| Technical build | All evidence, terminology, citation, and structure lint checks pass. There are no undefined citations or cross-references and no overfull boxes. All PDF fonts are embedded. |
| Visual review | All 14 pages, including the protocol algorithms, figure, result tables, declarations, and references, were rendered to images and inspected. |

## Author-supplied items still required before portal submission

- Institutional e-mail addresses and the corresponding-author designation.
- An ORCID for every author and the full mailing/contact data requested by the portal.
- Approved funding/support and acknowledgment wording, if applicable.
- Exact evaluated releases or commits for the Concrete ML and TFHE-rs software citations.
- A handwritten multi-author consent form uploaded as a supporting document.
- A cover letter confirming originality and explaining the work's innovation and significance for
  JBHI; any preliminary conference version must be disclosed and cited as required.

The 14-page manuscript is within the submission limit, but it exceeds the eight-page threshold for
mandatory overlength charges described in the supplied journal guide.

## Follow-up: figure restoration and column-layout repair

The user identified blank columns on pages 6–8 and algorithms grouped without intervening prose
in the earlier render. This follow-up supersedes the page-count and visual-layout conclusions
above: successful compilation did not establish satisfactory float placement.

The corrected PDF is **12 pages**, including **nine figures, six tables, four algorithms, and
23 references**. The IEEEtran 10-point body, two-column geometry, and journal margins are unchanged.
Automatic section float barriers were removed; the short algorithms now follow their explanatory
paragraphs, and the complete-block algorithm shares page 7 with body text. Matrix multiplication
precedes block evaluation in the protocol explanation, keeping algorithm numbering in reading order.
Both columns on pages 6–8 contain substantial content. All figures have captions and in-text calls.

All pages were rendered and visually inspected; pages 5–8 and the final results pages were inspected
again after float adjustments. PDF text bounding boxes additionally confirm substantive content in
both columns on every page. The final PDF has no missing-reference, missing-citation, overfull-box,
or floats-only-column warning, and all fonts are embedded. A few underfull horizontal boxes and one
underfull vertical box remain; their rendered locations were checked and do not produce a blank
column or clipped material. All four manuscript lint checks and `git diff --check` pass.

The author-supplied submission items above remain open. Twelve pages still exceed the eight-page
threshold for mandatory overlength charges in the supplied guide.

## Citation and Figure 2 follow-up

The final rebuild remains 12 pages and now contains 24 references, with no missing or uncited
bibliography entries. The source-audit follow-up records primary-source checks and corrections.
Figure 2 labels have verified box clearance; all fonts remain embedded. The rendered page contact
sheet and full-size Figure 2 and bibliography pages were checked again.

The earlier Concrete ML/TFHE-rs release-provenance blocker is superseded: the manuscript now
correctly describes a documentation review, not an experimental backend comparison. Official
documentation and access dates are cited. Pinned releases and measured runs would be required
if a comparative experiment were added later. Author-supplied items and overlength considerations
remain open.
