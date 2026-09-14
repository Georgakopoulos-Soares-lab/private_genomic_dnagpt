# Author metadata, Conflict of Interest/Acknowledgment, FIDESlib citation, prose edits — 2026-09-14

Author-directed changes, applied to `manuscript/source/` and propagated to `submission/main_manuscript/`.

## Author affiliations and corresponding author (`sections/00_frontmatter.tex`)

Per author instruction: Christos Galanopoulos carries only "The University of Texas at Austin";
Kimon Antonios Provatas and Ilias Georgakopoulos-Soares carry both "The University of Texas at
Austin" and "The University of Texas at Austin College of Pharmacy". Ilias Georgakopoulos-Soares
is designated corresponding author, e-mail `ilias@austin.utexas.edu`. Rendered as three `\thanks`
blocks under IEEEtran's journal `\author` macro. The frontmatter `TODO(authors)` comment now
covers only ORCIDs, which remain outstanding.

## Conflict of Interest and Acknowledgment (`sections/11_declarations.tex`)

Added, after Code Availability Statement:

- `\section*{Conflict of Interest Statement}` — "The authors declare no competing financial or
  non-financial interests." This resolves the gap `submission/README.md` had flagged: JBHI expects
  an in-manuscript COI disclosure, not only the portal-level `.docx`.
- `\section*{Acknowledgment}` — "This work has been supported by the National Institute of General
  Medical Sciences of the National Institutes of Health [R35GM155468 to I.G.S.]; start-up funds
  awarded to I.G.S.; and the Texas POC Award (2026) to I.G.S." Wording and grant identifiers as
  supplied by the author; not independently verified against a funding-agency record (out of scope
  for this repository's evidence discipline, which governs scientific claims, not funding text).

The prior placeholder comment gating the Acknowledgment on author-supplied, approved wording is
removed, since that wording has now been supplied.

## FIDESlib / `gpu-ckks-backend` citation (`refs.bib`)

Previous entry mixed the arXiv preprint (as the `@article` container) with an ISPASS 2025 DOI
folded in as a `note`. Per author instruction, this misrepresented the source: FIDESlib's own
repository (`github.com/CAPS-UMU/FIDESlib`) explicitly asks users to cite the ISPASS 2025 paper,
not the preprint. Verified independently:

- Crossref record for `10.1109/ISPASS64960.2025.00045`: title, full 10-author list, and venue
  ("2025 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)",
  Ghent, Belgium, IEEE) match. Crossref's own `page` field for this DOI is `1-3` (article-relative
  numbering).
- Universidad de Murcia's institutional repository record for the same work gives proceedings page
  range **365–367**, matching the author's instruction. IEEE conference proceedings frequently
  carry both an article-relative page count and a volume/proceedings page range; the latter is used
  here as it is what the author specified and what a reader would look up in the printed
  proceedings.
- The FIDESlib GitHub README's own citation section confirms "Poster paper" status and asks
  contributors to cite the ISPASS record.

The entry is now `@inproceedings` with `booktitle`, `pages = {365--367}`, `address = {Ghent,
Belgium}`, `publisher = {IEEE}`, `doi = {10.1109/ISPASS64960.2025.00045}`, `note = {Poster paper}`,
retaining `eprint`/`archivePrefix`/`url` for the arXiv full text as a secondary access point. This
is the `@inproceedings` alternative the 2026-09-13 audit had already flagged as an open decision
for the author (see `reviews/2026-09-13-citation-corrections.md`, item 1-E); the author has now
made that call.

## Prose edits (author-directed rewording)

| File | Before | After |
|---|---|---|
| `sections/01_introduction.tex` | "These results close arithmetic feasibility…" | "These results establish arithmetic feasibility…" |
| `sections/10_conclusion.tex` | "Arithmetic feasibility is therefore affirmative from…" | "Arithmetic feasibility therefore holds from…" |
| `sections/06_optimization.tex` | "The retained system completes the full classifier in one clean sample; the pre-optimization timing denominator lacks matching evidence quality, so we report the endpoint without a speedup factor." | "The retained system completes the full classifier in one clean sample. No pre-optimization timing was measured under matching conditions, so we report this endpoint directly rather than as a speedup factor." |
| `sections/08_limitations.tex` | "Weight recovery becomes inexpensive per-segment regression instead of a global inversion problem." | "An attacker can then recover each segment's weights by ordinary regression, rather than by inverting the whole model at once." |

No claim, tag, or number changed; these are wording-only edits matching the meaning already
established.

## Verification

- `bash paper-docs/scripts/build.sh lint`: banned terms / numbers-without-evidence / citations /
  structure all clean. The grant identifier `R35GM155468` never matches the number-lint's regex
  (every digit run in it is preceded by a letter or another digit, so the lookbehind that requires
  a non-word character before a number never fires); the year `2026` in "Texas POC Award (2026)"
  is already authorized by the ledger's own `2026-*` dated `source`/`scope` strings.
- `manuscript/source/main.pdf` rebuilt with `tectonic` (no `latexmk` in this environment); no
  undefined citations/references/labels. Rendered text spot-checked for every change above.
- `submission/main_manuscript/` regenerated from the updated `manuscript/source/` (the documented
  copy commands in `submission/README.md`) and compiled independently in a scratch copy: identical
  output, byte-for-byte, to `manuscript/source/main.pdf` (294,140 bytes). No build artifacts
  (`.aux`/`.bbl`/`.log`/`.pdf`) were left inside the committed `submission/main_manuscript/`
  directory, matching its existing convention.
- `submission/README.md`, `paper-docs/README.md`, and `paper-docs/TODO.md` updated to drop the
  now-resolved frontmatter/COI TODO items and note the remaining ones (ORCIDs, signed consent
  form, repository reuse license).

## Not changed

`paper-docs/evidence/*.yaml` was not touched — nothing here is a measured/scientific claim.
`09_related_work.tex` was not touched. No file under `submission/` other than `main_manuscript/`
and `README.md` was touched (the Conflict of Interest and Graphical Abstract Text `.docx` drafts,
the consent-form template, and the graphical-abstract PNG are unaffected by these edits and were
not regenerated).
