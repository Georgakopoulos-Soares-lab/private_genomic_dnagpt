# JBHI submission package

Built from `paper-docs/manuscript/` as of 2026-09-14 (author affiliations/corresponding-author/
Acknowledgment added, the citation-audit corrections applied, eight figures total). This directory
holds only the files the IEEE Author Portal's Upload Manuscript screen marks **Required** for this
submission board. No optional upload (tracked-changes copy, separate figure images, previously
published statement, supplementary material, cover-letter file, LaTeX supplementary file) is
included, per instruction to keep this to exactly what's needed.

Verified 2026-09-14: compiled with `tectonic` (no `latexmk` in this environment) and separately with
`paper-docs/scripts/build.sh lint` — both clean. No undefined citations/references/labels, and
`check_numbers.py`'s four checks (banned terms, numbers-without-evidence, citations, structure) all
pass. The compiled PDF is byte-identical to `manuscript/source/main.pdf`.

`main_manuscript/` is a **plain directory, not a zip** — kept unzipped so its `.tex` files stay
diffable in git (a zip is an opaque binary blob). Zip it only at the point of actually uploading to
the portal, e.g. `cd paper-docs/submission && zip -r main_manuscript.zip main_manuscript`; don't
commit that zip.

Regenerate `main_manuscript/` and the graphical-abstract PNG with:

```bash
cd paper-docs
rm -rf submission/main_manuscript
mkdir -p submission/main_manuscript/sections submission/main_manuscript/figures
cp manuscript/source/main.tex manuscript/source/preamble.tex manuscript/source/refs.bib submission/main_manuscript/
cp manuscript/source/sections/*.tex submission/main_manuscript/sections/
cp manuscript/figures/*.pdf submission/main_manuscript/figures/
cd manuscript/figures && pdftoppm -png -r 600 fig_graphical_abstract.pdf /tmp/ga && mv /tmp/ga-1.png ../../submission/DNAGPT-FHE_graphical_abstract.png
```

To compile and visually check the manuscript itself first (recommended after any `manuscript/`
change): `module load texlive/2023 && paper-docs/scripts/build.sh all`, then open
`paper-docs/manuscript/source/main.pdf` (committed separately, see the top-level `paper-docs/README.md`).

The Conflict of Interest and Graphical Abstract Text drafts are `.docx` (Microsoft Word), matching
the portal's accepted formats for those fields (PDF, RTF, or MS Word — not plain text). Regenerate
them after editing the wording in `paper-docs/scripts/make_submission_docs.py`:

```bash
paper-docs/.venv-paper/bin/python paper-docs/scripts/make_submission_docs.py
```

## Status per required field

| Portal field | File | Status |
|---|---|---|
| Main Manuscript | `main_manuscript/` (directory: `main.tex`, `preamble.tex`, `refs.bib`, `sections/*.tex`, `figures/*.pdf`) | Ready, and compiles clean (verified 2026-09-14, see above) — no build artifacts (`.aux`/`.bbl`/`.pdf` etc.) included. Zip this directory at upload time; the portal accepts "bundle LaTeX files in a single archive". `00_frontmatter.tex` now carries institutional affiliations, e-mail, and corresponding-author designation; `11_declarations.tex` now carries an in-manuscript Conflict of Interest statement and an Acknowledgment. **Still open**: ORCIDs (see `paper-docs/TODO.md`). |
| Conflict of Interest | `DNAGPT-FHE_conflict_of_interest_DRAFT.docx` | **Draft only.** Boilerplate "no known competing interests" text — not a substitute for each author's actual confirmation. Read and confirm (or amend) before uploading. The portal also accepts a "None of the authors have a conflict of interest to disclose" checkbox in lieu of a file. The manuscript itself now carries a matching in-text Conflict of Interest statement in `11_declarations.tex`. |
| Completed Author Consent form signed by all authors | `JBHI_author_consent_form_BLANK_template.pdf` | **Blank template only**, downloaded from the official JBHI page (`embs.org/jbhi/.../jbhi-consent_form_v3-1_fixed.pdf`). This requires a genuine hand-written signature from each of the three authors (Christos Galanopoulos, Kimon Antonios Provatas, Ilias Georgakopoulos-Soares) — it cannot be produced automatically. Print/sign/scan and replace this file before submitting. |
| Graphical Abstract | `DNAGPT-FHE_graphical_abstract.png` | Ready. 600 dpi PNG (4296×1230 px) rendered from `manuscript/figures/fig_graphical_abstract.pdf`, the paper's existing overview figure. |
| Graphical Abstract Text | `DNAGPT-FHE_graphical_abstract_text.docx` | **Draft**, 47 words, 1 sentence — under the 50-word/1-2-sentence limit. Author review recommended before submission. |

## Blocking items before actual submission

1. Sign the Author Consent Form (all three authors, hand-written signature).
2. Confirm or amend the draft Conflict of Interest statement (`DNAGPT-FHE_conflict_of_interest_DRAFT.docx`).
3. Add ORCIDs to `00_frontmatter.tex` (see `paper-docs/TODO.md`).
4. Review the graphical-abstract text/image and `main_manuscript/` for currency against the latest
   manuscript revision — rebuild if `manuscript/` changes after this package was built.
5. Zip `main_manuscript/` immediately before uploading (see above); don't commit the zip.
