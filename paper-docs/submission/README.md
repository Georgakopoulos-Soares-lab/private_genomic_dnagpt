# JBHI submission package

Built from `paper-docs/manuscript/` at commit `f1774ab` ("JBHI restructuring"). This directory
holds only the files the IEEE Author Portal's Upload Manuscript screen marks **Required** for this
submission board. No optional upload (tracked-changes copy, separate figure images, previously
published statement, supplementary material, cover-letter file, LaTeX supplementary file) is
included, per instruction to keep this to exactly what's needed.

Regenerate with (manuscript zip + graphical-abstract PNG only — the other two files are hand-authored/downloaded, see below):

```bash
cd paper-docs/manuscript/source
zip -r ../../submission/DNAGPT-FHE_main_manuscript_source.zip main.tex preamble.tex refs.bib sections/
cd ../figures && zip -r ../../submission/DNAGPT-FHE_main_manuscript_source.zip fig_*.pdf
pdftoppm -png -r 600 fig_graphical_abstract.pdf /tmp/ga && mv /tmp/ga-1.png ../../submission/DNAGPT-FHE_graphical_abstract.png
```

## Status per required field

| Portal field | File | Status |
|---|---|---|
| Main Manuscript | `DNAGPT-FHE_main_manuscript_source.zip` | Ready. LaTeX source bundle (`main.tex`, `preamble.tex`, `refs.bib`, `sections/*.tex`, `figures/*.pdf`) — no build artifacts (`.aux`/`.bbl`/`.pdf` etc.) included, matches the portal's "bundle LaTeX files in a single archive" option. **Not yet ready to submit**: `00_frontmatter.tex` still carries `TODO(authors)` placeholders for institutional e-mails, corresponding-author designation, and funding/support text (see `paper-docs/TODO.md`). |
| Conflict of Interest | `DNAGPT-FHE_conflict_of_interest_DRAFT.txt` | **Draft only.** Boilerplate "no known competing interests" text — not a substitute for each author's actual confirmation. Read and confirm (or amend) before uploading. The portal also accepts a "None of the authors have a conflict of interest to disclose" checkbox in lieu of a file. Separately: the manuscript itself has no in-text Conflict-of-Interest disclosure (only Ethics/Data/Code Availability in `11_declarations.tex`), which JBHI's guidelines say is also required — add one before submission. |
| Completed Author Consent form signed by all authors | `JBHI_author_consent_form_BLANK_template.pdf` | **Blank template only**, downloaded from the official JBHI page (`embs.org/jbhi/.../jbhi-consent_form_v3-1_fixed.pdf`). This requires a genuine hand-written signature from each of the three authors (Christos Galanopoulos, Kimon Antonios Provatas, Ilias Georgakopoulos-Soares) — it cannot be produced automatically. Print/sign/scan and replace this file before submitting. |
| Graphical Abstract | `DNAGPT-FHE_graphical_abstract.png` | Ready. 600 dpi PNG (4296×1230 px) rendered from `manuscript/figures/fig_graphical_abstract.pdf`, the paper's existing overview figure. |
| Graphical Abstract Text | `DNAGPT-FHE_graphical_abstract_text.txt` | **Draft**, 47 words, 1 sentence — under the 50-word/1-2-sentence limit. Author review recommended before submission. |

## Blocking items before actual submission

1. Sign the Author Consent Form (all three authors, hand-written signature).
2. Confirm or amend the Conflict of Interest statement; add an in-manuscript COI disclosure.
3. Fill in `TODO(authors)` items in `00_frontmatter.tex`: e-mails, corresponding author, funding/support.
4. Review the graphical-abstract text/image and the manuscript bundle for currency against the
   latest manuscript revision — rebuild if `manuscript/` changes after this package was built.
