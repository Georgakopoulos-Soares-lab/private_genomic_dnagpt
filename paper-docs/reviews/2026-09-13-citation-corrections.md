# Citation corrections applied — 2026-09-13

Scope: applying the findings of the independent citation audit conducted 2026-09-13.
All corrections are traceable to primary sources; no new claims were introduced.

Files edited: `manuscript/source/refs.bib`, `manuscript/source/sections/07_plaintext_results.tex`.
`sections/09_related_work.tex` was read and required no change. Nothing under `submission/`,
`manuscript/figures/`, or `evidence/` was touched, and the tracked `manuscript/source/main.pdf`
was not rebuilt in place.

## Changes to refs.bib

### concreteml-inference
- URL corrected: `docs.zama.org` → `docs.zama.ai` (broken domain). The corrected URL was
  confirmed to answer (HTTP 301) at the time of editing.
- Verified comment updated to 2026-09-13.

### safhire2025
- Author name corrected: `Kerrmarec` → `Kermarrec` (Anne-Marie Kermarrec, EPFL).
- `%% NOTE` comment added above the entry flagging the typo in the arXiv source metadata.
  This comment is the only remaining occurrence of the string `Kerrmarec` in the file.

### iron2022
- Unresolvable DOI `10.52202/068431-1143` removed.
- Replaced with canonical NeurIPS 2022 proceedings URL
  (`proceedings.neurips.cc/paper_files/paper/2022/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract-Conference.html`).

### nimbus2024
- Unresolvable DOI `10.52202/079017-0680` removed.
- Replaced with canonical NeurIPS 2024 proceedings URL
  (`proceedings.neurips.cc/paper_files/paper/2024/hash/264a9b3ce46abdf572dcfe0401141989-Abstract-Conference.html`).

### gpu-ckks-backend
- Note added acknowledging ISPASS 2025 publication (DOI 10.1109/ISPASS64960.2025.00045). The DOI
  was confirmed to resolve (HTTP 302 → ieeexplore.ieee.org/document/11096404) at the time of
  editing.
- arXiv citation retained as primary full-text source; author list is the 10-author
  ISPASS version. Verified comment updated to 2026-09-13.
- Open decision for the author: switch to `@inproceedings` with the ISPASS booktitle/DOI if the
  journal requires citing the peer-reviewed version.

### dnagpt2023
- Note field added identifying this as a preprint (bioRxiv 10.1101/2023.07.11.548628);
  no peer-reviewed journal version found as of 2026-09-13.

### heblas2025
- Note field added clarifying that `pages = {25}` is an article number, not a page span.

Rendering note: IEEEtran lowercases the first character of a `note` field, so the three new
notes render as "preprint; also deposited as bioRxiv …", "also presented at IEEE ISPASS 2025; …",
and "article 25 (article number, not page span)". This is the style's behaviour, not an error.

## Changes to sections/07_plaintext_results.tex

- mRNA-regression sentence revised to disambiguate DNAGPT's r²=0.62 from Xpresso's
  own published human r²=0.59; citations split to attribute each value to its source.
  Before:

  > against the $0.62$ result reported over the Xpresso regression baseline~\cite{dnagpt2023,xpresso}.

  After:

  > against the $0.62$ result reported by DNAGPT on the Xpresso dataset (Xpresso's own
  > published human $r^2$ is $0.59$~\cite{xpresso})~\cite{dnagpt2023}.

## Unchanged — confirmed correct

The following entries were confirmed correct by the audit and required no edits:
dnagpt2023 (author list, architecture claims, PAS accuracy sourcing),
dnabert2 (ICLR 2024, MCC comparators 0.69/0.87/0.85),
deepgsr, xpresso, ckks2017, openfhe, concreteml, tfhers,
gazelle2018, bolt2024, bumblebee2025, thor, safhire2025 (claims only),
idash2018, privategenomicqueries2017, gdpr, gymrek2013, erlich2018,
heblas2025, aegis.

## Could-not-verify items (no correction applied)

- DNAGPT polyadenylation-signal accuracy 0.9151: present in DNAGPT Figure 3a / Table S2
  (image-embedded, not machine-extractable). Accepted on the basis of the existing internal
  audit in `reviews/2026-09-07-paper-sources.md` which confirmed it against the HTML version
  of arXiv:2307.05628v3.
- DNAGPT feed-forward width 3,072: not stated as a named hyperparameter in the DNAGPT paper;
  standard 4×768 = 3072 convention. Author should confirm against the model checkpoint config.
- DNABERT-2 OpenReview forum ID `oMLQB4EZE1`: could not be independently clicked through;
  accepted on the basis of prior internal audit.

## Build status

`latexmk` is not installed; `tectonic` is. `bash scripts/build.sh lint` was run before and after
the edits. Before: all four checks clean. After:

```
==> lint
Banned terms: clean

Numbers without evidence (1):
  paper-docs/manuscript/source/sections/07_plaintext_results.tex:38: '0.59' is not in evidence/*.yaml — add it to the ledger with a source, or remove it
Citations: clean
Structure: clean

1 finding(s).
```

**The lint fails, and the failure is caused by the Step-2 sentence itself.** AGENTS.md rule 1
requires every number in `manuscript/` to exist in `evidence/*.yaml`; Xpresso's own human
$r^2 = 0.59$ is a new number and the ledger has no entry for it. The correction prompt
explicitly excluded `evidence/` from the editable set, so the ledger was **not** modified here.
The manuscript is therefore in a lint-failing state until the author does one of:

1. add a ledger entry (suggested shape, alongside `base.mrna_r2` in
   `evidence/measurements.yaml`):
   `id: base.mrna_xpresso_r2`, `value: 0.59`, `tag: "[V]"`, `reference_name: "Xpresso"`,
   scope: Xpresso's own published human $r^2$ (Agarwal & Shendure 2020, Cell Reports 31(7):107663,
   Table 1), source: `refs.bib:xpresso`; or
2. drop the parenthetical `0.59` from the sentence and keep only the citation split.

Option 1 is recommended: the disambiguation was the point of the correction.

A full compile was run on a scratch copy of `manuscript/source/` (not in place, because
`main.pdf` is a tracked build product): `tectonic main.tex` succeeded, BibTeX emitted no
warnings, there are no undefined citations or references, and all 24 entries render. The only
TeX warnings are the pre-existing underfull boxes already noted in
`reviews/2026-09-07-paper-sources.md`.

## Reconciliation

- 24 cited keys, 24 bibliography entries, 0 missing, 0 uncited. Unchanged.

## Addendum — follow-up on the two open items (same day)

On the author's instruction, the two items left open above were closed.

### Ledger entry for Xpresso's 0.59 (resolves the lint failure)

`evidence/measurements.yaml` gained one row under `plaintext_baseline`, after `base.mrna_pearson`:

```yaml
  - id: base.mrna_xpresso_r2
    value: 0.59
    tag: "[V]"
    scope: Xpresso's own published human mRNA-abundance r^2 (Agarwal & Shendure 2020,
           Cell Reports 31(7):107663, Table 1); not measured in this work
    source: "refs.bib:xpresso; reviews/2026-09-13-citation-corrections.md"
```

The row is a published comparator, not a local measurement, and its scope says so. No existing
value or tag was changed. `scripts/figures.py` still imports and reads the ledger.

### safhire2025 comment (resolves the `Kerrmarec[^r]` checklist item)

The `%% NOTE` comment above `@article{safhire2025` was reworded so that it documents the arXiv
metadata misspelling without carrying the literal misspelt string:

```
%% NOTE: arXiv:2509.01253 metadata misspells the sixth author's surname (transposed
%%   double-r); corrected to "Kermarrec" (Anne-Marie Kermarrec, EPFL).
```

### Build status after the follow-up

```
==> lint
Banned terms: clean
Numbers without evidence: clean
Citations: clean
Structure: clean

All checks clean.
```

`tectonic` compile of a scratch copy succeeds; no BibTeX warnings; no undefined citations. The
"Invalid UTF-8 byte or sequence at line 11" message tectonic prints is pre-existing (it appears
identically on a pristine checkout of the current commit) and is unrelated to these edits.

### Final checklist

- `docs.zama.org` occurrences: 0
- `Kerrmarec[^r]` occurrences: 0
- `10.52202` occurrences: 0
- `safhire2025` entries: 1; `iron2022` + `nimbus2024` entries: 2
- `.tex` files modified: `sections/07_plaintext_results.tex` only
- `submission/` untouched; `manuscript/source/main.pdf` not rebuilt in place
- 24 cited keys, 24 bibliography entries, 0 missing, 0 uncited

All citation corrections applied and verified.
