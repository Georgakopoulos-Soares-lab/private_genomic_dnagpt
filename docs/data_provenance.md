# Data provenance

Every dataset used, with exact origin, retrieval, any recovery route, preprocessing, and license.
Retrieval date: **2026-07-24**. None of these files are committed (all under gitignored `data/`).

## Task 1 — GSR (human AATAAA polyadenylation signals)

- **Source** `[V]`: DeepGSR (Kalkatawi et al., *Bioinformatics* 2019), Zenodo record **1117159**,
  `Data.zip` → `https://zenodo.org/records/1117159/files/Data.zip?download=1` (255 MB).
- **Files used**: `data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa` (11,302 positives),
  `hs_negAATAAA_polyA.fa` (11,302 negatives). 606 bp records = 300 up + AATAAA(6) + 300 down.
- **Preprocessing** `[V]`: strip the central 6 bp motif (index `[300:306]`) → 600 bp, matching
  DNAGPT's `scripts/classification.sh` format. Confirmed against the repo's `processData.m`
  ("removes the motif from the middle + adding P or N at the end"). Done inline in `eval/eval_gsr.py`.
- **Label mapping**: DNAGPT decodes next token `N` → real GSR (label 1), `A` → fake (label 0).
- **License**: DeepGSR academic release (Zenodo). Redistribution not needed — re-downloadable.

## Task 3 — mRNA abundance (Xpresso human)

- **Original source (DEAD)** `[U]`: `krishna.gs.washington.edu/.../Xpresso/data/datasets/` —
  TCP-times-out / ECONNREFUSED globally (verified from local network and Anthropic egress). GitHub
  copies in `vagarwal87/Xpresso` are symlinks pointing at this dead host.
- **Recovery** `[V]`: **Internet Archive** (Wayback CDX enumerated the path). Retrieved:
  - `Roadmap_FantomAnnotations.InputData.pM10Kb.txt.gz` — snapshot `20240420170821` (104 MB, 18,413 rows)
  - `mask_histone_genes.txt`, `mask_histone_genes_mm10.txt` — snapshots `2024042017…`
  - URL form: `https://web.archive.org/web/<ts>id_/<original-url>`
- **Format**: TSV, columns `ENSID | EXPRESSION | UTR5LEN CDSLEN INTRONLEN UTR3LEN UTR5GC CDSGC
  UTR3GC ORFEXONDENSITY | PROMOTER(20 kb)`. The 8 middle columns = mRNA half-life features.
- **Preprocessing** `[V]` — faithful Python-3 port of Xpresso `Fig1_S2/setup_training_files.py`
  (`eval/build_mrna_testset.py`): drop histone/chrY genes → 18,377 genes (matches the paper);
  `log10(x+0.1)` on EXPRESSION+lengths+ORF density; shuffle `pandas.sample(frac=1, random_state=1)`;
  z-score the 9 numeric columns; **test = last 1,000 rows**. Promoter sliced to Xpresso's 10.5 kb
  window `[3000:13500]` (7 kb up + 3.5 kb down of the TSS at 10 kb) — matches the 10,500 bp sequences
  in `scripts/regression.sh`.
- **Caveat** `[U]`: `pandas.sample(random_state=1)` may not reproduce Xpresso's exact original 1,000
  test genes across library versions; the split is faithful in method but gene-for-gene identity is
  unverified. This is the leading candidate for the small r² gap vs the paper (0.562 vs ~0.62).
- **License**: Xpresso academic release (Agarwal & Shendure, *Cell Reports* 2020).

## Task 2 — GUE (Genome Understanding Evaluation)

- **Source** `[V]`: DNABERT-2 (Zhou et al., ICLR 2024), Google Drive
  `1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-` → `GUE.zip` (284 MB). Mirror: HF `leannmlindsey/GUE`.
- **Datasets used** (human promoters & splice sites, per the task brief):
  - `data/gue/GUE/prom/prom_core_all` — core-promoter detection (binary; ~70 bp; 47,356/5,920).
  - `data/gue/GUE/prom/prom_300_all` — promoter detection (binary; 300 bp; 47,356/5,920).
  - `data/gue/GUE/splice/reconstructed` — splice-site detection (3-class; 400 bp; 36,496/4,562).
- **Format**: CSV `sequence,label`. No preprocessing beyond k-mer tokenization.
- **Head**: none released → fine-tuned by `eval/finetune_gue.py` (backbone `dna_gpt0.1b_m` + linear
  head on masked-mean-pooled hidden state; DNABERT-2 protocol: full fine-tune, 3 epochs).
- **License**: DNABERT-2 / GUE academic release.
