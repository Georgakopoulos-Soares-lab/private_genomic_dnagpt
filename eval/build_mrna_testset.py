# -*- coding: utf-8 -*-
"""Build the Xpresso human TEST split (last 1000 genes) for DNAGPT mRNA eval.

Faithful Python-3 port of vagarwal87/Xpresso Fig1_S2/setup_training_files.py
preprocessing so the held-out 1000 genes match the canonical Xpresso split that
DNAGPT followed:
  1. drop histone + chrY genes (mask files)
  2. log10(x+0.1) on EXPRESSION, UTR5LEN, CDSLEN, INTRONLEN, UTR3LEN, ORFEXONDENSITY
  3. shuffle with pandas .sample(frac=1, random_state=1)
  4. z-score (sklearn StandardScaler) the 9 numeric columns (EXPRESSION + 8 feats)
  5. TEST = the LAST `test` rows (Xpresso appends test at the tail)

Promoter (20 kb, TSS at 10 kb) is sliced to Xpresso's 10.5 kb window [3000:13500]
= 7 kb upstream + 3.5 kb downstream, matching scripts/regression.sh (10500 bp).

Input (Internet Archive mirror of the dead krishna host):
  data/mrna/Roadmap_human_pM10Kb.txt.gz, mask_histone_genes.txt, mask_histone_genes_mm10.txt
Output: data/mrna/pM10Kb_1KTest/test.h5  with datasets seq/data/label/geneName
"""

import os

import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MRNA = os.path.join(REPO_ROOT, "data", "mrna")
WIN_L, WIN_R = 3000, 13500  # 10.5 kb window; TSS at 10 kb

LOG_COLS = ["EXPRESSION", "UTR5LEN", "CDSLEN", "INTRONLEN", "UTR3LEN", "ORFEXONDENSITY"]
FEATURES = [
    "UTR5LEN",
    "CDSLEN",
    "INTRONLEN",
    "UTR3LEN",
    "UTR5GC",
    "CDSGC",
    "UTR3GC",
    "ORFEXONDENSITY",
]  # the 8 half-life features
NUMERIC = ["EXPRESSION"] + FEATURES  # 9 cols z-scored together


def main(test_count=1000, valid_count=1000):
    table = pd.read_table(
        os.path.join(MRNA, "Roadmap_human_pM10Kb.txt.gz"), index_col=0
    )
    m1 = pd.read_table(os.path.join(MRNA, "mask_histone_genes_mm10.txt"), header=None)
    m2 = pd.read_table(os.path.join(MRNA, "mask_histone_genes.txt"), header=None)
    table = table[~table.index.isin(m1[0])]
    table = table[~table.index.isin(m2[0])]

    table[LOG_COLS] = np.log10(table[LOG_COLS] + 0.1)
    table = table.sample(frac=1.0, replace=False, random_state=1)
    table[NUMERIC] = preprocessing.scale(table[NUMERIC])

    n = table.shape[0]
    test = table.iloc[n - test_count :]  # last `test_count` rows
    print(
        f"total={n} test={test.shape[0]} (valid={valid_count} train={n - test_count - valid_count})"
    )

    seqs = np.array([s[WIN_L:WIN_R] for s in test["PROMOTER"].values], dtype=object)
    lens = {len(s) for s in seqs}
    assert lens == {WIN_R - WIN_L}, f"unexpected window lengths: {lens}"
    data = test[FEATURES].values.astype("float32")
    label = test["EXPRESSION"].values.astype("float32")
    genes = list(test.index)

    out_dir = os.path.join(MRNA, "pM10Kb_1KTest")
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(os.path.join(out_dir, "test.h5"), "w") as f:
        f.create_dataset("seq", data=np.array(seqs, dtype=h5py.string_dtype()))
        f.create_dataset("data", data=data)
        f.create_dataset("label", data=label)
        f.create_dataset("geneName", data=np.array(genes, dtype=h5py.string_dtype()))
    print(
        f"wrote {out_dir}/test.h5  seq_len={WIN_R - WIN_L} feats={data.shape} "
        f"label_range=[{label.min():.3f},{label.max():.3f}]"
    )


if __name__ == "__main__":
    main()
