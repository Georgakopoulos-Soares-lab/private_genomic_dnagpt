# -*- coding: utf-8 -*-
"""Task 3 - Human mRNA abundance regression batch evaluation.

Measures the released DNAGPT `regression.pth` head on the Xpresso human test
split (Agarwal & Shendure 2020). DNAGPT follows Xpresso: promoter sequence +
8 mRNA half-life features -> scaled expression level.

Input template (from scripts/regression.sh):  <R>{seq}<+><M><=><M>
with the 8 features passed as `numbers` (spliced in as number embeddings).

Reads the HDF5 produced by eval/build_mrna_testset.py:
    'seq'      : (N,)   10.5 kb promoter window strings (Xpresso [3000:13500])
    'data'     : (N, 8) scaled half-life features
    'label'    : (N,)   scaled expression target (the value to predict)

Xpresso data recovered from the Internet Archive (original krishna.gs.washington.edu
host is offline).
"""

import argparse
import csv
import json
import os
import time

import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from common import load_model, regress

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="data/mrna/pM10Kb_1KTest/test.h5")
    ap.add_argument("--weight", default="checkpoints/regression.pth")
    ap.add_argument("--name", default="dna_gpt0.1b_h")
    ap.add_argument("--limit", type=int, default=-1, help="-1 = all rows")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="mrna_xpresso_human")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    with h5py.File(args.h5, "r") as f:
        seqs = [s.decode() if isinstance(s, bytes) else s for s in f["seq"][:]]
        data = f["data"][:]
        label = f["label"][:]
    total = len(seqs)
    n = total if args.limit < 0 else min(args.limit, total)
    print(f"loaded {total} rows; evaluating {n}; features/row={data.shape[1]}")

    model, tok, device = load_model(args.name, args.weight, device=args.device)
    print(f"device={device} max_len={args.max_len}")

    preds, tgts, rows = [], [], []
    t0 = time.time()
    for i in range(n):
        seq = seqs[i]
        numbers = [float(x) for x in data[i]]
        prompt = f"<R>{seq}<+><M><=><M>"
        yhat = regress(model, tok, prompt, numbers, args.max_len, device)
        preds.append(yhat)
        tgts.append(float(label[i]))
        rows.append({"idx": i, "target": float(label[i]), "pred": yhat})
        if (i + 1) % 100 == 0:
            r2 = r2_score(tgts, preds)
            print(
                f"  {i + 1}/{n} running_r2={r2:.4f} "
                f"({(time.time() - t0) / (i + 1) * 1000:.0f} ms/seq)"
            )

    preds, tgts = np.array(preds), np.array(tgts)
    r2 = float(r2_score(tgts, preds))
    rho = float(spearmanr(tgts, preds).correlation)
    pear = float(np.corrcoef(tgts, preds)[0, 1])
    result = {
        "task": "Human mRNA abundance regression (Xpresso)",
        "tag": args.tag,
        "weight": args.weight,
        "model": args.name,
        "device": device,
        "n": int(n),
        "r2": r2,
        "pearson_r": pear,
        "spearman_r": rho,
        "seconds": time.time() - t0,
        "paper_reference_r2": 0.62,  # DNAGPT human mRNA abundance
    }
    os.makedirs("results/runs", exist_ok=True)
    with open(f"results/runs/{args.tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    with open(f"results/runs/{args.tag}_preds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx", "target", "pred"])
        w.writeheader()
        w.writerows(rows)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
