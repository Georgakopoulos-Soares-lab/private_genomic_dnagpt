# -*- coding: utf-8 -*-
"""Task 1 - Genomic Signal & Region Recognition (GSR) batch evaluation.

Measures the released DNAGPT `classification.pth` head on the DeepGSR human
AATAAA polyadenylation-signal dataset (Kalkatawi et al. 2019).

DeepGSR sequences are 606 bp = 300 up + AATAAA(6) + 300 down. DNAGPT's format
(scripts/classification.sh) removes the 6 bp motif -> 600 bp, then classifies
<R>{seq}<=><R>. Decoded next token 'N' => real GSR, 'A' => fake GSR.
"""

import argparse
import csv
import json
import os
import random
import time

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from common import classify, load_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOTIF_START, MOTIF_LEN = 300, 6  # 0-indexed center motif in the 606 bp record


def read_fa(path):
    seqs = []
    with open(path) as fh:
        seq = None
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq is not None:
                    seqs.append(seq)
                seq = ""
            else:
                seq += line
        if seq:
            seqs.append(seq)
    return seqs


def strip_motif(seq):
    """606 bp -> 600 bp by removing the central AATAAA motif (DNAGPT format)."""
    if len(seq) == 606:
        return seq[:MOTIF_START] + seq[MOTIF_START + MOTIF_LEN :]
    return seq  # already processed / unexpected length: pass through


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa")
    ap.add_argument("--neg", default="data/gsr/Data/Human/PAS/hs_negAATAAA_polyA.fa")
    ap.add_argument("--weight", default="checkpoints/classification.pth")
    ap.add_argument("--name", default="dna_gpt0.1b_m")
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="sequences PER class (balanced). -1 = all",
    )
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=40)
    ap.add_argument("--tag", default="gsr_aataaa_human")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    random.seed(args.seed)

    pos = read_fa(args.pos)
    neg = read_fa(args.neg)
    if args.limit > 0:
        pos = random.sample(pos, min(args.limit, len(pos)))
        neg = random.sample(neg, min(args.limit, len(neg)))
    # label 1 = real GSR (positive), 0 = fake (negative)
    samples = [(s, 1) for s in pos] + [(s, 0) for s in neg]
    random.shuffle(samples)
    print(f"pos={len(pos)} neg={len(neg)} total={len(samples)}")

    model, tok, device = load_model(args.name, args.weight, device=args.device)
    print(f"device={device} max_len={args.max_len}")

    y_true, y_pred, rows = [], [], []
    other = 0
    t0 = time.time()
    for i, (seq, label) in enumerate(samples):
        s600 = strip_motif(seq)
        char = classify(model, tok, s600, args.max_len, device)
        if char == "N":
            pred = 1
        elif char == "A":
            pred = 0
        else:
            pred = 0
            other += 1
        y_true.append(label)
        y_pred.append(pred)
        rows.append({"idx": i, "label": label, "pred": pred, "token": char})
        if (i + 1) % 200 == 0:
            acc = accuracy_score(y_true, y_pred)
            print(
                f"  {i + 1}/{len(samples)} running_acc={acc:.4f} "
                f"({(time.time() - t0) / (i + 1) * 1000:.0f} ms/seq)"
            )

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    elapsed = time.time() - t0

    result = {
        "task": "GSR recognition (human AATAAA PAS)",
        "tag": args.tag,
        "weight": args.weight,
        "model": args.name,
        "device": device,
        "n": len(samples),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "confusion_matrix_[0,1]x[0,1]": cm,
        "non_NA_tokens": other,
        "seconds": elapsed,
        "paper_reference_acc": 0.916,  # DeepGSR human AATAAA test acc (approx)
    }
    os.makedirs("results/runs", exist_ok=True)
    with open(f"results/runs/{args.tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    with open(f"results/runs/{args.tag}_preds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx", "label", "pred", "token"])
        w.writeheader()
        w.writerows(rows)

    print("\n=== RESULT ===")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "confusion_matrix_[0,1]x[0,1]"},
            indent=2,
        )
    )
    print("confusion [rows=true 0,1][cols=pred 0,1]:", cm)


if __name__ == "__main__":
    main()
