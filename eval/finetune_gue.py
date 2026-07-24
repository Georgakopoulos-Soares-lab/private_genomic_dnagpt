# -*- coding: utf-8 -*-
"""Task 2 - GUE (Genome Understanding Evaluation) fine-tuning + evaluation.

DNAGPT ships no GUE code, so this builds a minimal fine-tuning harness: the
DNAGPT foundation backbone (dna_gpt0.1b_m) + a linear classification head on the
masked-mean-pooled final hidden state. Trains per GUE dataset (DNABERT-2 protocol:
full fine-tune, 3 epochs) and reports MCC (the GUE primary metric) + accuracy/F1.

Dataset dir must contain train.csv / dev.csv / test.csv with header `sequence,label`.
Usage:
  python finetune_gue.py --data data/gue/GUE/prom/prom_300_all --tag gue_prom_300_all
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "DNAGPT"))
from dna_gpt.model import DNAGPT  # noqa: E402
from dna_gpt.tokenizer import KmerTokenizer  # noqa: E402
from common import SPECIAL_TOKENS, pick_device  # noqa: E402


def read_csv(path):
    seqs, labels = [], []
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            seqs.append(row["sequence"])
            labels.append(int(row["label"]))
    return seqs, labels


class GUEClassifier(nn.Module):
    def __init__(self, backbone, num_classes, pad_id):
        super().__init__()
        self.backbone = backbone
        self.pad_id = pad_id
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(backbone.embedding_dim, num_classes)

    def forward(self, tokens):
        mask = (tokens != self.pad_id).unsqueeze(-1).float()  # [B,T,1]
        emb = self.backbone._embedding_impl(tokens)
        hid = self.backbone._transformer_impl(emb)  # [B,T,C]
        pooled = (hid * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(self.dropout(pooled))


def make_batches(seqs, labels, tok, max_len, batch, device, shuffle, seed=0):
    idx = list(range(len(seqs)))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for i in range(0, len(idx), batch):
        chunk = idx[i : i + batch]
        enc = [
            tok.encode(f"<R>{seqs[j]}", max_len=max_len, to_tensor=False) for j in chunk
        ]
        m = max(len(e) for e in enc)
        padded = [e + [tok.pad_id] * (m - len(e)) for e in enc]
        x = torch.tensor(padded, dtype=torch.long, device=device)
        y = torch.tensor([labels[j] for j in chunk], dtype=torch.long, device=device)
        yield x, y


@torch.no_grad()
def evaluate(model, seqs, labels, tok, max_len, batch, device):
    model.eval()
    preds = []
    for x, _ in make_batches(seqs, labels, tok, max_len, batch, device, shuffle=False):
        preds.extend(model(x).argmax(-1).tolist())
    return (
        matthews_corrcoef(labels, preds),
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="macro"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="GUE dataset dir")
    ap.add_argument("--name", default="dna_gpt0.1b_m")
    ap.add_argument("--backbone", default="checkpoints/dna_gpt0.1b_m.pth")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--limit_train", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    tr_s, tr_y = read_csv(os.path.join(args.data, "train.csv"))
    te_s, te_y = read_csv(os.path.join(args.data, "test.csv"))
    if args.limit_train > 0:
        tr_s, tr_y = tr_s[: args.limit_train], tr_y[: args.limit_train]
    num_classes = len(set(tr_y) | set(te_y))
    print(
        f"{args.tag}: train={len(tr_s)} test={len(te_s)} classes={num_classes} device={device}"
    )

    tok = KmerTokenizer(6, SPECIAL_TOKENS, dynamic_kmer=True)
    backbone = DNAGPT.from_name(args.name, len(tok))
    state = torch.load(args.backbone, map_location="cpu")
    state = state["model"] if "model" in state else state
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    print(f"backbone loaded (missing={len(missing)} unexpected={len(unexpected)})")
    model = GUEClassifier(backbone, num_classes, tok.pad_id).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, seen = 0.0, 0
        for step, (x, y) in enumerate(
            make_batches(
                tr_s,
                tr_y,
                tok,
                args.max_len,
                args.batch,
                device,
                shuffle=True,
                seed=args.seed + ep,
            )
        ):
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            seen += len(y)
            if (step + 1) % 200 == 0:
                print(
                    f"  ep{ep + 1} step{step + 1} loss={tot / seen:.4f} "
                    f"({(time.time() - t0) / 60:.1f} min)"
                )
        mcc, acc, f1 = evaluate(
            model, te_s, te_y, tok, args.max_len, args.batch, device
        )
        print(
            f"  == epoch {ep + 1}: train_loss={tot / seen:.4f} "
            f"test MCC={mcc:.4f} acc={acc:.4f} f1={f1:.4f}"
        )

    mcc, acc, f1 = evaluate(model, te_s, te_y, tok, args.max_len, args.batch, device)
    result = {
        "task": "GUE fine-tune",
        "tag": args.tag,
        "dataset": args.data,
        "model": args.name,
        "backbone": args.backbone,
        "device": device,
        "num_classes": num_classes,
        "n_train": len(tr_s),
        "n_test": len(te_s),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "max_len": args.max_len,
        "mcc": mcc,
        "accuracy": acc,
        "macro_f1": f1,
        "minutes": (time.time() - t0) / 60,
    }
    os.makedirs("results/runs", exist_ok=True)
    with open(f"results/runs/{args.tag}.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
