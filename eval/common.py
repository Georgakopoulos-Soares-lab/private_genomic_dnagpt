# -*- coding: utf-8 -*-
"""Shared DNAGPT loading + inference helpers for local evaluation.

Reuses the upstream `dna_gpt` package (cloned into ../DNAGPT) but is
float32/mps-safe: upstream test.py defaults to float16+cuda which breaks on Mac.
"""

import os
import sys

import torch

# Make the cloned upstream package importable.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DNAGPT_DIR = os.path.join(REPO_ROOT, "DNAGPT")
if DNAGPT_DIR not in sys.path:
    sys.path.insert(0, DNAGPT_DIR)

from dna_gpt.model import DNAGPT  # noqa: E402
from dna_gpt.tokenizer import KmerTokenizer  # noqa: E402

# Special-token vocabulary, copied verbatim from upstream test.py:get_model.
SPECIAL_TOKENS = (
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    + ["+", "-", "*", "/", "=", "&", "|", "!"]
    + ["M", "B"]
    + ["P"]
    + ["R", "I", "K", "L", "O", "Q", "S", "U", "V"]
    + ["W", "Y", "X", "Z"]
)


def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model(model_name):
    """Build tokenizer + model skeleton (mirrors upstream test.py:get_model)."""
    dynamic = model_name not in ("dna_gpt0.1b_h",)
    tokenizer = KmerTokenizer(6, SPECIAL_TOKENS, dynamic_kmer=dynamic)
    model = DNAGPT.from_name(model_name, len(tokenizer))
    return model, tokenizer


def load_model(model_name, weight_path, device=None, dtype=torch.float32):
    device = pick_device(device)
    model, tokenizer = get_model(model_name)
    state = torch.load(weight_path, map_location="cpu")
    state = state["model"] if "model" in state else state
    model.load_state_dict(state, strict=False)
    model.to(device=device, dtype=dtype).eval()
    torch.set_grad_enabled(False)
    return model, tokenizer, device


@torch.no_grad()
def classify(model, tokenizer, seq, max_len, device):
    """Return decoded next-token char for a GSR prompt: 'N'=real, 'A'=fake.

    Template matches scripts/classification.sh: <R>{seq}<=><R>
    """
    prompt = f"<R>{seq}<=><R>"
    ids = tokenizer.encode(prompt, max_len=max_len, device=device)[None]
    logits = model(ids)[:, -1]
    tok_id = int(torch.argmax(logits, dim=-1)[0])
    return tokenizer.decode([tok_id])


@torch.no_grad()
def regress(model, tokenizer, prompt, numbers, max_len, device):
    """Predict a scalar (mirrors upstream test.py:regression)."""
    dtype = next(model.parameters()).dtype
    ids = tokenizer.encode(prompt, max_len=max_len, device=device)
    prompt_len = len(ids)
    x = ids[None]
    num_len = len(numbers)
    token_length = torch.tensor(
        prompt_len + num_len - 1, device=device, dtype=torch.long
    )[None, None]
    number_block = torch.full((x.shape[0],), x.shape[1] - 2)
    num = torch.tensor(numbers, device=device, dtype=dtype)
    y = model(x, num, token_length, number_block=number_block)
    return float(y[0][0, 0, 0])
