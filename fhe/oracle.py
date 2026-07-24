# -*- coding: utf-8 -*-
"""Plaintext oracle + acceptance metric for the FHE feasibility study (Phase B).

Pure numpy — importable outside Docker (no openfhe). Provides:
  * a plaintext reference of ONE DNAGPT-shaped transformer block (matches
    DNAGPT/dna_gpt/model/gpt.py: x + attn(ln_1(x)); x + mlp(ln_2(x))), and
  * evo2's `rel_inf` metric + dual acceptance gate (global & worst-token <= tol).

The encrypted toy block (fhe/block/toy_block_ckks.py) must reproduce block_forward()
within the gate, having decrypted only once at the very end.
"""

from __future__ import annotations

import numpy as np

TOL = 4e-2  # evo2 g4_l16 acceptance threshold


# ---- metric (evo2 tests/test_representative_oracles.py:89) --------------------
def rel_inf(dec: np.ndarray, oracle: np.ndarray) -> float:
    """max|dec-oracle| / max|oracle|; rejects non-finite / zero-oracle."""
    dec = np.asarray(dec, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    if not np.all(np.isfinite(dec)):
        return float("inf")
    denom = float(np.max(np.abs(oracle)))
    if denom == 0.0 or not np.isfinite(denom):
        return float("inf")
    return float(np.max(np.abs(dec - oracle)) / denom)


def gate(dec: np.ndarray, oracle: np.ndarray, tol: float = TOL) -> dict:
    """Dual gate: global rel_inf AND worst-per-token (row) rel_inf both <= tol."""
    dec = np.asarray(dec, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    g = rel_inf(dec, oracle)
    if dec.ndim == 2:
        worst = max(rel_inf(dec[i], oracle[i]) for i in range(dec.shape[0]))
    else:
        worst = g
    return {
        "global_rel_inf": g,
        "worst_token_rel_inf": worst,
        "tol": tol,
        "passed": bool(g <= tol and worst <= tol and np.all(np.isfinite(dec))),
    }


# ---- plaintext DNAGPT block reference ----------------------------------------
def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def gelu_tanh(x):  # nn.GELU('tanh')
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def softmax_causal(scores):  # scores [T,T]; lower-triangular allowed
    T = scores.shape[0]
    mask = np.tril(np.ones((T, T), bool))
    s = np.where(mask, scores, -1e9)
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def block_forward(x, p, n_heads=1):
    """One transformer block. x:[T,D]; p: dict of weights (numpy). Mirrors gpt.py:Block."""
    T, D = x.shape
    hd = D // n_heads
    # attn(ln_1(x))
    h = layernorm(x, p["ln1_w"], p["ln1_b"])
    qkv = h @ p["attn_w"].T + p["attn_b"]  # [T,3D]
    q, k, v = qkv[:, :D], qkv[:, D : 2 * D], qkv[:, 2 * D :]
    out = np.zeros((T, D))
    for hh in range(n_heads):
        sl = slice(hh * hd, (hh + 1) * hd)
        qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
        sc = (qh @ kh.T) / np.sqrt(hd)
        out[:, sl] = softmax_causal(sc) @ vh
    attn = out @ p["proj_w"].T + p["proj_b"]
    x = x + attn
    # mlp(ln_2(x))
    h = layernorm(x, p["ln2_w"], p["ln2_b"])
    h = gelu_tanh(h @ p["fc_w"].T + p["fc_b"])
    h = h @ p["fcp_w"].T + p["fcp_b"]
    return x + h


def toy_weights(D=8, T=4, n_heads=1, seed=0):
    """Small, well-scaled random weights so activations stay in a bounded range
    (keeps CKKS Chebyshev approximations valid)."""
    rng = np.random.default_rng(seed)
    s = 0.5 / np.sqrt(D)

    def lin(o, i):
        return rng.standard_normal((o, i)) * s, np.zeros(o)

    aw, ab = lin(3 * D, D)
    pw, pb = lin(D, D)
    fw, fb = lin(4 * D, D)
    fpw, fpb = lin(D, 4 * D)
    p = {
        "ln1_w": np.ones(D),
        "ln1_b": np.zeros(D),
        "ln2_w": np.ones(D),
        "ln2_b": np.zeros(D),
        "attn_w": aw,
        "attn_b": ab,
        "proj_w": pw,
        "proj_b": pb,
        "fc_w": fw,
        "fc_b": fb,
        "fcp_w": fpw,
        "fcp_b": fpb,
    }
    x = rng.standard_normal((T, D)) * 0.5
    return x, p


if __name__ == "__main__":
    x, p = toy_weights()
    y = block_forward(x, p)
    print(
        "plaintext block out shape:", y.shape, "range", f"[{y.min():.3f},{y.max():.3f}]"
    )
    print("self-gate:", gate(y, y))
