"""Validate the frozen public domains and optimized T=2 sigmoid schedule locally.

This is a plaintext preflight, not FHE evidence. It uses NumPy's Chebyshev
interpolant at the exact degrees/domains compiled into
real_dnagpt_fides_sigmoid.cpp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

D = 768
T = 2
HEADS = 12
HEAD_DIM = 64
EPS = 1e-5
TOL = 4e-2
EXPECTED_MANIFEST = "8d20a2841ee29a7a386171f1bd173b7189144359ce8f91ea5eb21cc60c0a78fe"

DOMAINS = {
    "ln1": (0.002, 0.012, 27),
    "ln2": (0.50, 1.20, 15),
    "sigmoid": (-6.0, 0.0, 13),
    "gelu": (-3.0, 3.0, 15),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_array(root: Path, manifest: dict, name: str) -> np.ndarray:
    metadata = manifest["arrays"][name]
    path = root / metadata["file"]
    if sha256(path) != metadata["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")
    values = np.fromfile(path, dtype="<f8")
    if values.size != metadata["elements"]:
        raise ValueError(f"element count mismatch: {path}")
    return values.reshape(metadata["shape"])


def cheb(function, domain: tuple[float, float, int]):
    lo, hi, degree = domain
    polynomial = np.polynomial.Chebyshev.interpolate(function, degree, domain=[lo, hi])
    return polynomial


def layernorm(
    value: np.ndarray,
    weight: np.ndarray,
    inverse_sqrt,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(value, axis=-1, keepdims=True)
    centered = value - mean
    variance_plus_eps = np.mean(centered * centered, axis=-1, keepdims=True) + EPS
    return centered * inverse_sqrt(variance_plus_eps) * weight, variance_plus_eps


def rel_inf(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(
        np.max(np.abs(reference - actual))
        / max(float(np.max(np.abs(reference))), 1e-15)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=Path(
            "checkpoints/fhe_exports/gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0"
        ),
    )
    args = parser.parse_args()
    root = args.fixture.resolve()
    manifest_path = root / "manifest.json"
    if sha256(manifest_path) != EXPECTED_MANIFEST:
        raise ValueError("refusing unpinned fixture manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    x = read_array(root, manifest, "input.embeddings")
    ln1_weight = read_array(root, manifest, "weights.ln1")
    qkv_weight = read_array(root, manifest, "weights.attn_qkv")
    projection_weight = read_array(root, manifest, "weights.attn_proj")
    ln2_weight = read_array(root, manifest, "weights.ln2")
    fc_weight = read_array(root, manifest, "weights.mlp_fc")
    mlp_projection_weight = read_array(root, manifest, "weights.mlp_proj")
    reference = read_array(root, manifest, "oracle.block_output")

    ln1_inverse = cheb(lambda value: 1.0 / np.sqrt(value), DOMAINS["ln1"])
    ln2_inverse = cheb(lambda value: 1.0 / np.sqrt(value), DOMAINS["ln2"])
    sigmoid = cheb(
        lambda value: 1.0 / (1.0 + np.exp(-value)),
        DOMAINS["sigmoid"],
    )
    gelu = cheb(
        lambda value: 0.5
        * value
        * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3))),
        DOMAINS["gelu"],
    )

    ln1, var1 = layernorm(x, ln1_weight, ln1_inverse)
    qkv = ln1 @ qkv_weight.T
    query, key, value = np.split(qkv, 3, axis=-1)
    query = query.reshape(T, HEADS, HEAD_DIM).transpose(1, 0, 2)
    key = key.reshape(T, HEADS, HEAD_DIM).transpose(1, 0, 2)
    value = value.reshape(T, HEADS, HEAD_DIM).transpose(1, 0, 2)

    context = np.empty_like(value)
    context[:, 0, :] = value[:, 0, :]
    score_10 = np.sum(query[:, 1, :] * key[:, 0, :], axis=-1) / math.sqrt(HEAD_DIM)
    score_11 = np.sum(query[:, 1, :] * key[:, 1, :], axis=-1) / math.sqrt(HEAD_DIM)
    delta = score_11 - score_10
    weight_1 = sigmoid(delta)
    context[:, 1, :] = value[:, 0, :] + weight_1[:, None] * (
        value[:, 1, :] - value[:, 0, :]
    )

    merged = context.transpose(1, 0, 2).reshape(T, D)
    attention_projection = merged @ projection_weight.T
    residual1 = x + attention_projection
    ln2, var2 = layernorm(residual1, ln2_weight, ln2_inverse)
    hidden_pre = ln2 @ fc_weight.T
    hidden = gelu(hidden_pre)
    output = residual1 + hidden @ mlp_projection_weight.T

    checks = {
        "ln1_variance_plus_eps": [
            float(np.min(var1)),
            float(np.max(var1)),
        ],
        "ln2_variance_plus_eps": [
            float(np.min(var2)),
            float(np.max(var2)),
        ],
        "attention_score_delta_s1_minus_s0": [
            float(np.min(delta)),
            float(np.max(delta)),
        ],
        "gelu_input": [
            float(np.min(hidden_pre)),
            float(np.max(hidden_pre)),
        ],
    }
    bounds = {
        "ln1_variance_plus_eps": DOMAINS["ln1"][:2],
        "ln2_variance_plus_eps": DOMAINS["ln2"][:2],
        "attention_score_delta_s1_minus_s0": DOMAINS["sigmoid"][:2],
        "gelu_input": DOMAINS["gelu"][:2],
    }
    domains_pass = all(
        observed[0] >= bounds[name][0] and observed[1] <= bounds[name][1]
        for name, observed in checks.items()
    )
    global_rel_inf = rel_inf(reference, output)
    worst_token_rel_inf = max(
        rel_inf(reference[token], output[token]) for token in range(T)
    )
    passed = (
        domains_pass
        and np.isfinite(output).all()
        and global_rel_inf <= TOL
        and worst_token_rel_inf <= TOL
    )
    print(
        json.dumps(
            {
                "scope": "plaintext preflight; not FHE evidence",
                "fixture_manifest_sha256": EXPECTED_MANIFEST,
                "public_domain_checks": checks,
                "domains_pass": domains_pass,
                "global_rel_inf": global_rel_inf,
                "worst_token_rel_inf": worst_token_rel_inf,
                "tol": TOL,
                "passed": passed,
            },
            indent=2,
        )
    )
    raise SystemExit(0 if passed else 5)


if __name__ == "__main__":
    main()
