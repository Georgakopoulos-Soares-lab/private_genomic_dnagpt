"""Independent NumPy helpers for the 12-block DNAGPT classifier bridge."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

EPS = 1e-5

# The first real-weight GPU block froze these public approximation domains.
# They are deliberately retained as a comparison contract: later blocks must
# report failures instead of silently adapting a domain to a private query.
BLOCK0_GPU_DOMAINS: dict[str, tuple[float, float]] = {
    "ln1_variance_plus_eps": (0.002, 0.012),
    "ln2_variance_plus_eps": (0.50, 1.20),
    "attention_delta_s1_minus_s0": (-6.0, 0.0),
    "attention_denominator": (1.0, 2.05),
    "gelu_input": (-3.0, 3.0),
}


def layer_norm(
    value: np.ndarray, weight: np.ndarray, eps: float = EPS
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bias-free LayerNorm in float64."""
    array = np.asarray(value, dtype=np.float64)
    mean = np.mean(array, axis=-1, keepdims=True, dtype=np.float64)
    centered = array - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float64)
    inv_std = 1.0 / np.sqrt(variance + eps)
    return (
        centered * inv_std * np.asarray(weight, dtype=np.float64),
        mean,
        variance,
        inv_std,
    )


def silu(value: np.ndarray) -> np.ndarray:
    """Numerically stable SiLU in float64."""
    array = np.asarray(value, dtype=np.float64)
    sigmoid = np.empty_like(array)
    positive = array >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return array * sigmoid


def classifier_head_reference(
    hidden: np.ndarray,
    weights: Mapping[str, np.ndarray],
    *,
    eps: float = EPS,
) -> dict[str, np.ndarray]:
    """Final LayerNorm and the released bias-free DNAGPT MLM head."""
    final_ln, final_mean, final_variance, final_inv_std = layer_norm(
        hidden, weights["final_ln"], eps
    )
    head_linear = final_ln @ np.asarray(weights["head_linear"], dtype=np.float64).T
    head_silu = silu(head_linear)
    head_ln, head_ln_mean, head_ln_variance, head_ln_inv_std = layer_norm(
        head_silu, weights["head_ln"], eps
    )
    logits = head_ln @ np.asarray(weights["head_readout"], dtype=np.float64).T
    return {
        "final_ln_mean": final_mean,
        "final_ln_variance": final_variance,
        "final_ln_inv_std": final_inv_std,
        "final_ln_output": final_ln,
        "head_linear": head_linear,
        "head_silu": head_silu,
        "head_ln_mean": head_ln_mean,
        "head_ln_variance": head_ln_variance,
        "head_ln_inv_std": head_ln_inv_std,
        "head_ln_output": head_ln,
        "logits": logits,
    }


def finite_range(value: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("range input must be non-empty and finite")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return {
        "min": minimum,
        "max": maximum,
        "abs_max": max(abs(minimum), abs(maximum)),
        "count": int(array.size),
    }


def t2_attention_schedule(
    attention_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the encrypted-source-0 shift delta and denominator for T=2.

    ``attention_scores`` must have layout ``[heads, 2, 2]``. Only the second
    query has two causal sources; its softmax is exactly:

      delta = score(source=1) - score(source=0)
      denominator = 1 + exp(delta)
    """
    scores = np.asarray(attention_scores, dtype=np.float64)
    if scores.ndim != 3 or scores.shape[1:] != (2, 2):
        raise ValueError(f"expected attention scores [heads,2,2], got {scores.shape}")
    delta = scores[:, 1, 1] - scores[:, 1, 0]
    denominator = 1.0 + np.exp(delta)
    if not np.isfinite(denominator).all():
        raise ValueError("T=2 attention denominator overflowed")
    return delta, denominator


def covers(observed: Mapping[str, float | int], domain: tuple[float, float]) -> bool:
    """Whether a public fixed domain contains an observed finite range."""
    return float(observed["min"]) >= domain[0] and float(observed["max"]) <= domain[1]


def outward_domain(
    observed: Mapping[str, float | int],
    margin: float,
    *,
    positive: bool = False,
    symmetric: bool = False,
) -> list[float]:
    """Build an explicitly sample-calibrated public-domain assumption."""
    if margin <= 1.0:
        raise ValueError("margin must exceed one")
    minimum = float(observed["min"])
    maximum = float(observed["max"])
    if positive:
        if minimum <= 0:
            raise ValueError("positive domain requires a positive observed minimum")
        return [minimum / margin, maximum * margin]
    if symmetric:
        radius = max(abs(minimum), abs(maximum)) * margin
        return [-radius, radius]
    lower = minimum * margin if minimum < 0 else minimum / margin
    upper = maximum * margin if maximum > 0 else maximum / margin
    return [lower, upper]


def block_calibration(
    block_oracle: Mapping[str, np.ndarray],
    *,
    eps: float = EPS,
    margin: float = 1.25,
) -> dict[str, Any]:
    """Ranges, proposed domains, and block-0-domain failures for one T=2 block."""
    delta, denominator = t2_attention_schedule(block_oracle["attention_scores"])
    ranges = {
        "ln1_variance_plus_eps": finite_range(
            np.asarray(block_oracle["ln1_variance"]) + eps
        ),
        "ln1_normalized_output": finite_range(block_oracle["ln1_output"]),
        "attention_delta_s1_minus_s0": finite_range(delta),
        "attention_denominator": finite_range(denominator),
        "ln2_variance_plus_eps": finite_range(
            np.asarray(block_oracle["ln2_variance"]) + eps
        ),
        "ln2_normalized_output": finite_range(block_oracle["ln2_output"]),
        "gelu_input": finite_range(block_oracle["mlp_fc"]),
        "gelu_output": finite_range(block_oracle["gelu_tanh"]),
        "block_output": finite_range(block_oracle["block_output"]),
    }
    coverage = {
        field: covers(ranges[field], domain)
        for field, domain in BLOCK0_GPU_DOMAINS.items()
    }
    recommended = {
        "ln1_variance_plus_eps": outward_domain(
            ranges["ln1_variance_plus_eps"], margin, positive=True
        ),
        "ln1_normalized_output": outward_domain(
            ranges["ln1_normalized_output"], margin, symmetric=True
        ),
        "attention_delta_s1_minus_s0": outward_domain(
            ranges["attention_delta_s1_minus_s0"], margin
        ),
        "attention_denominator": outward_domain(
            ranges["attention_denominator"], margin, positive=True
        ),
        "ln2_variance_plus_eps": outward_domain(
            ranges["ln2_variance_plus_eps"], margin, positive=True
        ),
        "ln2_normalized_output": outward_domain(
            ranges["ln2_normalized_output"], margin, symmetric=True
        ),
        "gelu_input": outward_domain(ranges["gelu_input"], margin, symmetric=True),
        "silu_input": None,
    }
    return {
        "ranges": ranges,
        "recommended_public_domains": recommended,
        "block0_gpu_fixed_domain_coverage": coverage,
        "block0_gpu_fixed_domain_passed": all(coverage.values()),
        "failed_block0_gpu_fixed_domains": [
            field for field, passed in coverage.items() if not passed
        ],
    }
