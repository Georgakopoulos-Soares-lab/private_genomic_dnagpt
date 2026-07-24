"""Independent NumPy reference for one bias-free DNAGPT transformer block."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def _layer_norm(
    x: np.ndarray, weight: np.ndarray, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float64)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float64)
    inv_std = 1.0 / np.sqrt(variance + eps)
    return centered * inv_std * weight, mean, variance, inv_std


def _linear(x: np.ndarray, weight_out_in: np.ndarray) -> np.ndarray:
    return x @ weight_out_in.T


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / np.sum(numerator, axis=axis, keepdims=True)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    scale = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(scale * (x + 0.044715 * x**3)))


def block_reference(
    embeddings: np.ndarray,
    weights: Mapping[str, np.ndarray],
    *,
    num_heads: int = 12,
    eps: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Return all load-bearing intermediates for one transformer block.

    All computation is NumPy float64. Linear weights use PyTorch's native
    ``[out_features, in_features]`` order.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be [T,D], got {x.shape}")
    token_count, width = x.shape
    if width % num_heads:
        raise ValueError(f"width {width} is not divisible by {num_heads} heads")
    head_dim = width // num_heads

    ln1, ln1_mean, ln1_variance, ln1_inv_std = _layer_norm(x, weights["ln1"], eps)
    qkv = _linear(ln1, weights["attn_qkv"])
    query_flat, key_flat, value_flat = np.split(qkv, 3, axis=-1)
    query = query_flat.reshape(token_count, num_heads, head_dim).transpose(1, 0, 2)
    key = key_flat.reshape(token_count, num_heads, head_dim).transpose(1, 0, 2)
    value = value_flat.reshape(token_count, num_heads, head_dim).transpose(1, 0, 2)
    scores = (query @ key.transpose(0, 2, 1)) / math.sqrt(head_dim)
    causal_mask = np.tril(np.ones((token_count, token_count), dtype=np.float64))
    masked_scores = np.where(causal_mask[None, :, :] > 0.0, scores, -np.inf)
    probabilities = _softmax(masked_scores)
    context_heads = probabilities @ value
    context_merged = context_heads.transpose(1, 0, 2).reshape(token_count, width)
    attention_projection = _linear(context_merged, weights["attn_proj"])
    residual1 = x + attention_projection

    ln2, ln2_mean, ln2_variance, ln2_inv_std = _layer_norm(
        residual1, weights["ln2"], eps
    )
    mlp_fc = _linear(ln2, weights["mlp_fc"])
    gelu = _gelu_tanh(mlp_fc)
    mlp_projection = _linear(gelu, weights["mlp_proj"])
    output = residual1 + mlp_projection

    return {
        "input_embeddings": x,
        "ln1_mean": ln1_mean,
        "ln1_variance": ln1_variance,
        "ln1_inv_std": ln1_inv_std,
        "ln1_output": ln1,
        "qkv": qkv,
        "query": query,
        "key": key,
        "value": value,
        "attention_scores": scores,
        "causal_mask": causal_mask,
        # Do not serialize masked_scores: -inf is deliberate but FHE fixtures
        # must remain finite. scores + causal_mask fully specify it.
        "attention_probabilities": probabilities,
        "attention_context_heads": context_heads,
        "attention_context_merged": context_merged,
        "attention_projection": attention_projection,
        "residual1": residual1,
        "ln2_mean": ln2_mean,
        "ln2_variance": ln2_variance,
        "ln2_inv_std": ln2_inv_std,
        "ln2_output": ln2,
        "mlp_fc": mlp_fc,
        "gelu_tanh": gelu,
        "mlp_projection": mlp_projection,
        "block_output": output,
    }


def relative_inf(reference: np.ndarray, actual: np.ndarray) -> float:
    denominator = max(float(np.max(np.abs(reference))), 1e-12)
    return float(np.max(np.abs(actual - reference)) / denominator)
