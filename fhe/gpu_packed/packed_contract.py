"""Float64 contract for the interleaved D=768/T=8 packed attention gate.

The encrypted layout is

    slot(copy, channel, token) = copy * (1024 * 8) + channel * 8 + token

for four identical copies.  All calibration values are computed from an
explicit public fixture before encryption; encrypted evaluation never adapts
an approximation interval or shift to a private query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.polynomial import Chebyshev


D = 768
T = 8
HEADS = 12
HEAD_DIM = 64
PACK_WIDTH = 1024
COPIES = 4
P = T
SLOTS_PER_COPY = PACK_WIDTH * P
SLOTS = COPIES * SLOTS_PER_COPY
BSGS_N1 = 32
BSGS_N2 = 32
EPS = 1e-5
TOL = 4e-2
PUBLIC_SHIFT_GUARD = 0.25
LN_VARIANCE_SCALE = 1.0 / 256.0
DEG_LN_INVSQRT = 7
DEG_EXP = 15
DEG_RECIPROCAL = 15
SCHEMA = "dnagpt.gpu_packed.public_t8_contract.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    REPO_ROOT
    / "checkpoints/fhe_exports"
    / "gsr_pos0_block0_t8_d63353abdc1a_52d046d1fcf0"
)
DEFAULT_CONTRACT = Path(__file__).with_name("public_t8_contract.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_fixture(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != "dnagpt.realweights.block0.v1":
        raise ValueError(f"unexpected fixture schema {manifest['schema']!r}")
    model = manifest["model"]
    tokenization = manifest["tokenization"]
    if (
        model["embedding_dim"] != D
        or model["heads"] != HEADS
        or tokenization["exported_token_count_T"] != T
        or manifest["format"]["export_dtype"] != "float64"
    ):
        raise ValueError("fixture is not the exact D=768/H=12/T=8/float64 contract")

    arrays: dict[str, np.ndarray] = {}
    for name, metadata in manifest["arrays"].items():
        path = directory / metadata["file"]
        if sha256(path) != metadata["sha256"]:
            raise ValueError(f"fixture array SHA-256 mismatch: {name}")
        if path.stat().st_size != metadata["bytes"]:
            raise ValueError(f"fixture array byte count mismatch: {name}")
        value = np.fromfile(path, dtype="<f8").reshape(metadata["shape"])
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite fixture array: {name}")
        arrays[name] = value
    return manifest, arrays


def pack(values: np.ndarray) -> np.ndarray:
    """Pack [T,D] into four identical interleaved copies."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (T, D):
        raise ValueError(f"expected [{T},{D}], got {matrix.shape}")
    one = np.zeros((PACK_WIDTH, P), dtype=np.float64)
    one[:D, :] = matrix.T
    return np.tile(one[None, :, :], (COPIES, 1, 1)).reshape(SLOTS)


def unpack(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64)
    if flat.shape != (SLOTS,):
        raise ValueError(f"expected {SLOTS} packed slots, got {flat.shape}")
    copies = flat.reshape(COPIES, PACK_WIDTH, P)
    for copy in range(1, COPIES):
        np.testing.assert_allclose(copies[copy], copies[0], rtol=0.0, atol=0.0)
    return copies[0, :D, :].T.copy()


def rotate(values: np.ndarray, index: int) -> np.ndarray:
    """OpenFHE/FIDES logical left rotation on the flattened slot vector."""
    return np.roll(np.asarray(values), -index)


def _channel_plain(values: np.ndarray) -> np.ndarray:
    channel_values = np.asarray(values, dtype=np.float64)
    if channel_values.shape != (PACK_WIDTH, P):
        raise ValueError("channel plaintext must be [1024,8]")
    return np.tile(channel_values[None, :, :], (COPIES, 1, 1)).reshape(SLOTS)


def bsgs_matmul(values: np.ndarray, weight_out_in: np.ndarray) -> np.ndarray:
    """Exact float64 simulation of the 32x32 interleaved BSGS graph."""
    weight = np.asarray(weight_out_in, dtype=np.float64)
    if weight.shape != (D, D):
        raise ValueError(f"expected [{D},{D}] weight, got {weight.shape}")
    padded = np.zeros((PACK_WIDTH, PACK_WIDTH), dtype=np.float64)
    padded[:D, :D] = weight
    baby = [rotate(values, small * P) for small in range(BSGS_N1)]
    result = np.zeros(SLOTS, dtype=np.float64)
    for giant in range(BSGS_N2):
        inner = np.zeros(SLOTS, dtype=np.float64)
        giant_offset = BSGS_N1 * giant
        for small in range(BSGS_N1):
            diagonal = giant_offset + small
            rows = np.arange(PACK_WIDTH)
            rolled_rows = (rows - giant_offset) % PACK_WIDTH
            columns = (rolled_rows + diagonal) % PACK_WIDTH
            diagonal_values = padded[rolled_rows, columns]
            plain = np.repeat(diagonal_values[:, None], P, axis=1)
            inner += baby[small] * _channel_plain(plain)
        result += rotate(inner, giant_offset * P)
    return result


def _active_channel_token_mask(min_query: int = 0) -> np.ndarray:
    plain = np.zeros((PACK_WIDTH, P), dtype=np.float64)
    plain[:D, min_query:] = 1.0
    return _channel_plain(plain)


def _head_anchor_mask(min_query: int = 0) -> np.ndarray:
    plain = np.zeros((PACK_WIDTH, P), dtype=np.float64)
    plain[np.arange(HEADS) * HEAD_DIM, min_query:] = 1.0
    return _channel_plain(plain)


def _sum_channels_broadcast(values: np.ndarray) -> np.ndarray:
    shaped = np.asarray(values).reshape(COPIES, PACK_WIDTH, P)
    sums = shaped.sum(axis=1, keepdims=True)
    return np.broadcast_to(sums, shaped.shape).copy().reshape(SLOTS)


def _head_sum_to_anchors(values: np.ndarray, min_query: int) -> np.ndarray:
    shaped = np.asarray(values).reshape(COPIES, PACK_WIDTH, P)
    output = np.zeros_like(shaped)
    for head in range(HEADS):
        first = head * HEAD_DIM
        output[:, first, min_query:] = shaped[
            :, first : first + HEAD_DIM, min_query:
        ].sum(axis=1)
    return output.reshape(SLOTS)


def _broadcast_anchors(values: np.ndarray) -> np.ndarray:
    shaped = np.asarray(values).reshape(COPIES, PACK_WIDTH, P)
    output = np.zeros_like(shaped)
    for head in range(HEADS):
        first = head * HEAD_DIM
        output[:, first : first + HEAD_DIM, :] = shaped[:, first : first + 1, :]
    return output.reshape(SLOTS)


def _anchor_plain(values_heads_tokens: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values_heads_tokens, dtype=np.float64)
    if matrix.shape != (HEADS, T):
        raise ValueError(f"expected [{HEADS},{T}], got {matrix.shape}")
    plain = np.zeros((PACK_WIDTH, P), dtype=np.float64)
    plain[np.arange(HEADS) * HEAD_DIM, :] = matrix
    return _channel_plain(plain)


def _chebyshev(
    function: Callable[[np.ndarray], np.ndarray],
    degree: int,
    domain: tuple[float, float],
) -> Callable[[np.ndarray], np.ndarray]:
    polynomial = Chebyshev.interpolate(function, degree, domain=domain)
    return lambda value: polynomial(value)


def _domains(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    variance = arrays["oracle.ln1_variance"].reshape(-1) + EPS
    scores = arrays["oracle.attention_scores"]
    shifts = np.empty((HEADS, T), dtype=np.float64)
    deltas: list[np.ndarray] = []
    denominators = np.empty((HEADS, T), dtype=np.float64)
    for query in range(T):
        causal = scores[:, query, : query + 1]
        shifts[:, query] = causal.max(axis=1) + PUBLIC_SHIFT_GUARD
        shifted = causal - shifts[:, query, None]
        deltas.append(shifted.reshape(-1))
        denominators[:, query] = np.exp(shifted).sum(axis=1)
    all_deltas = np.concatenate(deltas)

    variance_lo = math.floor((float(variance.min()) / 1.25) / 1e-6) * 1e-6
    variance_hi = math.ceil((float(variance.max()) * 1.25) / 1e-5) * 1e-5
    delta_lo = math.floor((float(all_deltas.min()) * 1.25) * 10.0) / 10.0
    denominator_lo = math.floor((float(denominators.min()) / 1.25) * 100.0) / 100.0
    denominator_hi = math.ceil((float(denominators.max()) * 1.25) * 10.0) / 10.0
    return {
        "ln1_variance_plus_eps": [variance_lo, variance_hi],
        "ln1_variance_public_scale": LN_VARIANCE_SCALE,
        "ln1_scaled_variance": [
            variance_lo / LN_VARIANCE_SCALE,
            variance_hi / LN_VARIANCE_SCALE,
        ],
        "attention_public_shift_guard": PUBLIC_SHIFT_GUARD,
        "attention_public_shifts_heads_by_query": shifts.tolist(),
        "attention_shifted_score": [delta_lo, 0.0],
        "attention_denominator": [denominator_lo, denominator_hi],
        "observed": {
            "ln1_variance_plus_eps": [
                float(variance.min()),
                float(variance.max()),
            ],
            "attention_shifted_score": [
                float(all_deltas.min()),
                float(all_deltas.max()),
            ],
            "attention_denominator": [
                float(denominators.min()),
                float(denominators.max()),
            ],
        },
    }


def simulate(
    fixture_dir: Path, *, approximate: bool
) -> tuple[dict[str, float | bool], dict[str, Any]]:
    manifest, arrays = _load_fixture(fixture_dir)
    domains = _domains(arrays)
    x = pack(arrays["input.embeddings"])

    mean = _sum_channels_broadcast(x) / D
    centered = (x - mean) * _active_channel_token_mask()
    variance = _sum_channels_broadcast(centered * centered) / D + EPS
    if approximate:
        scaled_domain = tuple(domains["ln1_scaled_variance"])
        inv_poly = _chebyshev(
            lambda value: 1.0 / np.sqrt(value),
            DEG_LN_INVSQRT,
            scaled_domain,
        )
        inverse = inv_poly(variance / LN_VARIANCE_SCALE) / math.sqrt(LN_VARIANCE_SCALE)
    else:
        inverse = 1.0 / np.sqrt(variance)
    ln1_weight = arrays["weights.ln1"]
    weight_plain = np.zeros((PACK_WIDTH, P), dtype=np.float64)
    weight_plain[:D, :] = ln1_weight[:, None]
    normalized = centered * inverse * _channel_plain(weight_plain)

    qkv_weight = arrays["weights.attn_qkv"]
    query = bsgs_matmul(normalized, qkv_weight[:D])
    key = bsgs_matmul(normalized, qkv_weight[D : 2 * D])
    value = bsgs_matmul(normalized, qkv_weight[2 * D :])

    shifts = np.asarray(
        domains["attention_public_shifts_heads_by_query"], dtype=np.float64
    )
    shift_plain = _anchor_plain(shifts)
    exp_domain = tuple(domains["attention_shifted_score"])
    reciprocal_domain = tuple(domains["attention_denominator"])
    exp_fn = _chebyshev(np.exp, DEG_EXP, exp_domain) if approximate else np.exp
    reciprocal_fn = (
        _chebyshev(lambda value: 1.0 / value, DEG_RECIPROCAL, reciprocal_domain)
        if approximate
        else lambda value: 1.0 / value
    )

    numerators: list[np.ndarray] = []
    for offset in range(T):
        aligned_key = rotate(key, -offset)
        products = query * aligned_key * _active_channel_token_mask(offset)
        anchors = _head_sum_to_anchors(products, offset)
        anchors *= 1.0 / math.sqrt(HEAD_DIM)
        active_anchors = _head_anchor_mask(offset)
        # Inactive slots stay at zero, which is inside the public exp domain.
        # In particular, do not subtract shifts from masked causal rows.
        shifted = anchors - shift_plain * active_anchors
        numerator = exp_fn(shifted) * active_anchors
        numerators.append(numerator)
    denominator = np.sum(numerators, axis=0)
    all_anchors = _head_anchor_mask()
    # Reciprocal is evaluated at public 1.0 outside active anchors so every
    # evaluated slot remains inside the fixed positive domain.
    denominator_guarded = denominator + (1.0 - all_anchors)
    reciprocal = reciprocal_fn(denominator_guarded) * all_anchors

    context = np.zeros(SLOTS, dtype=np.float64)
    for offset, numerator in enumerate(numerators):
        weight_anchors = numerator * reciprocal
        weights = _broadcast_anchors(weight_anchors)
        aligned_value = rotate(value, -offset)
        context += weights * aligned_value * _active_channel_token_mask(offset)
    projection = bsgs_matmul(context, arrays["weights.attn_proj"])
    actual = unpack(projection)
    expected = arrays["oracle.attention_projection"]
    error = np.abs(actual - expected)
    global_rel = float(error.max() / max(float(np.abs(expected).max()), 1e-15))
    worst_token = max(
        float(error[token].max() / max(float(np.abs(expected[token]).max()), 1e-15))
        for token in range(T)
    )
    metrics: dict[str, float | bool] = {
        "global_rel_inf": global_rel,
        "worst_token_rel_inf": worst_token,
        "max_abs_error": float(error.max()),
        "all_finite": bool(np.isfinite(actual).all()),
        "tol": TOL,
        "passed": bool(
            np.isfinite(actual).all() and global_rel <= TOL and worst_token <= TOL
        ),
    }
    identity = {
        "fixture_manifest_sha256": sha256(fixture_dir / "manifest.json"),
        "checkpoint_sha256": manifest["model"]["checkpoint_sha256"],
        "T": manifest["tokenization"]["exported_token_count_T"],
    }
    return metrics, {"domains": domains, "identity": identity}


def build_contract(fixture_dir: Path) -> dict[str, Any]:
    exact, details = simulate(fixture_dir, approximate=False)
    approximate, _ = simulate(fixture_dir, approximate=True)
    if not exact["passed"] or exact["global_rel_inf"] > 1e-12:
        raise AssertionError(f"packed exact-layout parity failed: {exact}")
    if not approximate["passed"]:
        raise AssertionError(f"packed approximate preflight failed: {approximate}")
    return {
        "schema": SCHEMA,
        "status": (
            "[A] fixed public T=8 fixture calibration; never selected from a "
            "decrypted private query"
        ),
        "layout": {
            "formula": "copy*(1024*P)+channel*P+token",
            "D": D,
            "T": T,
            "P": P,
            "pack_width": PACK_WIDTH,
            "copies": COPIES,
            "slots": SLOTS,
            "input_ciphertexts": 1,
            "output_ciphertexts": 1,
            "bsgs": [BSGS_N1, BSGS_N2],
        },
        "identity": details["identity"],
        "domains": details["domains"],
        "degrees": {
            "ln1_inverse_sqrt": DEG_LN_INVSQRT,
            "attention_exp": DEG_EXP,
            "attention_reciprocal": DEG_RECIPROCAL,
        },
        "exact_float64_layout_parity": exact,
        "approximate_float64_preflight": approximate,
        "security_boundary": {
            "calibration_is_public_preprocessing": True,
            "private_query_domain_adaptation": False,
            "intermediate_decrypts": 0,
            "final_decrypts": 1,
        },
    }


def write_exclusive(path: Path, result: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite public contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build_contract(args.fixture.resolve())
    if args.check:
        expected = json.loads(args.output.read_text(encoding="utf-8"))
        if result != expected:
            raise SystemExit("public packed contract is stale")
    else:
        write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
