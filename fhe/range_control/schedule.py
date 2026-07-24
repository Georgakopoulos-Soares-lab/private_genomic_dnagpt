"""Reusable numerical pieces for a CKKS-friendly DNAGPT circuit schedule.

This module does not perform FHE.  It evaluates the exact real-weight graph
with the same polynomial functions that can later be evaluated on ciphertexts.
Every approximation domain is public and fixed before the simulated query is
evaluated.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Iterable

import numpy as np
from numpy.polynomial import Chebyshev

EPS = 1e-5
DEFAULT_TOLERANCE = 4e-2
DEFAULT_DEGREES = {
    "inverse_sqrt": 63,
    "sigmoid": 47,
    "gelu": 127,
    "silu": 95,
}

# Smallest degrees from a fixed candidate set whose isolated public-fixture
# approximation error is <= 1e-4, capped at degree 127.  GELU in blocks 3 and
# 4 reaches that cap before the target.  The full cumulative lineage is
# independently gated after every block; the isolated target is not final.
BLOCK_DEGREES: tuple[dict[str, int], ...] = (
    {"ln1_inverse_sqrt": 7, "sigmoid": 9, "ln2_inverse_sqrt": 7, "gelu": 15},
    {"ln1_inverse_sqrt": 7, "sigmoid": 11, "ln2_inverse_sqrt": 7, "gelu": 39},
    {"ln1_inverse_sqrt": 7, "sigmoid": 13, "ln2_inverse_sqrt": 7, "gelu": 63},
    {"ln1_inverse_sqrt": 7, "sigmoid": 27, "ln2_inverse_sqrt": 9, "gelu": 127},
    {"ln1_inverse_sqrt": 13, "sigmoid": 13, "ln2_inverse_sqrt": 15, "gelu": 127},
    {"ln1_inverse_sqrt": 39, "sigmoid": 15, "ln2_inverse_sqrt": 39, "gelu": 27},
    {"ln1_inverse_sqrt": 39, "sigmoid": 27, "ln2_inverse_sqrt": 39, "gelu": 23},
    {"ln1_inverse_sqrt": 31, "sigmoid": 13, "ln2_inverse_sqrt": 39, "gelu": 17},
    {"ln1_inverse_sqrt": 63, "sigmoid": 13, "ln2_inverse_sqrt": 63, "gelu": 19},
    {"ln1_inverse_sqrt": 63, "sigmoid": 9, "ln2_inverse_sqrt": 63, "gelu": 21},
    {"ln1_inverse_sqrt": 79, "sigmoid": 23, "ln2_inverse_sqrt": 79, "gelu": 23},
    {"ln1_inverse_sqrt": 79, "sigmoid": 17, "ln2_inverse_sqrt": 63, "gelu": 47},
)
HEAD_DEGREES = {
    "final_ln_inverse_sqrt": 47,
    "silu": 95,
    "head_ln_inverse_sqrt": 13,
}
ISOLATED_APPROXIMATION_TARGET = 1e-4
POLYNOMIAL_DEGREE_CAP = 127
ATTENTION_ZERO_GUARD = 0.25


class DomainViolation(ValueError):
    """Raised when a simulated private value leaves its fixed public domain."""


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


def domain_check(
    value: np.ndarray,
    domain: Iterable[float],
    *,
    name: str,
) -> dict[str, Any]:
    lower, upper = (float(item) for item in domain)
    if not lower < upper:
        raise ValueError(f"{name}: invalid public domain [{lower}, {upper}]")
    observed = finite_range(value)
    passed = float(observed["min"]) >= lower and float(observed["max"]) <= upper
    return {
        "name": name,
        "public_domain": [lower, upper],
        "observed": observed,
        "passed": passed,
        "lower_overflow": max(lower - float(observed["min"]), 0.0),
        "upper_overflow": max(float(observed["max"]) - upper, 0.0),
    }


def require_public_domain(
    value: np.ndarray,
    domain: Iterable[float],
    *,
    name: str,
) -> dict[str, Any]:
    result = domain_check(value, domain, name=name)
    if not result["passed"]:
        raise DomainViolation(
            f"{name}: observed [{result['observed']['min']}, "
            f"{result['observed']['max']}] leaves fixed public domain "
            f"{result['public_domain']}"
        )
    return result


def attention_delta_public_domain(
    calibrated_domain: Iterable[float],
    *,
    zero_guard: float = ATTENTION_ZERO_GUARD,
) -> list[float]:
    """Add a fixed public zero-crossing guard to a calibrated delta domain.

    Several exact public-fixture delta domains end just below zero.  A small,
    predeclared guard prevents ordinary polynomial error from turning that
    boundary into query-adaptive control flow.  The guard is a public circuit
    constant, not selected from the value being evaluated.
    """
    lower, upper = (float(item) for item in calibrated_domain)
    if not lower < upper:
        raise ValueError("attention delta domain must be ordered")
    if zero_guard <= 0.0:
        raise ValueError("attention zero guard must be positive")
    return [min(lower, -zero_guard), max(upper, zero_guard)]


def sigmoid_stable(value: np.ndarray) -> np.ndarray:
    """Overflow-free sigmoid, used only to build/reference a public polynomial."""
    array = np.asarray(value, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def gelu_tanh(value: np.ndarray) -> np.ndarray:
    """The exact tanh-GELU semantics used by the released DNAGPT model."""
    array = np.asarray(value, dtype=np.float64)
    scale = math.sqrt(2.0 / math.pi)
    return 0.5 * array * (1.0 + np.tanh(scale * (array + 0.044715 * array**3)))


def silu(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return array * sigmoid_stable(array)


def polynomial_depth_proxy(degree: int) -> int:
    """Balanced-power multiplicative-depth lower-bound proxy.

    OpenFHE/FIDES evaluation details add implementation-specific level costs.
    This value is deliberately labeled a proxy, not a measured ciphertext
    level count.
    """
    if degree < 1:
        raise ValueError("polynomial degree must be positive")
    return int(math.ceil(math.log2(degree + 1)))


@lru_cache(maxsize=None)
def _interpolant(
    function_name: str,
    degree: int,
    lower: float,
    upper: float,
) -> Chebyshev:
    functions: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "inverse_sqrt": lambda value: 1.0 / np.sqrt(value),
        "sigmoid": sigmoid_stable,
        "gelu": gelu_tanh,
        "silu": silu,
    }
    try:
        function = functions[function_name]
    except KeyError as exc:
        raise ValueError(f"unknown approximation function: {function_name}") from exc
    return Chebyshev.interpolate(
        function,
        degree,
        domain=[float(lower), float(upper)],
    )


def evaluate_public_polynomial(
    function_name: str,
    value: np.ndarray,
    domain: Iterable[float],
    degree: int,
    *,
    name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    check = require_public_domain(value, domain, name=name)
    lower, upper = check["public_domain"]
    polynomial = _interpolant(function_name, degree, lower, upper)
    result = np.asarray(polynomial(np.asarray(value, dtype=np.float64)))
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name}: polynomial produced a non-finite value")
    check.update(
        {
            "function": function_name,
            "degree": degree,
            "multiplicative_depth_proxy": polynomial_depth_proxy(degree),
        }
    )
    return result, check


def public_power_of_two_scale(domain: Iterable[float]) -> float:
    """Choose a CKKS-friendly public scale near the domain geometric mean."""
    lower, upper = (float(item) for item in domain)
    if not 0.0 < lower < upper:
        raise ValueError("inverse-sqrt domain must be positive and ordered")
    exponent = round(math.log2(math.sqrt(lower * upper)))
    return float(2.0**exponent)


def public_scaled_inverse_sqrt(
    value: np.ndarray,
    domain: Iterable[float],
    degree: int = DEFAULT_DEGREES["inverse_sqrt"],
    *,
    name: str = "inverse_sqrt",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Approximate ``1/sqrt(value)`` after an exact public scaling identity.

    With public ``s``:

        1/sqrt(value) = (1/sqrt(s)) * 1/sqrt(value/s)

    The Chebyshev polynomial therefore sees a normalized, fixed interval.  No
    scale or interval is selected from the simulated private value.
    """
    check = require_public_domain(value, domain, name=name)
    lower, upper = check["public_domain"]
    scale = public_power_of_two_scale((lower, upper))
    normalized_domain = (lower / scale, upper / scale)
    normalized = np.asarray(value, dtype=np.float64) / scale
    polynomial = _interpolant(
        "inverse_sqrt",
        degree,
        normalized_domain[0],
        normalized_domain[1],
    )
    result = np.asarray(polynomial(normalized)) / math.sqrt(scale)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name}: inverse-sqrt polynomial is non-finite")
    check.update(
        {
            "function": "inverse_sqrt",
            "degree": degree,
            "multiplicative_depth_proxy": polynomial_depth_proxy(degree),
            "public_power_of_two_scale": scale,
            "normalized_public_domain": list(normalized_domain),
            "identity": ("inv_sqrt(v) = inv_sqrt(v/public_scale) / sqrt(public_scale)"),
        }
    )
    return result, check


def t2_attention_probabilities(
    scores: np.ndarray,
    delta_domain: Iterable[float],
    degree: int = DEFAULT_DEGREES["sigmoid"],
    *,
    name: str = "attention_delta",
) -> tuple[np.ndarray, dict[str, Any]]:
    """T=2 causal attention using the exact sigmoid-of-score-difference identity.

    Query 0 has public causal probabilities ``[1, 0]``.  Query 1 uses
    ``p(source=1) = sigmoid(score_11 - score_10)`` and ``p(source=0)=1-p``.
    This removes the unbounded ``1 + exp(delta)`` denominator from the
    encrypted circuit while retaining exact model semantics before polynomial
    approximation.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (2, 2):
        raise ValueError(f"expected scores [heads,2,2], got {array.shape}")
    delta = array[:, 1, 1] - array[:, 1, 0]
    probability_one, check = evaluate_public_polynomial(
        "sigmoid",
        delta,
        delta_domain,
        degree,
        name=name,
    )
    probabilities = np.zeros_like(array)
    probabilities[:, 0, 0] = 1.0
    probabilities[:, 1, 0] = 1.0 - probability_one
    probabilities[:, 1, 1] = probability_one
    check.update(
        {
            "schedule": "p1=sigmoid(score_11-score_10); p0=1-p1",
            "probability_range": finite_range(probabilities),
            "unbounded_exp_denominator_eliminated": True,
        }
    )
    return probabilities, check


def dual_rel_inf(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Global and worst-leading-axis relative-infinity acceptance gate."""
    expected = np.asarray(reference, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if expected.shape != observed.shape:
        raise ValueError(
            f"shape mismatch: reference={expected.shape}, actual={observed.shape}"
        )
    all_finite = bool(np.isfinite(expected).all() and np.isfinite(observed).all())
    if not all_finite:
        return {
            "global_rel_inf": float("inf"),
            "worst_token_rel_inf": float("inf"),
            "max_abs_error": float("inf"),
            "tolerance": tolerance,
            "all_finite": False,
            "passed": False,
        }
    difference = np.abs(observed - expected)
    global_denominator = max(float(np.max(np.abs(expected))), 1e-12)
    global_rel = float(np.max(difference) / global_denominator)
    if expected.ndim <= 1:
        worst_token = global_rel
    else:
        rows = expected.shape[0]
        worst_token = max(
            float(np.max(difference[index]))
            / max(float(np.max(np.abs(expected[index]))), 1e-12)
            for index in range(rows)
        )
    return {
        "global_rel_inf": global_rel,
        "worst_token_rel_inf": worst_token,
        "max_abs_error": float(np.max(difference)),
        "tolerance": tolerance,
        "all_finite": True,
        "passed": global_rel <= tolerance and worst_token <= tolerance,
    }
