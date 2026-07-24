"""Simulate a fixed public approximation schedule through all 12 DNAGPT blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .schedule import (
    ATTENTION_ZERO_GUARD,
    BLOCK_DEGREES,
    DEFAULT_TOLERANCE,
    DomainViolation,
    HEAD_DEGREES,
    ISOLATED_APPROXIMATION_TARGET,
    POLYNOMIAL_DEGREE_CAP,
    attention_delta_public_domain,
    dual_rel_inf,
    evaluate_public_polynomial,
    finite_range,
    polynomial_depth_proxy,
    public_scaled_inverse_sqrt,
    sigmoid_stable,
    t2_attention_probabilities,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = (
    REPO_ROOT
    / "checkpoints/fhe_exports"
    / "gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0"
)
DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "results"
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
SCHEMA = "dnagpt.range_control.plaintext.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Fixture:
    """Manifest-directed, shape-checked loader for immutable float64 arrays."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest["schema"] != "dnagpt.multiblock.gsr_t2.v1":
            raise ValueError(f"unsupported fixture schema: {self.manifest['schema']}")
        if self.manifest["model"]["layers"] != 12:
            raise ValueError("range-control simulation requires all 12 blocks")
        if self.manifest["tokenization"]["exported_token_count_T"] != 2:
            raise ValueError("range-control simulation is specialized to T=2")
        if not self.manifest["oracle_gate"]["passed"]:
            raise ValueError("fixture plaintext oracle did not pass")

    def array(self, name: str) -> np.ndarray:
        item = self.manifest["arrays"][name]
        filename = Path(item["file"])
        if filename.is_absolute() or filename.name != item["file"]:
            raise ValueError(f"unsafe fixture array path: {item['file']!r}")
        value = np.fromfile(self.directory / filename, dtype=np.dtype("<f8"))
        expected_elements = int(item["elements"])
        if value.size != expected_elements:
            raise ValueError(
                f"{name}: expected {expected_elements} elements, read {value.size}"
            )
        return value.reshape(item["shape"])

    def block_weights(self, block: int) -> dict[str, np.ndarray]:
        return {
            name: self.array(f"weights.block{block}.{name}")
            for name in (
                "ln1",
                "attn_qkv",
                "attn_proj",
                "ln2",
                "mlp_fc",
                "mlp_proj",
            )
        }


def _layer_norm_approx(
    value: np.ndarray,
    weight: np.ndarray,
    domain: list[float],
    degree: int,
    *,
    name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = np.mean(value, axis=-1, keepdims=True, dtype=np.float64)
    centered = value - mean
    variance_plus_eps = (
        np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float64) + 1e-5
    )
    inverse, report = public_scaled_inverse_sqrt(
        variance_plus_eps,
        domain,
        degree,
        name=name,
    )
    return centered * inverse * weight, report


def _head_gate(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    return dual_rel_inf(reference, actual, tolerance=DEFAULT_TOLERANCE)


def _isolated_ablation_summary(
    fixture: Fixture,
    blocks: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = fixture.manifest
    calibration = manifest["public_calibration"]
    legacy_domains = calibration["block0_gpu_fixed_domains"]
    exact_domain_failures = calibration["block0_gpu_fixed_domain_failures"]

    attention_identity_error = 0.0
    sigmoid_polynomial_error = 0.0
    inverse_sqrt_relative_error = 0.0
    gelu_polynomial_error = 0.0
    gelu_per_block_errors: list[dict[str, Any]] = []
    gelu_legacy_violations: list[int] = []
    proposed_gelu_violations: list[int] = []

    for block in range(12):
        public = calibration["per_block"][block]["recommended_public_domains"]
        scores = fixture.array(f"oracle.block{block}.attention_scores")
        exact_probabilities = fixture.array(
            f"oracle.block{block}.attention_probabilities"
        )
        delta = scores[:, 1, 1] - scores[:, 1, 0]
        exact_p1 = sigmoid_stable(delta)
        attention_identity_error = max(
            attention_identity_error,
            float(np.max(np.abs(exact_p1 - exact_probabilities[:, 1, 1]))),
        )
        approximate_probabilities, _ = t2_attention_probabilities(
            scores,
            attention_delta_public_domain(public["attention_delta_s1_minus_s0"]),
            BLOCK_DEGREES[block]["sigmoid"],
            name=f"ablation.block{block}.attention_delta",
        )
        sigmoid_polynomial_error = max(
            sigmoid_polynomial_error,
            float(
                np.max(
                    np.abs(
                        approximate_probabilities[:, 1, 1]
                        - exact_probabilities[:, 1, 1]
                    )
                )
            ),
        )

        for layer_norm in ("ln1", "ln2"):
            variance = (
                fixture.array(f"oracle.block{block}.{layer_norm}_variance") + 1e-5
            )
            expected = fixture.array(f"oracle.block{block}.{layer_norm}_inv_std")
            actual, _ = public_scaled_inverse_sqrt(
                variance,
                public[f"{layer_norm}_variance_plus_eps"],
                BLOCK_DEGREES[block][f"{layer_norm}_inverse_sqrt"],
                name=f"ablation.block{block}.{layer_norm}",
            )
            relative = np.max(
                np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-12)
            )
            inverse_sqrt_relative_error = max(
                inverse_sqrt_relative_error, float(relative)
            )

        gelu_input = fixture.array(f"oracle.block{block}.mlp_fc")
        expected_gelu = fixture.array(f"oracle.block{block}.gelu_tanh")
        legacy = legacy_domains["gelu_input"]
        if (
            float(np.min(gelu_input)) < legacy[0]
            or float(np.max(gelu_input)) > legacy[1]
        ):
            gelu_legacy_violations.append(block)
        proposed = public["gelu_input"]
        if (
            float(np.min(gelu_input)) < proposed[0]
            or float(np.max(gelu_input)) > proposed[1]
        ):
            proposed_gelu_violations.append(block)
        actual_gelu, _ = evaluate_public_polynomial(
            "gelu",
            gelu_input,
            proposed,
            BLOCK_DEGREES[block]["gelu"],
            name=f"ablation.block{block}.gelu",
        )
        gelu_error = float(np.max(np.abs(actual_gelu - expected_gelu)))
        gelu_polynomial_error = max(gelu_polynomial_error, gelu_error)
        gelu_per_block_errors.append(
            {
                "block": block,
                "degree": BLOCK_DEGREES[block]["gelu"],
                "isolated_max_abs_error": gelu_error,
            }
        )

    old_denominator = calibration["global_ranges"]["attention_denominator"]
    return {
        "A_fixed_public_domains": {
            "status": "[A]",
            "source": "fixture manifest public_calibration.per_block",
            "selected_before_simulated_query": True,
            "private_query_adaptation": False,
            "margin_multiplier": calibration["margin_multiplier"],
            "legacy_block0_domain_failure_count": len(exact_domain_failures),
            "legacy_block0_domain_failures": exact_domain_failures,
            "proposed_schedule_domain_violations": sum(
                len(item["domain_violations"]) for item in blocks
            ),
        },
        "B_t2_sigmoid_attention": {
            "identity": "softmax([s0,s1])[1] = sigmoid(s1-s0)",
            "exact_identity_max_abs_error": attention_identity_error,
            "per_block_sigmoid_degrees": [item["sigmoid"] for item in BLOCK_DEGREES],
            "sigmoid_polynomial_max_abs_error": sigmoid_polynomial_error,
            "old_exp_denominator_observed_range": old_denominator,
            "new_probability_codomain": [0.0, 1.0],
            "unbounded_exp_denominator_eliminated": True,
        },
        "C_public_scaled_layernorm": {
            "identity": (
                "inv_sqrt(v) = inv_sqrt(v/public_power_of_two_scale)"
                " / sqrt(public_power_of_two_scale)"
            ),
            "per_block_inverse_sqrt_degrees": [
                {
                    "ln1": item["ln1_inverse_sqrt"],
                    "ln2": item["ln2_inverse_sqrt"],
                }
                for item in BLOCK_DEGREES
            ],
            "worst_isolated_relative_error": inverse_sqrt_relative_error,
            "scale_is_public_and_per_block": True,
        },
        "D_tail_aware_full_domain_gelu": {
            "semantics": "released tanh-GELU",
            "strategy": (
                "one full-domain Chebyshev polynomial; tails included, "
                "no clipping and no encrypted comparison"
            ),
            "per_block_degrees": [item["gelu"] for item in BLOCK_DEGREES],
            "maximum_degree": max(item["gelu"] for item in BLOCK_DEGREES),
            "legacy_minus3_plus3_violating_blocks": gelu_legacy_violations,
            "proposed_domain_violating_blocks": proposed_gelu_violations,
            "worst_isolated_max_abs_error": gelu_polynomial_error,
            "degree_cap_exceptions": [
                item
                for item in gelu_per_block_errors
                if item["degree"] == POLYNOMIAL_DEGREE_CAP
                and item["isolated_max_abs_error"] > ISOLATED_APPROXIMATION_TARGET
            ],
            "fixture_observed_global_input_range": calibration["global_ranges"][
                "gelu_input"
            ],
        },
    }


def run_simulation(fixture_dir: Path = DEFAULT_FIXTURE) -> dict[str, Any]:
    started = time.perf_counter()
    fixture = Fixture(fixture_dir)
    manifest = fixture.manifest
    calibration = manifest["public_calibration"]
    value = fixture.array("input.embeddings")
    block_reports: list[dict[str, Any]] = []
    all_domain_violations: list[dict[str, Any]] = []

    try:
        for block in range(12):
            public = calibration["per_block"][block]["recommended_public_domains"]
            degrees = BLOCK_DEGREES[block]
            weights = fixture.block_weights(block)
            checks: list[dict[str, Any]] = []

            normalized1, check = _layer_norm_approx(
                value,
                weights["ln1"],
                public["ln1_variance_plus_eps"],
                degrees["ln1_inverse_sqrt"],
                name=f"block{block}.ln1_variance_plus_eps",
            )
            checks.append(check)
            qkv = normalized1 @ weights["attn_qkv"].T
            query_flat, key_flat, value_flat = np.split(qkv, 3, axis=-1)
            query = query_flat.reshape(2, 12, 64).transpose(1, 0, 2)
            key = key_flat.reshape(2, 12, 64).transpose(1, 0, 2)
            attention_value = value_flat.reshape(2, 12, 64).transpose(1, 0, 2)
            scores = (query @ key.transpose(0, 2, 1)) / math.sqrt(64)
            probabilities, check = t2_attention_probabilities(
                scores,
                attention_delta_public_domain(public["attention_delta_s1_minus_s0"]),
                degrees["sigmoid"],
                name=f"block{block}.attention_delta_s1_minus_s0",
            )
            checks.append(check)
            context_heads = probabilities @ attention_value
            context = context_heads.transpose(1, 0, 2).reshape(2, 768)
            residual = value + context @ weights["attn_proj"].T

            normalized2, check = _layer_norm_approx(
                residual,
                weights["ln2"],
                public["ln2_variance_plus_eps"],
                degrees["ln2_inverse_sqrt"],
                name=f"block{block}.ln2_variance_plus_eps",
            )
            checks.append(check)
            mlp_input = normalized2 @ weights["mlp_fc"].T
            gelu, check = evaluate_public_polynomial(
                "gelu",
                mlp_input,
                public["gelu_input"],
                degrees["gelu"],
                name=f"block{block}.gelu_input",
            )
            checks.append(check)
            value = residual + gelu @ weights["mlp_proj"].T

            gate = dual_rel_inf(
                fixture.array(f"oracle.block{block}.block_output"),
                value,
                tolerance=DEFAULT_TOLERANCE,
            )
            violations = [item for item in checks if not item["passed"]]
            all_domain_violations.extend(violations)
            block_reports.append(
                {
                    "block": block,
                    "degrees": dict(degrees),
                    "output_gate": gate,
                    "output_range": finite_range(value),
                    "domain_checks": checks,
                    "domain_violations": violations,
                    "passed": gate["passed"] and not violations,
                }
            )

        head_public = calibration["recommended_global_public_domains"]
        head_checks: list[dict[str, Any]] = []
        final_ln, check = _layer_norm_approx(
            value,
            fixture.array("weights.final_ln"),
            head_public["final_ln_variance_plus_eps"],
            HEAD_DEGREES["final_ln_inverse_sqrt"],
            name="head.final_ln_variance_plus_eps",
        )
        head_checks.append(check)
        final_ln_gate = _head_gate(
            fixture.array("oracle.head.final_ln_output"), final_ln
        )

        head_linear = final_ln @ fixture.array("weights.head_linear").T
        head_linear_gate = _head_gate(
            fixture.array("oracle.head.head_linear"), head_linear
        )
        head_silu, check = evaluate_public_polynomial(
            "silu",
            head_linear,
            head_public["head_silu_input"],
            HEAD_DEGREES["silu"],
            name="head.silu_input",
        )
        head_checks.append(check)
        head_silu_gate = _head_gate(fixture.array("oracle.head.head_silu"), head_silu)

        head_ln, check = _layer_norm_approx(
            head_silu,
            fixture.array("weights.head_ln"),
            head_public["head_ln_variance_plus_eps"],
            HEAD_DEGREES["head_ln_inverse_sqrt"],
            name="head.head_ln_variance_plus_eps",
        )
        head_checks.append(check)
        head_ln_gate = _head_gate(fixture.array("oracle.head.head_ln_output"), head_ln)
        exact_logits = fixture.array("oracle.head.logits")
        n_logit = float(head_ln[-1] @ fixture.array("weights.head_readout_n"))
        a_logit = float(head_ln[-1] @ fixture.array("weights.head_readout_a"))
        margin = float(
            head_ln[-1] @ fixture.array("weights.head_readout_margin_n_minus_a")
        )
        exact_classification = manifest["classification"]["truncated_t2_graph_gate"]
        readout_reference = np.array(
            [
                exact_logits[-1, exact_classification["n_token_id"]],
                exact_logits[-1, exact_classification["a_token_id"]],
                exact_classification["margin_n_minus_a"],
            ]
        )
        readout_actual = np.array([n_logit, a_logit, margin])
        readout_gate = dual_rel_inf(
            readout_reference,
            readout_actual,
            tolerance=DEFAULT_TOLERANCE,
        )
        head_violations = [item for item in head_checks if not item["passed"]]
        all_domain_violations.extend(head_violations)
        head_gates = {
            "final_ln_output": final_ln_gate,
            "head_linear": head_linear_gate,
            "head_silu": head_silu_gate,
            "head_ln_output": head_ln_gate,
            "n_a_margin_readout": readout_gate,
        }
        head_passed = all(item["passed"] for item in head_gates.values())
        head_report = {
            "stage_gates": head_gates,
            "domain_checks": head_checks,
            "domain_violations": head_violations,
            "readout": {
                "n_logit": n_logit,
                "a_logit": a_logit,
                "margin_n_minus_a": margin,
                "binary_label": "N" if margin >= 0.0 else "A",
                "reference_binary_label": exact_classification["binary_label"],
                "label_preserved": (
                    ("N" if margin >= 0.0 else "A")
                    == exact_classification["binary_label"]
                ),
            },
            "passed": head_passed and not head_violations,
        }
        ablations = _isolated_ablation_summary(fixture, block_reports)
        passed = (
            all(item["passed"] for item in block_reports)
            and head_report["passed"]
            and not all_domain_violations
        )
        failure = None
    except (DomainViolation, FloatingPointError) as exc:
        head_report = None
        ablations = None
        passed = False
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
        }

    block_polynomial_schedules = []
    for block, degrees in enumerate(BLOCK_DEGREES):
        depths = {
            name: polynomial_depth_proxy(degree) for name, degree in degrees.items()
        }
        block_polynomial_schedules.append(
            {
                "block": block,
                "degrees": dict(degrees),
                "balanced_multiplicative_depth_proxies": depths,
                "nonlinear_depth_proxy": sum(depths.values()),
            }
        )
    head_depths = {
        name: polynomial_depth_proxy(degree) for name, degree in HEAD_DEGREES.items()
    }
    return {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "evidence": (
            "[V] deterministic NumPy float64 approximation simulation; "
            "[A] domains calibrated from one fixed public T=2 fixture"
        ),
        "not_fhe_measurement": True,
        "fixture": {
            "directory": str(fixture.directory),
            "manifest_sha256": sha256_file(fixture.manifest_path),
            "content_identity_sha256": manifest["content_identity_sha256"],
            "checkpoint_sha256": manifest["model"]["checkpoint_sha256"],
            "dataset_sha256": manifest["dataset"]["sha256"],
            "token_ids": manifest["tokenization"]["token_ids"],
        },
        "contract": {
            "layers": 12,
            "token_count": 2,
            "embedding_dim": 768,
            "heads": 12,
            "same_released_weights_and_model_semantics": True,
            "acceptance": {
                "global_rel_inf_max": DEFAULT_TOLERANCE,
                "worst_token_rel_inf_max": DEFAULT_TOLERANCE,
                "all_finite": True,
                "zero_public_domain_violations": True,
            },
            "public_calibration": (
                "per-block fixed domains loaded from the immutable public "
                "fixture; never selected from an evaluated private query"
            ),
        },
        "polynomials": {
            "degree_selection": (
                "smallest public candidate meeting <=1e-4 isolated error, "
                "capped at degree 127, followed by the cumulative 4e-2 dual "
                "gate; two wide GELU stages hit the cap"
            ),
            "isolated_approximation_target": ISOLATED_APPROXIMATION_TARGET,
            "degree_cap": POLYNOMIAL_DEGREE_CAP,
            "attention_zero_guard": [
                -ATTENTION_ZERO_GUARD,
                ATTENTION_ZERO_GUARD,
            ],
            "per_block": block_polynomial_schedules,
            "head": {
                "degrees": dict(HEAD_DEGREES),
                "balanced_multiplicative_depth_proxies": head_depths,
                "nonlinear_depth_proxy": sum(head_depths.values()),
            },
            "warning": (
                "proxies exclude CKKS rescale, plaintext-multiply, rotation, "
                "relinearization, and bootstrap costs"
            ),
        },
        "blocks": block_reports,
        "head": head_report,
        "ablations": ablations,
        "domain_violation_count": len(all_domain_violations),
        "failure": failure,
        "elapsed_seconds": time.perf_counter() - started,
        "passed": passed,
    }


def write_immutable_result(
    result: dict[str, Any],
    tag: str,
    output_dir: Path = DEFAULT_RESULT_DIR,
) -> Path:
    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError(
            "tag must start with lowercase alphanumeric and contain only "
            "lowercase alphanumeric, dot, underscore, or dash"
        )
    status = result["status"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"range_control_{tag}_{status}.json"
    encoded = json.dumps(result, indent=2, sort_keys=False, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--tag",
        help=(
            "write one new immutable PASS/FAIL diagnostic; without this flag "
            "the result is printed and no file is created"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_simulation(args.fixture)
    if args.tag:
        path = write_immutable_result(result, args.tag, args.output_dir)
        result["written_result"] = str(path)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
