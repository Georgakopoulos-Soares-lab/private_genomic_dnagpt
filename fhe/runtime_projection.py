"""Produce an assumption-labeled runtime boundary for the 0.1b GPU path.

This is not a benchmark. It combines immutable project work counts with published
GPU CKKS primitive measurements to reject the unoptimized per-token schedule and
to record the explicit planning hypothesis that the real GPU gates must replace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60

# FIDESlib paper, Table V, RTX 4090, [N, L, delta, dnum]=[2^16, 29, 59, 4].
FIDES_ROTATION_S = 1.107e-3
FIDES_CT_MULT_S = 1.084e-3
FIDES_PT_MULT_S = 21.74e-6


def _load(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(work_path, toy_path):
    work = _load(work_path)
    toy = _load(toy_path)
    counts = work["per_token_layout_work_counts"]

    toy_scale_seconds = (
        toy["evaluation_latency_s"] * counts["causal_pair_count_ratio_vs_one_toy_block"]
    )
    rotations = counts["linear_rotations_12_blocks"]
    ct_mults = counts["score_plus_value_ciphertext_products_12_blocks"]
    linear_evaluations = counts["linear_evaluations_12_blocks"]
    dimension = work["target"]["dimension"]
    pt_mults = linear_evaluations * dimension

    rotation_seconds = rotations * FIDES_ROTATION_S
    ct_mult_seconds = ct_mults * FIDES_CT_MULT_S
    pt_mult_seconds = pt_mults * FIDES_PT_MULT_S
    primitive_subtotal = rotation_seconds + ct_mult_seconds + pt_mult_seconds

    return {
        "schema_version": 1,
        "task": "DNAGPT 0.1b runtime boundary",
        "derived_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "[A/U] planning boundary, not an encrypted 0.1b benchmark",
        "target": work["target"],
        "inputs": {
            "work_count_result": work_path,
            "work_count_sha256": _sha256(work_path),
            "toy_block_result": toy_path,
            "toy_block_sha256": _sha256(toy_path),
        },
        "rejected_cpu_layout": {
            "toy_evaluation_seconds": toy["evaluation_latency_s"],
            "causal_pair_ratio": counts["causal_pair_count_ratio_vs_one_toy_block"],
            "toy_pair_scaled_seconds": toy_scale_seconds,
            "toy_pair_scaled_years": toy_scale_seconds / SECONDS_PER_YEAR,
            "qualification": (
                "[A] Multiplies only the measured toy evaluation by the causal-pair "
                "ratio. It ignores the D=8 to D=768 widening and is not a forecast. "
                "Its purpose is to reject repetition of the toy per-token CPU layout."
            ),
        },
        "unoptimized_gpu_primitive_substitution": {
            "published_platform": "FIDESlib RTX 4090",
            "published_parameter_set": "[N,L,delta,dnum]=[2^16,29,59,4]",
            "rotations": rotations,
            "rotation_seconds": rotation_seconds,
            "ciphertext_multiplications": ct_mults,
            "ciphertext_multiplication_seconds": ct_mult_seconds,
            "plaintext_multiplications": pt_mults,
            "plaintext_multiplication_seconds": pt_mult_seconds,
            "subtotal_seconds": primitive_subtotal,
            "subtotal_hours": primitive_subtotal / 3600,
            "excluded": [
                "LayerNorm, softmax, and GELU polynomial evaluation",
                "bootstrapping",
                "key generation and encoding",
                "host/device transfers",
                "memory-capacity stalls",
            ],
            "qualification": (
                "[A] Directly substitutes published single-primitive timings into the "
                "current unoptimized per-token work counts. Parameters and GPU differ "
                "from the planned A100. Packing, fusion, hoisting, and concurrency "
                "invalidate it as an end-to-end prediction; it is a schedule-rejection "
                "diagnostic."
            ),
        },
        "external_transformer_comparator": {
            "system": "EncryptedLLM, GPT-2 small",
            "shape": {
                "blocks": 12,
                "dimension": 768,
                "token_position": 128,
                "prior_input_processing": "amortized away",
            },
            "platform": "NVIDIA A100 80GB PCIe",
            "reported_result": (
                "over 200x GPU speedup, from several hours to a few minutes"
            ),
            "reported_128bit_bootstrap": (
                "roughly 550 ms to refresh 20 ciphertext levels"
            ),
            "qualification": (
                "[V, external] Relevant evidence that a packed GPU transformer can "
                "reach minutes. It is not a DNAGPT T=512 full-input benchmark and "
                "does not supply a transferable exact latency."
            ),
        },
        "working_expectation": {
            "first_unoptimized_gpu_run": "[A] order of 10+ hours",
            "optimized_gpu_resident_pass": "[A] roughly 0.5-4 hours",
            "not_expected": "seconds or interactive latency",
            "qualification": (
                "This is a deliberately broad engineering planning interval, not a "
                "confidence interval. Replace it after the D=768, T=8/16 one-block "
                "benchmark and again after T=512 packing measurements."
            ),
        },
        "primary_sources": [
            {
                "title": "FIDESlib paper",
                "url": "https://arxiv.org/abs/2507.04775",
                "used_for": "Table V primitive timings and GPU design",
            },
            {
                "title": "EncryptedLLM (ICML 2025)",
                "url": "https://proceedings.mlr.press/v267/de-castro25a.html",
                "used_for": "A100 GPT-2 comparator and bootstrap timing",
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-result",
        default="results/runs/fhe_0p1b_extrapolation_20260724.json",
    )
    parser.add_argument("--toy-result", default="results/runs/fhe_toy_block.json")
    parser.add_argument("--tag")
    args = parser.parse_args()
    result = project(args.work_result, args.toy_result)
    print(json.dumps(result, indent=2))
    if args.tag:
        output = Path("results/runs") / f"{args.tag}.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable result: {output}")
        os.makedirs(output.parent, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
