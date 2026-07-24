"""Derive a transparent 0.1b work-count boundary from the toy measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone

LAYERS = 12
D = 768
HEADS = 12
SEQ_LEN = 512
LINEARS_PER_TOKEN_PER_BLOCK = 12  # Q/K/V + proj + 4 FC chunks + 4 FC-proj chunks


def bsgs_plan(n):
    n1 = math.isqrt(n)
    while n % n1:
        n1 += 1
    return n1, n // n1


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project(toy, bootstrap, toy_path, bootstrap_path):
    n1, n2 = bsgs_plan(D)
    rotations_per_linear = (n1 - 1) + (n2 - 1)
    causal_pairs_per_block = HEADS * SEQ_LEN * (SEQ_LEN + 1) // 2
    linear_evals_per_block = LINEARS_PER_TOKEN_PER_BLOCK * SEQ_LEN
    block_levels = max(toy["config"]["levels_consumed_before_pack"])
    critical_path_levels = LAYERS * block_levels
    initial_level_budget = bootstrap["level_before_bootstrap"]
    restored_levels = bootstrap["levels_restored"]
    critical_path_bootstrap_lower_bound = max(
        0,
        math.ceil(
            max(0, critical_path_levels - initial_level_budget) / restored_levels
        ),
    )

    ring_dim = toy["config"]["ring_dim"]
    max_slots = ring_dim // 2
    activation_slots = D * SEQ_LEN
    minimum_dense_activation_ciphertexts = math.ceil(activation_slots / max_slots)

    # A two-polynomial CKKS ciphertext with depth+1 64-bit RNS limbs. This excludes
    # object/serialization overhead and evaluation keys, and is therefore a lower bound.
    assumed_fresh_q_towers = toy["config"]["mult_depth"] + 1
    coefficient_bytes_per_fresh_ciphertext = 2 * ring_dim * assumed_fresh_q_towers * 8

    toy_pairs = (
        toy["config"]["heads"] * toy["config"]["T"] * (toy["config"]["T"] + 1) // 2
    )
    return {
        "schema_version": 1,
        "task": "DNAGPT 0.1b CKKS work-count extrapolation",
        "derived_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "[A/U] derived boundary, not an encrypted 0.1b run",
        "target": {
            "layers": LAYERS,
            "dimension": D,
            "heads": HEADS,
            "head_dimension": D // HEADS,
            "sequence_length": SEQ_LEN,
        },
        "measured_inputs": {
            "toy_block_result": toy_path,
            "toy_block_result_sha256": file_sha256(toy_path),
            "bootstrap_result": bootstrap_path,
            "bootstrap_result_sha256": file_sha256(bootstrap_path),
            "toy_block_levels_before_output_pack": block_levels,
            "bootstrap_levels_restored": restored_levels,
            "bootstrap_latency_emu_s": bootstrap["bootstrap_latency_emu_s"],
        },
        "per_token_layout_work_counts": {
            "bsgs_plan": {
                "baby_steps": n1,
                "giant_steps": n2,
                "rotations_per_DxD_linear": rotations_per_linear,
            },
            "linear_evaluations_per_block": linear_evals_per_block,
            "linear_evaluations_12_blocks": linear_evals_per_block * LAYERS,
            "linear_rotations_per_block": (
                linear_evals_per_block * rotations_per_linear
            ),
            "linear_rotations_12_blocks": (
                linear_evals_per_block * rotations_per_linear * LAYERS
            ),
            "layernorm_polynomials_12_blocks": 2 * SEQ_LEN * LAYERS,
            "softmax_rows_12_blocks": HEADS * SEQ_LEN * LAYERS,
            "gelu_polynomials_12_blocks": 4 * SEQ_LEN * LAYERS,
            "causal_query_key_pairs_per_block": causal_pairs_per_block,
            "causal_query_key_pairs_12_blocks": causal_pairs_per_block * LAYERS,
            "score_plus_value_ciphertext_products_12_blocks": (
                2 * causal_pairs_per_block * LAYERS
            ),
            "causal_pair_count_ratio_vs_one_toy_block": (
                causal_pairs_per_block * LAYERS / toy_pairs
            ),
        },
        "depth_boundary": {
            "assumed_sequential_levels_12_blocks": critical_path_levels,
            "optimistic_critical_path_bootstrap_lower_bound": (
                critical_path_bootstrap_lower_bound
            ),
            "bootstrap_only_emu_time_lower_bound_s": (
                critical_path_bootstrap_lower_bound
                * bootstrap["bootstrap_latency_emu_s"]
            ),
            "qualification": (
                "[A] Multiplies one measured toy-block depth by 12 and treats each "
                "standalone refresh as restoring the measured number of levels. A real "
                "attention schedule must refresh many parallel ciphertexts, so this is "
                "not a total bootstrap count or a latency forecast."
            ),
        },
        "ciphertext_volume_lower_bound": {
            "ring_dim": ring_dim,
            "maximum_ckks_slots": max_slots,
            "dense_activation_values": activation_slots,
            "minimum_ciphertexts_at_ideal_slot_occupancy": (
                minimum_dense_activation_ciphertexts
            ),
            "assumed_fresh_q_towers": assumed_fresh_q_towers,
            "coefficient_bytes_per_fresh_ciphertext": (
                coefficient_bytes_per_fresh_ciphertext
            ),
            "minimum_fresh_activation_coefficient_bytes": (
                minimum_dense_activation_ciphertexts
                * coefficient_bytes_per_fresh_ciphertext
            ),
            "qualification": (
                "[A] Coefficient-only lower bound. It excludes guard slots, layout "
                "constraints, object/serialization overhead, evaluation keys, and live "
                "intermediates."
            ),
        },
        "latency_projection": (
            "[U] No end-to-end 0.1b latency is projected: the toy uses a per-token "
            "correctness layout, while any full implementation must pack and parallelize "
            "tokens and heads. Multiplying [emu] toy latency would be misleading."
        ),
    }


def write_immutable(result, tag):
    path = os.path.join("results", "runs", f"{tag}.json")
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite immutable run: {path}")
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-result", default="results/runs/fhe_toy_block.json")
    parser.add_argument(
        "--bootstrap-result",
        default="results/runs/fhe_bootstrap_d8_20260724.json",
    )
    parser.add_argument("--tag")
    args = parser.parse_args()
    result = project(
        load_json(args.toy_result),
        load_json(args.bootstrap_result),
        args.toy_result,
        args.bootstrap_result,
    )
    print(json.dumps(result, indent=2))
    if args.tag:
        print(f"[evidence] {write_immutable(result, args.tag)}")


if __name__ == "__main__":
    main()
