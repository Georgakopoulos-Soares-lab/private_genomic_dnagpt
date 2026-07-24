"""Derive the first measured lower bound for a 12-block D768/T2 GPU pass.

This is deliberately not a full-model forecast. It repeats only the measured
block-0 LayerNorm/QKV/12-head-attention/projection evaluation time 12 times and
therefore excludes every MLP, bootstrap, later-block domain change, and task head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ATTENTION_RESULT = Path(
    "results/runs/fhe_fides_real_d768_t2_attention_a100_asymfix2_20260724.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(attention_path: Path) -> dict:
    attention = json.loads(attention_path.read_text())
    if not attention["passed"] or attention["gate"] != "attention":
        raise ValueError("measured input must be a passing attention gate")
    config = attention["config"]
    if (config["D"], config["T"], config["heads"]) != (768, 2, 12):
        raise ValueError("measured input shape is not D768/T2/H12")

    measured_seconds = float(attention["timings_seconds"]["encrypted_evaluation"])
    repeated_seconds = measured_seconds * 12
    return {
        "schema_version": 1,
        "task": "DNAGPT measured D768/T2 12-block attention-only runtime boundary",
        "derived_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "[A] derived lower boundary; not a 12-block encrypted run",
        "input": {
            "result": str(attention_path),
            "sha256": sha256(attention_path),
            "measured_gate": "block-0 LayerNorm + QKV + 12-head attention + projection",
            "encrypted_evaluation_seconds_gpu": measured_seconds,
            "environment": attention["environment"],
        },
        "target": {
            "layers": 12,
            "dimension": 768,
            "heads": 12,
            "sequence_length": 2,
        },
        "derived": {
            "attention_only_repeated_seconds": repeated_seconds,
            "attention_only_repeated_hours": repeated_seconds / 3600.0,
            "qualification": (
                "[A] Repeats the measured block-0 attention gate 12 times. "
                "It excludes all MLPs, bootstraps, the classifier head, key setup, "
                "and later-block polynomial-domain changes, so it is a lower "
                "boundary for this unoptimized T=2 schedule, not a forecast."
            ),
        },
        "full_task_boundary": {
            "task_representative_token_count": 103,
            "current_gpu_schedule_specialized_to_t2": True,
            "runtime": "[U] not extrapolated",
            "reason": (
                "A T=103 implementation needs a different packed softmax/layout; "
                "linear or quadratic scaling from the T=2 specialization would "
                "create false precision."
            ),
        },
        "next_replacements": [
            "direct T=2 sigmoid full-block measurement",
            "native block-boundary bootstrap measurement",
            "two-block same-lineage measurement",
            "packed T=8/16 attention measurement before longer sequences",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-result", type=Path, default=DEFAULT_ATTENTION_RESULT
    )
    parser.add_argument("--tag")
    args = parser.parse_args()
    result = derive(args.attention_result)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.tag:
        output = Path("results/runs") / f"{args.tag}.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite immutable result: {output}")
        os.makedirs(output.parent, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
