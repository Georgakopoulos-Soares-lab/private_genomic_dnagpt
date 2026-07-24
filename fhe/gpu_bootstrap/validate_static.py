#!/usr/bin/env python3
"""Fail-closed local policy checks for the encrypted refresh source."""

from __future__ import annotations

import re
from pathlib import Path


SOURCE = Path(__file__).parent / "src" / "dnagpt_fides_refresh.cpp"


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    module = SOURCE.parents[1]
    launcher = (module / "launch_brev_refresh.sh").read_text(encoding="utf-8")
    scheduler = (module / "schedule_refresh_when_free.sh").read_text(encoding="utf-8")
    decrypt_calls = re.findall(r"\bDecrypt\s*\(", text)
    if len(decrypt_calls) != 1:
        raise SystemExit(
            f"expected exactly one Decrypt call; found {len(decrypt_calls)}"
        )

    start = text.index("class EncryptedRefreshEvaluator")
    end = text.index("struct Metrics", start)
    evaluator = text[start:end]
    forbidden = ("Decrypt(", "PrivateKey", "secretKey")
    found = [item for item in forbidden if item in evaluator]
    if found:
        raise SystemExit(f"evaluator contains private/readout operations: {found}")

    required = (
        "EvalBootstrap",
        "EvalSquare",
        "HEStd_128_classic",
        "intermediate_decrypt_attempts",
        "final_decrypt_calls",
        "evaluator_has_private_key",
        "O_EXCL",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise SystemExit(f"missing fail-closed evidence markers: {missing}")
    for name, script in (("launcher", launcher), ("scheduler", scheduler)):
        if "--query-compute-apps=gpu_uuid" not in script:
            raise SystemExit(f"{name} does not reject resident compute processes")
    if "COMPUTE_PROCESS_COUNT != 0" not in launcher:
        raise SystemExit("launcher process-occupancy check is not fail-closed")
    if "REQUIRED_STABLE_POLLS=2" not in scheduler:
        raise SystemExit("scheduler does not require a stable idle window")
    if "/tmp/dnagpt-fhe-gpu-${gpu}.lock" not in scheduler:
        raise SystemExit("scheduler lacks cross-gate GPU reservation")
    if "trap cleanup_lock EXIT" not in scheduler:
        raise SystemExit("scheduler does not clean its GPU reservation")

    print(
        "STATIC_REFRESH_POLICY_PASS "
        "decrypt_calls=1 evaluator_private_key=false immutable_output=true"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
