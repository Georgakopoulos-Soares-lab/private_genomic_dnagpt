#!/usr/bin/env python3
"""Fail-closed static audit for the public-cache security boundary."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rotation_contract import MEASURED_FULL_ROTATION_COUNT, cpp_text, derive


def between(text: str, start: str, end: str) -> str:
    first = text.find(start)
    last = text.find(end, first + len(start))
    if first < 0 or last < 0:
        raise ValueError(f"could not locate static boundary {start!r}..{end!r}")
    return text[first:last]


def require_in_order(text: str, tokens: tuple[str, ...]) -> None:
    cursor = 0
    for token in tokens:
        found = text.find(token, cursor)
        if found < 0:
            raise ValueError(f"missing/out-of-order load token: {token}")
        cursor = found + len(token)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    root = args.module_dir.resolve()
    cpp = (root / "src/dnagpt_fides_cache.cpp").read_text(encoding="utf-8")
    real_source = (root / "../gpu_real/src/real_dnagpt_fides.cpp").resolve()

    rotation = derive(real_source)
    if rotation["count"] != MEASURED_FULL_ROTATION_COUNT:
        raise ValueError("rotation contract count drift")
    committed_header = (root / "include/generated_rotation_contract.hpp").read_text(
        encoding="utf-8"
    )
    if committed_header != cpp_text(rotation):
        raise ValueError("committed generated rotation header drift")

    decrypt_calls = len(re.findall(r"\bcontext->Decrypt\s*\(", cpp))
    if decrypt_calls != 1:
        raise ValueError(f"expected exactly one Decrypt call, found {decrypt_calls}")

    evaluator = between(cpp, "class ReloadEvaluator", "struct Metrics")
    forbidden_evaluator_tokens = (
        "PrivateKey",
        "Decrypt(",
        "Deserialize",
        "Serialize",
        "oracle_key",
        "secretKey",
    )
    present = [token for token in forbidden_evaluator_tokens if token in evaluator]
    if present:
        raise ValueError(f"evaluator contains private/decrypt token(s): {present}")

    reload_body = between(cpp, "bool reload(", "}  // namespace")
    require_in_order(
        reload_body,
        (
            "Serial::DeserializeFromFile(\n            (options.cache_dir / CONTEXT_FILE)",
            "Serial::DeserializeFromFile(\n            (options.cache_dir / PUBLIC_KEY_FILE)",
            "context->DeserializeEvalMultKey",
            "context->DeserializeEvalAutomorphismKey",
            "context->LoadContext(public_key)",
            "evaluator.evaluate(encrypted_input)",
            "Serial::DeserializeFromFile(options.oracle_key.string()",
            "context->Decrypt(oracle_key, evaluation.packed, &decoded)",
        ),
    )

    if "path_is_within(options.oracle_key, options.cache_dir)" not in cpp:
        raise ValueError("missing fail-closed oracle-key-outside-cache check")
    if '"private_key_cached\\": false' not in cpp:
        raise ValueError("provision metadata lacks private_key_cached=false")

    provision = (root / "provision_cache.sh").read_text(encoding="utf-8")
    reload_script = (root / "reload_cache.sh").read_text(encoding="utf-8")
    parity = (root / "parity_gate.sh").read_text(encoding="utf-8")
    host_runner = (root / "run_brev_host.sh").read_text(encoding="utf-8")
    if "run_brev_host.sh" not in parity:
        raise ValueError("parity wrapper does not delegate to Brev host runner")
    for path, contents in (
        (root / "CMakeLists.txt", (root / "CMakeLists.txt").read_text()),
        (root / "build_in_fideslib.sh", (root / "build_in_fideslib.sh").read_text()),
        (root / "provision_cache.sh", provision),
        (root / "reload_cache.sh", reload_script),
    ):
        if re.search(r"\bpython(?:3)?\b|find_package\s*\(\s*Python", contents):
            raise ValueError(f"in-container path depends on Python: {path}")
    if 'cache_contract.py" seal' not in host_runner:
        raise ValueError("host runner does not seal project-owned manifest")
    if host_runner.count('cache_contract.py" verify') < 2:
        raise ValueError("host runner lacks pre/post cache verification")
    if host_runner.count("docker run --rm") < 3:
        raise ValueError("host runner lacks build/provision/reload containers")
    if "--query-compute-apps=gpu_uuid" not in host_runner:
        raise ValueError("host runner does not reject resident GPU compute processes")
    if "compute_process_count != 0" not in host_runner:
        raise ValueError("host runner process-occupancy check is not fail-closed")
    if (
        host_runner.count("--entrypoint /repo/fhe/gpu_cache/build/dnagpt_fides_cache")
        != 2
    ):
        raise ValueError("provision/reload are not separate C++ containers")
    if '-v "${CACHE_DIR}:/public-cache:ro"' not in host_runner:
        raise ValueError("reload public cache mount is not read-only")
    if '-v "${CLIENT_DIR}:/client:ro"' not in host_runner:
        raise ValueError("reload client material is not a separate read-only mount")
    for path, contents in (
        (root / "provision_cache.sh", provision),
        (root / "reload_cache.sh", reload_script),
        (root / "parity_gate.sh", parity),
        (root / "run_brev_host.sh", host_runner),
    ):
        if re.search(r"(?:^|[;&|]\s*|\n\s*)rm(?:\s|$)", contents):
            raise ValueError(f"destructive broad rm command forbidden: {path}")

    print(
        "STATIC_CACHE_GATE_PASS "
        f"rotation_keys={rotation['count']} decrypt_calls={decrypt_calls} "
        "evaluator_has_private_key=false"
    )


if __name__ == "__main__":
    main()
