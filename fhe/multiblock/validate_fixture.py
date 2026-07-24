"""Validate a generated 12-block fixture without loading torch or a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fhe.realweights.contract import (
    CHECKPOINT_SHA256,
    POSITIVE_FASTA_SHA256,
    sha256_file,
)

from .export_fixture import SCHEMA, _canonical_content_identity, block_weight_keys


def validate_fixture(directory: Path, *, hash_arrays: bool = True) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA:
        raise ValueError(f"schema mismatch: {manifest['schema']!r}")
    if manifest["model"]["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("checkpoint identity mismatch")
    if manifest["dataset"]["sha256"] != POSITIVE_FASTA_SHA256:
        raise ValueError("positive FASTA identity mismatch")
    if manifest["model"]["layers"] != 12:
        raise ValueError("fixture must contain all 12 blocks")
    if manifest["tokenization"]["exported_token_count_T"] != 2:
        raise ValueError("fixture must use T=2")
    if not manifest["oracle_gate"]["passed"]:
        raise ValueError("plaintext oracle gate did not pass")

    arrays = manifest["arrays"]
    expected_weights = {f"weights.{name}" for name in block_weight_keys(12)}
    missing_weights = sorted(expected_weights - set(arrays))
    if missing_weights:
        raise ValueError(f"missing block weights: {missing_weights}")

    total_bytes = 0
    for name, item in arrays.items():
        relative = Path(item["file"])
        if relative.name != item["file"] or relative.is_absolute():
            raise ValueError(f"unsafe array filename for {name}: {item['file']!r}")
        path = directory / relative
        stat_bytes = path.stat().st_size
        if stat_bytes != int(item["bytes"]):
            raise ValueError(
                f"byte count mismatch for {name}: manifest={item['bytes']}, "
                f"file={stat_bytes}"
            )
        expected_bytes = int(item["elements"]) * np.dtype("<f8").itemsize
        if expected_bytes != stat_bytes:
            raise ValueError(f"shape/dtype byte contract mismatch for {name}")
        if int(np.prod(item["shape"])) != int(item["elements"]):
            raise ValueError(f"shape element count mismatch for {name}")
        if hash_arrays and sha256_file(path) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}")
        total_bytes += stat_bytes

    token_ids = [int(value) for value in manifest["tokenization"]["token_ids"]]
    actual_identity = _canonical_content_identity(arrays, token_ids)
    if actual_identity != manifest["content_identity_sha256"]:
        raise ValueError(
            "content identity mismatch: "
            f"manifest={manifest['content_identity_sha256']}, "
            f"actual={actual_identity}"
        )

    classification = manifest["classification"]["truncated_t2_graph_gate"]
    if (
        classification["margin_n_minus_a"] >= 0
        and classification["binary_label"] != "N"
    ):
        raise ValueError("classification sign/label mismatch")
    if classification["margin_n_minus_a"] < 0 and classification["binary_label"] != "A":
        raise ValueError("classification sign/label mismatch")
    if manifest["public_calibration"]["block0_gpu_fixed_domains_compose_all_12"]:
        raise ValueError(
            "unexpected claim: block-0 GPU fixed domains are known not to cover all blocks"
        )

    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "content_identity_sha256": actual_identity,
        "arrays": len(arrays),
        "bytes": total_bytes,
        "hash_arrays": hash_arrays,
        "record_header": manifest["dataset"]["record_header"],
        "token_ids": token_ids,
        "token_pieces": manifest["tokenization"]["token_pieces"],
        "margin_n_minus_a": classification["margin_n_minus_a"],
        "binary_label": classification["binary_label"],
        "full_prompt_margin_n_minus_a": manifest["classification"][
            "complete_public_prompt_reference"
        ]["margin_n_minus_a"],
        "full_prompt_label": manifest["classification"][
            "complete_public_prompt_reference"
        ]["binary_label"],
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="skip re-hashing every binary (structural checks still run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_fixture(
        args.fixture.resolve(), hash_arrays=not args.metadata_only
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
