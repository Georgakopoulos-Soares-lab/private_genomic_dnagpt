"""Validate the immutable two-block fixture and range-control contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEFAULT_FIXTURE = (
    REPO
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0"
)
DEFAULT_RANGE_CONTROL = (
    REPO
    / "fhe"
    / "range_control"
    / "results"
    / "range_control_public_fixture_optimized_v2_PASS.json"
)
MANIFEST_SHA256 = "3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c"
RANGE_CONTROL_SHA256 = (
    "b148e30b42c430405a0a2c41401ab295701e0e1c3655ef00f06984ff9f8faba2"
)
EXPECTED_DEGREES = (
    {
        "ln1_inverse_sqrt": 7,
        "sigmoid": 9,
        "ln2_inverse_sqrt": 7,
        "gelu": 15,
    },
    {
        "ln1_inverse_sqrt": 7,
        "sigmoid": 11,
        "ln2_inverse_sqrt": 7,
        "gelu": 39,
    },
)
EXPECTED_DOMAINS = (
    {
        "block0.ln1_variance_plus_eps": [0.0030030699438575933, 0.010106134790696561],
        "block0.attention_delta_s1_minus_s0": [-6.069866743337161, 0.25],
        "block0.ln2_variance_plus_eps": [0.504245970267154, 1.289585701152818],
        "block0.gelu_input": [-3.46661639687445, 3.46661639687445],
    },
    {
        "block1.ln1_variance_plus_eps": [0.7231809701265808, 1.5063129576985312],
        "block1.attention_delta_s1_minus_s0": [-8.981407641021296, 0.25],
        "block1.ln2_variance_plus_eps": [0.7487770832251114, 1.3440084965164667],
        "block1.gelu_input": [-9.905612243716092, 9.905612243716092],
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_contract() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (ROOT / "fixture.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        entries[filename] = digest
    return entries


def validate(fixture: Path, range_control_path: Path) -> dict[str, object]:
    if sha256(fixture / "manifest.json") != MANIFEST_SHA256:
        raise ValueError("refusing unpinned multiblock manifest")
    if sha256(range_control_path) != RANGE_CONTROL_SHA256:
        raise ValueError("refusing unpinned optimized range-control result")

    for filename, expected in fixture_contract().items():
        if sha256(fixture / filename) != expected:
            raise ValueError(f"fixture SHA-256 mismatch: {filename}")

    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "dnagpt.multiblock.gsr_t2.v1":
        raise ValueError("unexpected fixture schema")
    if manifest["model"]["layers"] != 12:
        raise ValueError("fixture is not bound to the released 12-block model")
    if manifest["tokenization"]["exported_token_count_T"] != 2:
        raise ValueError("fixture is not T=2")
    if not manifest["oracle_gate"]["passed"]:
        raise ValueError("fixture oracle gate failed")

    schedule = json.loads(range_control_path.read_text(encoding="utf-8"))
    if schedule["status"] != "PASS" or not schedule["not_fhe_measurement"]:
        raise ValueError("range-control result must be a passing plaintext contract")
    for block in (0, 1):
        if (
            schedule["polynomials"]["per_block"][block]["degrees"]
            != EXPECTED_DEGREES[block]
        ):
            raise ValueError(f"block {block} polynomial degree contract changed")
        checks = {
            item["name"]: item["public_domain"]
            for item in schedule["blocks"][block]["domain_checks"]
        }
        if checks != EXPECTED_DOMAINS[block]:
            raise ValueError(f"block {block} public domain contract changed")
        if not schedule["blocks"][block]["passed"]:
            raise ValueError(f"block {block} plaintext schedule did not pass")

    return {
        "scope": "static/plaintext contract validation; not FHE evidence",
        "fixture_manifest_sha256": MANIFEST_SHA256,
        "range_control_sha256": RANGE_CONTROL_SHA256,
        "consumed_fixture_files": len(fixture_contract()),
        "blocks": [0, 1],
        "degrees": list(EXPECTED_DEGREES),
        "plaintext_block1_gate": schedule["blocks"][1]["output_gate"],
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="?", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--range-control", type=Path, default=DEFAULT_RANGE_CONTROL)
    args = parser.parse_args()
    print(
        json.dumps(
            validate(args.fixture.resolve(), args.range_control.resolve()),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
