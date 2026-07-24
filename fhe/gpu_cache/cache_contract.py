#!/usr/bin/env python3
"""Seal and verify the public FIDESlib evaluation cache manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
from pathlib import Path

from rotation_contract import derive, sha256

PINNED_FIDES_COMMIT = "786c7600fb2f16b724e0acf73df367b27b8afed6"
PINNED_PATCHED_IMAGE_DIGEST = (
    "sha256:8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4"
)
ARTIFACT_NAMES = (
    "crypto-context.bin",
    "crypto-context.bin.dev",
    "public-key.bin",
    "eval-mult.bin",
    "eval-automorphism.bin",
)
MANIFEST_NAME = "manifest.json"
TIMING_NAMES = {
    "context_generation",
    "keygen",
    "eval_mult_keygen",
    "eval_automorphism_keygen",
    "public_cache_serialization",
    "client_oracle_key_serialization",
}


def _constant(source: str, name: str) -> int:
    match = re.search(
        rf"constexpr\s+(?:std::size_t|std::uint32_t)\s+{re.escape(name)}\s*=\s*"
        r"([0-9]+)\s*;",
        source,
    )
    if match:
        return int(match.group(1))
    if name == "SLOTS" and re.search(
        r"constexpr\s+std::size_t\s+SLOTS\s*=\s*COPIES\s*\*\s*PACK_WIDTH\s*;",
        source,
    ):
        return _constant(source, "COPIES") * _constant(source, "PACK_WIDTH")
    raise ValueError(f"missing real-gate constant {name}")


def parameters(real_source: Path) -> dict:
    source = real_source.read_text(encoding="utf-8")
    required_literals = (
        "parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);",
        "parameters.SetSecretKeyDist(UNIFORM_TERNARY);",
        "parameters.SetScalingTechnique(FLEXIBLEAUTO);",
        "parameters.SetKeySwitchTechnique(HYBRID);",
    )
    for literal in required_literals:
        if literal not in source:
            raise ValueError(f"real-gate parameter contract changed: {literal}")
    return {
        "security": "HEStd_128_classic",
        "secret_key_distribution": "UNIFORM_TERNARY",
        "multiplicative_depth": _constant(source, "MULT_DEPTH"),
        "scale_bits": _constant(source, "SCALE_BITS"),
        "first_mod_bits": _constant(source, "FIRST_MOD_BITS"),
        "large_digits": _constant(source, "LARGE_DIGITS"),
        "batch_slots": _constant(source, "SLOTS"),
        "pack_width": _constant(source, "PACK_WIDTH"),
        "scaling_technique": "FLEXIBLEAUTO",
        "key_switch_technique": "HYBRID",
        "plaintext_autoload": False,
        "ciphertext_autoload": True,
    }


def _require_digest_image(image: str) -> None:
    if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("container image must be bound to an exact sha256 digest")
    if not image.endswith("@" + PINNED_PATCHED_IMAGE_DIGEST):
        raise ValueError(
            "refusing image other than the measured asymmetric-Chebyshev patch"
        )


def _regular_file(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"cache artifact must be a regular non-symlink file: {path}")
    if metadata.st_size <= 0:
        raise ValueError(f"cache artifact is empty: {path}")


def _check_directory_entries(cache_dir: Path, include_manifest: bool) -> None:
    allowed = set(ARTIFACT_NAMES)
    if include_manifest:
        allowed.add(MANIFEST_NAME)
    actual = {path.name for path in cache_dir.iterdir()}
    if actual != allowed:
        raise ValueError(
            f"cache contains unexpected/missing entries: "
            f"expected={sorted(allowed)} actual={sorted(actual)}"
        )
    forbidden = re.compile(
        r"(private|secret|oracle|client|decrypt|sk(?:[._-]|$))", re.IGNORECASE
    )
    rejected = [name for name in actual if forbidden.search(name)]
    if rejected:
        raise ValueError(f"private-key-like cache entries rejected: {rejected}")


def _parse_context_sidecar(path: Path) -> list[int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rotation_line = next(
        (line for line in lines if line.startswith("RotationIndexes:")), None
    )
    if rotation_line is None:
        raise ValueError("context sidecar has no RotationIndexes line")
    match = re.fullmatch(r"RotationIndexes:\s*\{\s*(.*?)\s*\}\s*", rotation_line)
    if not match:
        raise ValueError("malformed RotationIndexes line")
    return [int(value) for value in match.group(1).split()]


def artifact_records(cache_dir: Path) -> dict:
    records = {}
    for name in ARTIFACT_NAMES:
        path = cache_dir / name
        _regular_file(path)
        records[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return records


def validate_timings(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != TIMING_NAMES
        or any(
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
            for seconds in value.values()
        )
    ):
        raise ValueError("provision timing schema mismatch")
    return value


def expected_contract(
    real_source: Path, module_source: Path, image: str
) -> tuple[dict, dict]:
    _require_digest_image(image)
    rotation = derive(real_source)
    manifest_rotation = {
        key: rotation[key]
        for key in (
            "gate",
            "count",
            "steps",
            "pack_width",
            "slots",
            "bsgs_n1",
            "bsgs_n2",
            "real_source_sha256",
            "derivation",
            "evidence_note",
        )
    }
    contract = {
        "backend": {
            "name": "FIDESlib CKKS/CUDA",
            "commit": PINNED_FIDES_COMMIT,
            "container_image": image,
        },
        "parameters": parameters(real_source),
        "rotation_contract": manifest_rotation,
        "sources": {
            "cache_gate_cpp_sha256": sha256(module_source),
            "generated_rotation_header_sha256": sha256(
                module_source.parent.parent / "include/generated_rotation_contract.hpp"
            ),
            "real_gate_cpp_sha256": sha256(real_source),
            "rotation_contract_py_sha256": sha256(
                Path(__file__).resolve().with_name("rotation_contract.py")
            ),
            "cache_contract_py_sha256": sha256(Path(__file__).resolve()),
        },
    }
    return contract, rotation


def seal(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir.resolve()
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        raise ValueError("cache directory must be an existing non-symlink directory")
    manifest_path = cache_dir / MANIFEST_NAME
    if manifest_path.exists():
        raise ValueError(f"refusing to overwrite immutable manifest: {manifest_path}")
    _check_directory_entries(cache_dir, include_manifest=False)

    contract, rotation = expected_contract(
        args.real_source.resolve(), args.module_source.resolve(), args.image
    )
    sidecar_rotations = _parse_context_sidecar(cache_dir / "crypto-context.bin.dev")
    if sidecar_rotations != rotation["steps"]:
        raise ValueError(
            "serialized FIDESlib context rotation vector differs from real gate"
        )
    metadata = json.loads(args.provision_metadata.read_text(encoding="utf-8"))
    if metadata.get("rotation_count") != rotation["count"]:
        raise ValueError("provision metadata rotation count mismatch")
    if metadata.get("private_key_cached") is not False:
        raise ValueError("provision metadata does not attest private_key_cached=false")
    provision_timings = validate_timings(metadata.get("timings_seconds"))

    manifest = {
        "schema_version": 1,
        "task": "DNAGPT FIDESlib public evaluation-key cache",
        "cache_scope": (
            "context + public key + eval-mult + eval-automorphism only; "
            "never a private key"
        ),
        **contract,
        "artifacts": artifact_records(cache_dir),
        "provision_timings_seconds": provision_timings,
        "private_key_cached": False,
        "load_order": (
            "crypto context -> public key -> eval-mult -> eval-automorphism "
            "-> GPU LoadContext"
        ),
    }
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "cache_manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "rotation_count": rotation["count"],
                "private_key_cached": False,
            },
            sort_keys=True,
        )
    )


def verify(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir.resolve()
    manifest_path = cache_dir / MANIFEST_NAME
    if not cache_dir.is_dir() or cache_dir.is_symlink():
        raise ValueError("cache directory must be an existing non-symlink directory")
    _check_directory_entries(cache_dir, include_manifest=True)
    _regular_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_top_level = {
        "schema_version",
        "task",
        "cache_scope",
        "backend",
        "parameters",
        "rotation_contract",
        "sources",
        "artifacts",
        "provision_timings_seconds",
        "private_key_cached",
        "load_order",
    }
    if set(manifest) != expected_top_level:
        raise ValueError("manifest top-level schema mismatch")
    fixed_fields = {
        "schema_version": 1,
        "task": "DNAGPT FIDESlib public evaluation-key cache",
        "cache_scope": (
            "context + public key + eval-mult + eval-automorphism only; "
            "never a private key"
        ),
        "private_key_cached": False,
        "load_order": (
            "crypto context -> public key -> eval-mult -> eval-automorphism "
            "-> GPU LoadContext"
        ),
    }
    for name, expected in fixed_fields.items():
        if manifest.get(name) != expected:
            raise ValueError(f"manifest fixed field mismatch: {name}")
    validate_timings(manifest.get("provision_timings_seconds"))

    contract, rotation = expected_contract(
        args.real_source.resolve(), args.module_source.resolve(), args.image
    )
    for name, expected in contract.items():
        if manifest.get(name) != expected:
            raise ValueError(f"manifest {name} contract mismatch")
    if manifest.get("private_key_cached") is not False:
        raise ValueError("manifest does not attest private_key_cached=false")
    if manifest.get("artifacts") != artifact_records(cache_dir):
        raise ValueError("cache artifact hash/size mismatch")
    if (
        _parse_context_sidecar(cache_dir / "crypto-context.bin.dev")
        != rotation["steps"]
    ):
        raise ValueError("context sidecar rotation vector mismatch")

    digest = sha256(manifest_path)
    if args.print_sha_only:
        print(digest)
    else:
        print(
            json.dumps(
                {
                    "verified": True,
                    "manifest_sha256": digest,
                    "rotation_count": rotation["count"],
                    "private_key_cached": False,
                },
                sort_keys=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "verify"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--cache-dir", required=True, type=Path)
        subparser.add_argument("--real-source", required=True, type=Path)
        subparser.add_argument("--module-source", required=True, type=Path)
        subparser.add_argument("--image", required=True)
    seal_parser = subparsers.choices["seal"]
    seal_parser.add_argument("--provision-metadata", required=True, type=Path)
    verify_parser = subparsers.choices["verify"]
    verify_parser.add_argument("--print-sha-only", action="store_true")
    args = parser.parse_args()

    if args.command == "seal":
        seal(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
