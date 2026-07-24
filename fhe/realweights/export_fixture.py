"""Export a deterministic real-weight DNAGPT block-0 fixture for C++/CUDA.

The export contains no pickle or framework-specific tensor container. Each
array is a flat, little-endian C-order float32/float64 binary plus one JSON
manifest that records shape, dtype, byte count, and SHA-256.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .contract import (
    BLOCK0_EXPORT_KEYS,
    CHECKPOINT_SHA256,
    EXPECTED_MISSING_KEYS,
    MODEL_NAME,
    POSITIVE_FASTA_SHA256,
    REPO_ROOT,
    build_verified_model,
    expected_checkpoint_keys,
    gsr_prompt,
    repo_relative_or_absolute,
    sha256_file,
    sha256_text,
    verify_positive_fasta,
)
from .manual import block_reference, relative_inf

SCHEMA = "dnagpt.realweights.block0.v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/classification.pth"
DEFAULT_POSITIVE_FASTA = REPO_ROOT / "data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa"


def _export_dtype(name: str) -> np.dtype:
    if name == "float64":
        return np.dtype("<f8")
    if name == "float32":
        return np.dtype("<f4")
    raise ValueError(f"unsupported export dtype {name!r}")


def _safe_filename(name: str) -> str:
    return name.replace(".", "__").replace("/", "_") + ".bin"


def _array_metadata(path: Path, array: np.ndarray, role: str) -> dict[str, object]:
    return {
        "file": path.name,
        "role": role,
        "dtype": array.dtype.name,
        "endianness": "little",
        "order": "C",
        "shape": list(array.shape),
        "elements": int(array.size),
        "bytes": int(array.nbytes),
        "sha256": sha256_file(path),
    }


def _write_array(
    output_dir: Path,
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
    role: str,
) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if not np.isfinite(array).all():
        raise ValueError(f"refusing to export non-finite array {name}")
    path = output_dir / _safe_filename(name)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite array: {path}")
    with path.open("xb") as handle:
        array.tofile(handle)
    return _array_metadata(path, array, role)


def _default_output(token_count: int) -> Path:
    name = (
        f"gsr_pos0_block0_t{token_count}_"
        f"{CHECKPOINT_SHA256[:12]}_{POSITIVE_FASTA_SHA256[:12]}"
    )
    return REPO_ROOT / "checkpoints/fhe_exports" / name


def build_fixture(
    checkpoint: Path,
    positive_fasta: Path,
    token_count: int,
) -> tuple[OrderedDict[str, tuple[np.ndarray, str]], dict[str, object]]:
    """Build all arrays and manifest metadata without touching the output path."""
    if token_count <= 0:
        raise ValueError(f"T must be positive, got {token_count}")

    records = verify_positive_fasta(positive_fasta)
    header, sequence = records[0]
    prompt = gsr_prompt(sequence)
    model, tokenizer, state = build_verified_model(checkpoint)

    full_ids = tokenizer.encode(
        prompt, max_len=model.max_len, pad=False, device=torch.device("cpu")
    )
    if token_count > int(full_ids.numel()):
        raise ValueError(
            f"T={token_count} exceeds tokenized prompt length {full_ids.numel()}"
        )
    ids = full_ids[:token_count].unsqueeze(0)
    pieces = tokenizer.tokenize(prompt)[:token_count]

    with torch.inference_mode():
        token_embeddings = model.transformer.wte(ids)
        positions = torch.arange(token_count, dtype=torch.long).unsqueeze(0)
        position_embeddings = model.transformer.wpe(positions)
        upstream_embeddings = model._embedding_impl(ids)
        direct_embeddings = token_embeddings + position_embeddings
        if not torch.equal(upstream_embeddings, direct_embeddings):
            max_error = float(
                torch.max(torch.abs(upstream_embeddings - direct_embeddings))
            )
            raise AssertionError(
                f"upstream embedding reproduction failed: max_abs={max_error}"
            )
        upstream_output = model.transformer.h[0](upstream_embeddings)

    weights = {
        short_name: state[state_name].detach().cpu().numpy().astype(np.float64)
        for short_name, state_name in BLOCK0_EXPORT_KEYS.items()
    }
    manual = block_reference(
        upstream_embeddings[0].detach().cpu().numpy().astype(np.float64),
        weights,
        num_heads=model.num_heads,
        eps=model.transformer.h[0].ln_1.eps,
    )
    upstream_np = upstream_output[0].detach().cpu().numpy().astype(np.float64)
    max_abs_error = float(np.max(np.abs(manual["block_output"] - upstream_np)))
    rel_inf_error = relative_inf(upstream_np, manual["block_output"])
    if not np.allclose(manual["block_output"], upstream_np, rtol=2e-5, atol=2e-5):
        raise AssertionError(
            "manual NumPy block does not match upstream torch block: "
            f"max_abs={max_abs_error:.6g}, rel_inf={rel_inf_error:.6g}"
        )

    arrays: OrderedDict[str, tuple[np.ndarray, str]] = OrderedDict()
    for short_name, state_name in BLOCK0_EXPORT_KEYS.items():
        arrays[f"weights.{short_name}"] = (
            weights[short_name],
            f"checkpoint:{state_name}",
        )
    arrays["input.token_embeddings"] = (
        token_embeddings[0].detach().cpu().numpy(),
        "upstream tokenizer token lookup",
    )
    arrays["input.position_embeddings"] = (
        position_embeddings[0].detach().cpu().numpy(),
        "upstream position lookup",
    )
    arrays["input.embeddings"] = (
        upstream_embeddings[0].detach().cpu().numpy(),
        "token plus position embedding; encrypted model input",
    )
    for name, value in manual.items():
        arrays[f"oracle.{name}"] = (value, "independent NumPy float64 reference")
    arrays["oracle.upstream_torch_block_output"] = (
        upstream_np,
        "upstream torch float32 block output",
    )

    metadata: dict[str, object] = {
        "schema": SCHEMA,
        "model": {
            "name": MODEL_NAME,
            "layers": model.num_layers,
            "heads": model.num_heads,
            "embedding_dim": model.embedding_dim,
            "head_dim": model.embedding_dim // model.num_heads,
            "bias": False,
            "layernorm_eps": model.transformer.h[0].ln_1.eps,
            "checkpoint": repo_relative_or_absolute(checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_present_keys": list(expected_checkpoint_keys()),
            "checkpoint_missing_keys": list(EXPECTED_MISSING_KEYS),
            "checkpoint_unexpected_keys": [],
        },
        "dataset": {
            "path": repo_relative_or_absolute(positive_fasta),
            "sha256": POSITIVE_FASTA_SHA256,
            "record_index": 0,
            "record_header": header,
            "record_length_bp": len(sequence),
            "record_sequence_sha256": sha256_text(sequence),
            "transform": "remove sequence[300:306], asserted equal to AATAAA",
            "prompt_template": "<R>{sequence_without_central_AATAAA}<=><R>",
            "prompt_sha256": sha256_text(prompt),
        },
        "tokenization": {
            "implementation": "DNAGPT/dna_gpt/tokenizer.py:KmerTokenizer",
            "k": tokenizer.k,
            "dynamic_kmer": True,
            "vocab_size": len(tokenizer),
            "upstream_max_len": model.max_len,
            "full_prompt_token_count": int(full_ids.numel()),
            "exported_token_count_T": token_count,
            "token_ids": [int(value) for value in ids[0].tolist()],
            "token_pieces": pieces,
        },
        "oracle_gate": {
            "manual_reference": "NumPy float64, explicit causal attention",
            "upstream_reference": "torch float32 DNAGPT block 0",
            "rtol": 2e-5,
            "atol": 2e-5,
            "max_abs_error": max_abs_error,
            "rel_inf_error": rel_inf_error,
            "passed": True,
        },
    }
    return arrays, metadata


def export_fixture(
    checkpoint: Path,
    positive_fasta: Path,
    output_dir: Path,
    token_count: int,
    dtype_name: str,
) -> Path:
    """Build and write a new immutable fixture directory."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixture directory: {output_dir}")
    arrays, metadata = build_fixture(checkpoint, positive_fasta, token_count)
    dtype = _export_dtype(dtype_name)
    output_dir.mkdir(parents=True, exist_ok=False)

    array_manifest: OrderedDict[str, dict[str, object]] = OrderedDict()
    for name, (value, role) in arrays.items():
        array_manifest[name] = _write_array(output_dir, name, value, dtype, role)

    metadata["format"] = {
        "description": (
            "Headerless flat arrays. Read exactly `elements` IEEE-754 values "
            "using `dtype`, little endian, then reshape in C row-major order."
        ),
        "export_dtype": dtype.name,
        "endianness": "little",
        "order": "C",
    }
    metadata["arrays"] = array_manifest
    metadata["source"] = {
        "export_fixture_sha256": sha256_file(Path(__file__)),
        "manual_reference_sha256": sha256_file(Path(__file__).with_name("manual.py")),
        "contract_sha256": sha256_file(Path(__file__).with_name("contract.py")),
    }

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--positive-fasta", type=Path, default=DEFAULT_POSITIVE_FASTA)
    parser.add_argument(
        "--tokens",
        "-T",
        type=int,
        default=2,
        help="prefix token count to export (default: 2)",
    )
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32"),
        default="float64",
        help="binary array dtype (default: float64)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new output directory; default is checkpoints/fhe_exports/<identity>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    positive_fasta = args.positive_fasta.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else _default_output(args.tokens).resolve()
    )
    manifest = export_fixture(
        checkpoint, positive_fasta, output, args.tokens, args.dtype
    )
    with manifest.open(encoding="utf-8") as handle:
        result = json.load(handle)
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "arrays": len(result["arrays"]),
                "T": result["tokenization"]["exported_token_count_T"],
                "max_abs_error": result["oracle_gate"]["max_abs_error"],
                "rel_inf_error": result["oracle_gate"]["rel_inf_error"],
                "passed": result["oracle_gate"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
