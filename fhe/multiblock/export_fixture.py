"""Export a deterministic all-12-block DNAGPT T=2 plaintext/FHE fixture.

Large generated arrays live under the ignored ``checkpoints/fhe_exports`` tree.
The committed source contains only the reproducible exporter, validator, tests,
and documentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from fhe.realweights.contract import (
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
from fhe.realweights.manual import block_reference, relative_inf

from .manual import (
    BLOCK0_GPU_DOMAINS,
    block_calibration,
    classifier_head_reference,
    finite_range,
    outward_domain,
)

SCHEMA = "dnagpt.multiblock.gsr_t2.v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/classification.pth"
DEFAULT_POSITIVE_FASTA = REPO_ROOT / "data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa"
DEFAULT_MARGIN = 1.25

BLOCK_WEIGHT_SUFFIXES = OrderedDict(
    (
        ("ln1", "ln_1.weight"),
        ("attn_qkv", "attn.c_attn.weight"),
        ("attn_proj", "attn.c_proj.weight"),
        ("ln2", "ln_2.weight"),
        ("mlp_fc", "mlp.c_fc.weight"),
        ("mlp_proj", "mlp.c_proj.weight"),
    )
)


def block_weight_keys(layer_count: int = 12) -> OrderedDict[str, str]:
    result: OrderedDict[str, str] = OrderedDict()
    for layer in range(layer_count):
        for short_name, suffix in BLOCK_WEIGHT_SUFFIXES.items():
            result[f"block{layer}.{short_name}"] = f"transformer.h.{layer}.{suffix}"
    return result


def _default_output(token_count: int) -> Path:
    identity = (
        f"gsr_pos0_multiblock_t{token_count}_"
        f"{CHECKPOINT_SHA256[:12]}_{POSITIVE_FASTA_SHA256[:12]}"
    )
    return REPO_ROOT / "checkpoints/fhe_exports" / identity


def _safe_filename(name: str) -> str:
    return name.replace(".", "__").replace("/", "_") + ".bin"


def _write_array(
    directory: Path,
    name: str,
    value: np.ndarray,
    role: str,
    manifest: OrderedDict[str, dict[str, Any]],
) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.dtype("<f8")))
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"refusing to export empty/non-finite array {name}")
    path = directory / _safe_filename(name)
    with path.open("xb") as handle:
        array.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    manifest[name] = {
        "file": path.name,
        "role": role,
        "dtype": "float64",
        "endianness": "little",
        "order": "C",
        "shape": list(array.shape),
        "elements": int(array.size),
        "bytes": int(array.nbytes),
        "sha256": sha256_file(path),
    }


def _canonical_content_identity(
    arrays: OrderedDict[str, dict[str, Any]],
    token_ids: list[int],
) -> str:
    load_bearing = {
        "schema": SCHEMA,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "positive_fasta_sha256": POSITIVE_FASTA_SHA256,
        "token_ids": token_ids,
        "arrays": [
            {
                "name": name,
                "shape": item["shape"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for name, item in arrays.items()
        ],
    }
    encoded = json.dumps(load_bearing, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _block_weights(state: dict[str, torch.Tensor], layer: int) -> dict[str, np.ndarray]:
    return {
        short_name: state[f"transformer.h.{layer}.{suffix}"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        for short_name, suffix in BLOCK_WEIGHT_SUFFIXES.items()
    }


def _classification_summary(
    logits: np.ndarray,
    tokenizer: Any,
) -> dict[str, Any]:
    n_id = int(tokenizer.piece_to_id("N"))
    a_id = int(tokenizer.piece_to_id("A"))
    last = np.asarray(logits[-1], dtype=np.float64)
    argmax_id = int(np.argmax(last))
    n_logit = float(last[n_id])
    a_logit = float(last[a_id])
    margin = n_logit - a_logit
    return {
        "n_token_id": n_id,
        "a_token_id": a_id,
        "n_logit": n_logit,
        "a_logit": a_logit,
        "margin_n_minus_a": margin,
        "binary_label": "N" if margin >= 0.0 else "A",
        "binary_meaning": "real GSR" if margin >= 0.0 else "fake GSR",
        "global_argmax_token_id": argmax_id,
        "global_argmax_decoded": tokenizer.decode([argmax_id]),
        "global_argmax_is_binary_label": argmax_id in (n_id, a_id),
    }


def _full_prompt_reference(
    model: torch.nn.Module,
    tokenizer: Any,
    full_ids: torch.Tensor,
) -> dict[str, Any]:
    with torch.inference_mode():
        logits = model(full_ids.unsqueeze(0))[0].detach().cpu().numpy()
    result = _classification_summary(logits, tokenizer)
    result.update(
        {
            "token_count": int(full_ids.numel()),
            "evidence": "[V] upstream torch float32, complete public prompt",
        }
    )
    return result


def _global_calibration(
    per_block: list[dict[str, Any]],
    head: dict[str, np.ndarray],
    margin: float,
) -> dict[str, Any]:
    fields = tuple(per_block[0]["ranges"])
    global_ranges: dict[str, dict[str, float | int]] = {}
    for field in fields:
        values = [block["ranges"][field] for block in per_block]
        minimum = min(float(value["min"]) for value in values)
        maximum = max(float(value["max"]) for value in values)
        global_ranges[field] = {
            "min": minimum,
            "max": maximum,
            "abs_max": max(abs(minimum), abs(maximum)),
            "count": sum(int(value["count"]) for value in values),
        }

    head_ranges = {
        "final_ln_variance_plus_eps": finite_range(head["final_ln_variance"] + 1e-5),
        "final_ln_normalized_output": finite_range(head["final_ln_output"]),
        "head_silu_input": finite_range(head["head_linear"]),
        "head_silu_output": finite_range(head["head_silu"]),
        "head_ln_variance_plus_eps": finite_range(head["head_ln_variance"] + 1e-5),
        "head_ln_normalized_output": finite_range(head["head_ln_output"]),
        "head_logits": finite_range(head["logits"]),
    }
    recommended = {
        "ln1_variance_plus_eps": outward_domain(
            global_ranges["ln1_variance_plus_eps"], margin, positive=True
        ),
        "ln1_normalized_output": outward_domain(
            global_ranges["ln1_normalized_output"], margin, symmetric=True
        ),
        "attention_delta_s1_minus_s0": outward_domain(
            global_ranges["attention_delta_s1_minus_s0"], margin
        ),
        "attention_denominator": outward_domain(
            global_ranges["attention_denominator"], margin, positive=True
        ),
        "ln2_variance_plus_eps": outward_domain(
            global_ranges["ln2_variance_plus_eps"], margin, positive=True
        ),
        "ln2_normalized_output": outward_domain(
            global_ranges["ln2_normalized_output"], margin, symmetric=True
        ),
        "gelu_input": outward_domain(
            global_ranges["gelu_input"], margin, symmetric=True
        ),
        "head_silu_input": outward_domain(
            head_ranges["head_silu_input"], margin, symmetric=True
        ),
        "final_ln_variance_plus_eps": outward_domain(
            head_ranges["final_ln_variance_plus_eps"], margin, positive=True
        ),
        "head_ln_variance_plus_eps": outward_domain(
            head_ranges["head_ln_variance_plus_eps"], margin, positive=True
        ),
    }
    failures = [
        {
            "block": index,
            "fields": block["failed_block0_gpu_fixed_domains"],
        }
        for index, block in enumerate(per_block)
        if block["failed_block0_gpu_fixed_domains"]
    ]
    return {
        "evidence": (
            "[V] ranges on one fixed public T=2 sample; [A] outward domains "
            "are calibration assumptions, not private-query-adaptive bounds"
        ),
        "margin_multiplier": margin,
        "block0_gpu_fixed_domains": {
            key: list(value) for key, value in BLOCK0_GPU_DOMAINS.items()
        },
        "per_block": per_block,
        "global_ranges": global_ranges,
        "head_ranges": head_ranges,
        "recommended_global_public_domains": recommended,
        "block0_gpu_fixed_domain_failures": failures,
        "block0_gpu_fixed_domains_compose_all_12": not failures,
    }


def export_fixture(
    checkpoint: Path,
    positive_fasta: Path,
    output_dir: Path,
    *,
    token_count: int = 2,
    margin: float = DEFAULT_MARGIN,
) -> Path:
    """Create the immutable fixture and return its manifest path."""
    if token_count != 2:
        raise ValueError(
            "this bridge is intentionally specialized to T=2 non-degenerate attention"
        )
    if margin <= 1.0:
        raise ValueError("calibration margin must exceed one")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixture directory: {output_dir}")

    records = verify_positive_fasta(positive_fasta)
    record_header, sequence = records[0]
    prompt = gsr_prompt(sequence)
    model, tokenizer, state = build_verified_model(checkpoint)
    full_ids = tokenizer.encode(
        prompt, max_len=model.max_len, pad=False, device=torch.device("cpu")
    )
    ids = full_ids[:token_count].unsqueeze(0)
    token_pieces = tokenizer.tokenize(prompt)[:token_count]

    with torch.inference_mode():
        token_embeddings = model.transformer.wte(ids)
        positions = torch.arange(token_count, dtype=torch.long).unsqueeze(0)
        position_embeddings = model.transformer.wpe(positions)
        upstream_embeddings = model._embedding_impl(ids)
        if not torch.equal(upstream_embeddings, token_embeddings + position_embeddings):
            raise AssertionError("upstream embedding reproduction failed")

    manual_x = upstream_embeddings[0].detach().cpu().numpy().astype(np.float64)
    upstream_x = upstream_embeddings
    block_oracles: list[dict[str, np.ndarray]] = []
    upstream_outputs: list[np.ndarray] = []
    block_gates: list[dict[str, Any]] = []
    per_block_calibration: list[dict[str, Any]] = []
    for layer, block in enumerate(model.transformer.h):
        weights = _block_weights(state, layer)
        oracle = block_reference(
            manual_x,
            weights,
            num_heads=model.num_heads,
            eps=block.ln_1.eps,
        )
        manual_x = oracle["block_output"]
        with torch.inference_mode():
            upstream_x = block(upstream_x)
        upstream_np = upstream_x[0].detach().cpu().numpy().astype(np.float64)
        max_abs = float(np.max(np.abs(manual_x - upstream_np)))
        rel_inf = relative_inf(upstream_np, manual_x)
        passed = bool(np.allclose(manual_x, upstream_np, rtol=2e-5, atol=2e-5))
        if not passed:
            raise AssertionError(
                f"manual 12-block lineage diverged at block {layer}: "
                f"max_abs={max_abs:.6g}, rel_inf={rel_inf:.6g}"
            )
        block_oracles.append(oracle)
        upstream_outputs.append(upstream_np)
        block_gates.append(
            {
                "block": layer,
                "max_abs_error": max_abs,
                "rel_inf_error": rel_inf,
                "rtol": 2e-5,
                "atol": 2e-5,
                "passed": True,
            }
        )
        calibration = block_calibration(oracle, margin=margin)
        calibration["block"] = layer
        per_block_calibration.append(calibration)

    head_weights = {
        "final_ln": state["transformer.ln_f.weight"].detach().cpu().numpy(),
        "head_linear": state["mlm_head.0.weight"].detach().cpu().numpy(),
        "head_ln": state["mlm_head.2.weight"].detach().cpu().numpy(),
        "head_readout": state["mlm_head.3.weight"].detach().cpu().numpy(),
    }
    head_oracle = classifier_head_reference(manual_x, head_weights)
    with torch.inference_mode():
        upstream_final = model.transformer.ln_f(upstream_x)
        upstream_logits = model.mlm_head(upstream_final)[0].detach().cpu().numpy()
    head_max_abs = float(np.max(np.abs(head_oracle["logits"] - upstream_logits)))
    head_rel_inf = relative_inf(upstream_logits, head_oracle["logits"])
    if not np.allclose(head_oracle["logits"], upstream_logits, rtol=2e-5, atol=2e-5):
        raise AssertionError(
            "manual classifier head diverged from upstream torch: "
            f"max_abs={head_max_abs:.6g}, rel_inf={head_rel_inf:.6g}"
        )

    truncated_classification = _classification_summary(head_oracle["logits"], tokenizer)
    full_reference = _full_prompt_reference(model, tokenizer, full_ids)
    calibration = _global_calibration(per_block_calibration, head_oracle, margin)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{output_dir.name}.partial.", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary)
        arrays: OrderedDict[str, dict[str, Any]] = OrderedDict()

        for export_name, state_name in block_weight_keys(model.num_layers).items():
            _write_array(
                staging,
                f"weights.{export_name}",
                state[state_name].detach().cpu().numpy(),
                f"checkpoint:{state_name}",
                arrays,
            )
        _write_array(
            staging,
            "weights.final_ln",
            head_weights["final_ln"],
            "checkpoint:transformer.ln_f.weight",
            arrays,
        )
        _write_array(
            staging,
            "weights.head_linear",
            head_weights["head_linear"],
            "checkpoint:mlm_head.0.weight",
            arrays,
        )
        _write_array(
            staging,
            "weights.head_ln",
            head_weights["head_ln"],
            "checkpoint:mlm_head.2.weight",
            arrays,
        )
        n_id = truncated_classification["n_token_id"]
        a_id = truncated_classification["a_token_id"]
        _write_array(
            staging,
            "weights.head_readout_n",
            head_weights["head_readout"][n_id],
            "checkpoint:mlm_head.3.weight row for token N",
            arrays,
        )
        _write_array(
            staging,
            "weights.head_readout_a",
            head_weights["head_readout"][a_id],
            "checkpoint:mlm_head.3.weight row for token A",
            arrays,
        )
        _write_array(
            staging,
            "weights.head_readout_margin_n_minus_a",
            head_weights["head_readout"][n_id] - head_weights["head_readout"][a_id],
            "derived exact N-minus-A classifier row; encrypted scalar readout",
            arrays,
        )

        _write_array(
            staging,
            "input.token_embeddings",
            token_embeddings[0].detach().cpu().numpy(),
            "upstream tokenizer lookup",
            arrays,
        )
        _write_array(
            staging,
            "input.position_embeddings",
            position_embeddings[0].detach().cpu().numpy(),
            "upstream position lookup",
            arrays,
        )
        _write_array(
            staging,
            "input.embeddings",
            upstream_embeddings[0].detach().cpu().numpy(),
            "token plus position embedding; encrypted numeric input boundary",
            arrays,
        )

        for layer, oracle in enumerate(block_oracles):
            for field, value in oracle.items():
                _write_array(
                    staging,
                    f"oracle.block{layer}.{field}",
                    value,
                    f"independent NumPy float64 block-{layer} lineage",
                    arrays,
                )
            _write_array(
                staging,
                f"oracle.block{layer}.upstream_torch_output",
                upstream_outputs[layer],
                f"upstream torch float32 block-{layer} output",
                arrays,
            )
        for field, value in head_oracle.items():
            _write_array(
                staging,
                f"oracle.head.{field}",
                value,
                "independent NumPy float64 final-LN/classifier reference",
                arrays,
            )

        token_ids = [int(value) for value in ids[0].tolist()]
        content_identity = _canonical_content_identity(arrays, token_ids)
        metadata: dict[str, Any] = {
            "schema": SCHEMA,
            "content_identity_sha256": content_identity,
            "evidence": (
                "[V] released real weights, public T=2 prefix, independent "
                "NumPy float64 12-block lineage and classifier readout"
            ),
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
                "record_header": record_header,
                "record_length_bp": len(sequence),
                "record_sequence_sha256": sha256_text(sequence),
                "transform": ("remove sequence[300:306], asserted equal to AATAAA"),
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
                "token_ids": token_ids,
                "token_pieces": token_pieces,
            },
            "classification": {
                "truncated_t2_graph_gate": {
                    **truncated_classification,
                    "evidence": (
                        "[V] manual NumPy float64; T=2 is a graph gate and "
                        "does not preserve the full GSR task semantics"
                    ),
                },
                "complete_public_prompt_reference": full_reference,
                "encrypted_readout": (
                    "last token -> final LN -> head linear -> SiLU -> head LN "
                    "-> dot(head_readout_N - head_readout_A); sign selects N/A"
                ),
            },
            "oracle_gate": {
                "manual_reference": (
                    "single NumPy float64 lineage through all 12 blocks and head"
                ),
                "upstream_reference": "torch float32 DNAGPT",
                "per_block": block_gates,
                "head": {
                    "max_abs_error": head_max_abs,
                    "rel_inf_error": head_rel_inf,
                    "rtol": 2e-5,
                    "atol": 2e-5,
                    "passed": True,
                },
                "all_finite": True,
                "passed": True,
            },
            "public_calibration": calibration,
            "format": {
                "description": (
                    "Headerless little-endian float64 arrays; reshape in C "
                    "row-major order using the recorded shape."
                ),
                "export_dtype": "float64",
                "endianness": "little",
                "order": "C",
            },
            "arrays": arrays,
            "source": {
                "export_fixture_sha256": sha256_file(Path(__file__)),
                "manual_sha256": sha256_file(Path(__file__).with_name("manual.py")),
                "realweights_contract_sha256": sha256_file(
                    Path(__file__).parents[1] / "realweights" / "contract.py"
                ),
                "realweights_manual_sha256": sha256_file(
                    Path(__file__).parents[1] / "realweights" / "manual.py"
                ),
            },
        }
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        checksums_path = staging / "SHA256SUMS"
        with checksums_path.open("x", encoding="ascii") as handle:
            for item in arrays.values():
                handle.write(f"{item['sha256']}  {item['file']}\n")
            handle.write(f"{sha256_file(manifest_path)}  manifest.json\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output_dir)
    return output_dir / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--positive-fasta", type=Path, default=DEFAULT_POSITIVE_FASTA)
    parser.add_argument("--tokens", "-T", type=int, default=2)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        args.output.resolve()
        if args.output is not None
        else _default_output(args.tokens).resolve()
    )
    manifest_path = export_fixture(
        args.checkpoint.resolve(),
        args.positive_fasta.resolve(),
        output,
        token_count=args.tokens,
        margin=args.margin,
    )
    result = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "content_identity_sha256": result["content_identity_sha256"],
                "arrays": len(result["arrays"]),
                "bytes": sum(int(item["bytes"]) for item in result["arrays"].values()),
                "T": result["tokenization"]["exported_token_count_T"],
                "margin_n_minus_a": result["classification"]["truncated_t2_graph_gate"][
                    "margin_n_minus_a"
                ],
                "binary_label": result["classification"]["truncated_t2_graph_gate"][
                    "binary_label"
                ],
                "fixed_domains_compose_all_12": result["public_calibration"][
                    "block0_gpu_fixed_domains_compose_all_12"
                ],
                "passed": result["oracle_gate"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
