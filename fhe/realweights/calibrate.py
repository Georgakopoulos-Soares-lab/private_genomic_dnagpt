"""Calibrate fixed public polynomial domains from real GSR activations.

This is plaintext, public preprocessing. It never exports weights or sequence
contents. The result is an immutable JSON file and is created only for a new
explicit tag.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .contract import (
    CHECKPOINT_SHA256,
    POSITIVE_FASTA_SHA256,
    REPO_ROOT,
    build_verified_model,
    gsr_prompt,
    read_fasta,
    repo_relative_or_absolute,
    sha256_file,
    sha256_text,
    verify_positive_fasta,
)

SCHEMA = "dnagpt.realweights.public_calibration.v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/classification.pth"
DEFAULT_POSITIVE_FASTA = REPO_ROOT / "data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa"
DEFAULT_NEGATIVE_FASTA = REPO_ROOT / "data/gsr/Data/Human/PAS/hs_negAATAAA_polyA.fa"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/runs"
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class Range:
    minimum: float = math.inf
    maximum: float = -math.inf
    count: int = 0

    def update(self, tensor: torch.Tensor) -> None:
        value = tensor.detach()
        if value.numel() == 0:
            raise ValueError("cannot calibrate an empty tensor")
        if not torch.isfinite(value).all():
            raise ValueError("calibration encountered a non-finite tensor")
        self.minimum = min(self.minimum, float(torch.min(value)))
        self.maximum = max(self.maximum, float(torch.max(value)))
        self.count += int(value.numel())

    def as_json(self) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("range has no observations")
        return {
            "min": self.minimum,
            "max": self.maximum,
            "abs_max": max(abs(self.minimum), abs(self.maximum)),
            "count": self.count,
        }


FIELDS = (
    "pre_ln1_variance_plus_eps",
    "ln1_output",
    "attention_scores_causal",
    "attention_probabilities_causal",
    "post_attention_residual",
    "pre_ln2_variance_plus_eps",
    "ln2_output",
    "gelu_input",
    "gelu_output",
    "block_output",
)


def _new_layer_ranges(layer_count: int) -> list[dict[str, Range]]:
    return [{field: Range() for field in FIELDS} for _ in range(layer_count)]


def _instrumented_block(
    block: torch.nn.Module, x: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Explicit upstream-equivalent forward that exposes polynomial inputs."""
    _, token_count, width = x.shape
    heads = block.attn.num_heads
    head_dim = width // heads
    eps = block.ln_1.eps

    variance1 = torch.var(x, dim=-1, unbiased=False) + eps
    ln1 = block.ln_1(x)
    qkv = block.attn.c_attn(ln1)
    query, key, value = qkv.split(width, dim=-1)
    query = query.view(1, token_count, heads, head_dim).transpose(1, 2)
    key = key.view(1, token_count, heads, head_dim).transpose(1, 2)
    value = value.view(1, token_count, heads, head_dim).transpose(1, 2)
    scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = torch.tril(
        torch.ones(token_count, token_count, dtype=torch.bool, device=x.device)
    )
    masked_scores = scores.masked_fill(~causal_mask[None, None, :, :], -torch.inf)
    probabilities = torch.softmax(masked_scores, dim=-1)
    context = probabilities @ value
    context = context.transpose(1, 2).contiguous().view(1, token_count, width)
    attention_projection = block.attn.c_proj(context)
    residual1 = x + attention_projection

    variance2 = torch.var(residual1, dim=-1, unbiased=False) + block.ln_2.eps
    ln2 = block.ln_2(residual1)
    gelu_input = block.mlp.c_fc(ln2)
    gelu_output = F.gelu(gelu_input, approximate="tanh")
    output = residual1 + block.mlp.c_proj(gelu_output)

    stats = {
        "pre_ln1_variance_plus_eps": variance1,
        "ln1_output": ln1,
        "attention_scores_causal": scores[..., causal_mask],
        "attention_probabilities_causal": probabilities[..., causal_mask],
        "post_attention_residual": residual1,
        "pre_ln2_variance_plus_eps": variance2,
        "ln2_output": ln2,
        "gelu_input": gelu_input,
        "gelu_output": gelu_output,
        "block_output": output,
    }
    return output, stats


def _outward_round(value: float, quantum: float, upper: bool) -> float:
    scaled = value / quantum
    rounded = math.ceil(scaled) if upper else math.floor(scaled)
    return float(rounded * quantum)


def _symmetric_domain(value: Range, margin: float, quantum: float = 0.1) -> list[float]:
    radius = max(abs(value.minimum), abs(value.maximum)) * margin
    radius = _outward_round(radius, quantum, upper=True)
    return [-radius, radius]


def _positive_domain(value: Range, margin: float) -> list[float]:
    lower = max(1e-6, value.minimum / margin)
    upper = value.maximum * margin
    return [
        _outward_round(lower, 1e-6, upper=False),
        _outward_round(upper, 0.01, upper=True),
    ]


def _asymmetric_domain(value: Range, margin: float) -> list[float]:
    lower = value.minimum * margin if value.minimum < 0 else value.minimum / margin
    upper = value.maximum * margin if value.maximum > 0 else value.maximum / margin
    return [
        _outward_round(lower, 0.1, upper=False),
        _outward_round(upper, 0.1, upper=True),
    ]


def _global_range(layer_ranges: list[dict[str, Range]], field: str) -> Range:
    result = Range()
    for layer in layer_ranges:
        item = layer[field]
        result.minimum = min(result.minimum, item.minimum)
        result.maximum = max(result.maximum, item.maximum)
        result.count += item.count
    return result


def _recommend_domains(
    layer_ranges: list[dict[str, Range]], margin: float
) -> dict[str, Any]:
    variance1 = _global_range(layer_ranges, "pre_ln1_variance_plus_eps")
    variance2 = _global_range(layer_ranges, "pre_ln2_variance_plus_eps")
    variance = Range(
        minimum=min(variance1.minimum, variance2.minimum),
        maximum=max(variance1.maximum, variance2.maximum),
        count=variance1.count + variance2.count,
    )
    ln1 = _global_range(layer_ranges, "ln1_output")
    ln2 = _global_range(layer_ranges, "ln2_output")
    normalized = Range(
        minimum=min(ln1.minimum, ln2.minimum),
        maximum=max(ln1.maximum, ln2.maximum),
        count=ln1.count + ln2.count,
    )
    scores = _global_range(layer_ranges, "attention_scores_causal")
    gelu_input = _global_range(layer_ranges, "gelu_input")
    return {
        "evidence": "[A] sample-calibrated fixed domains; validate on a broader set",
        "margin_multiplier": margin,
        "layernorm_inverse_sqrt_input_variance_plus_eps": _positive_domain(
            variance, margin
        ),
        "layernorm_normalized_output": _symmetric_domain(normalized, margin),
        "attention_score_before_mask_and_softmax": _asymmetric_domain(scores, margin),
        "gelu_input": _symmetric_domain(gelu_input, margin),
        "method": (
            "Global extrema across every sampled token and all 12 layers, "
            "expanded outward by margin_multiplier and rounded outward."
        ),
    }


def _select_samples(
    positive_records: list[tuple[str, str]],
    negative_records: list[tuple[str, str]],
    count: int,
    positive_only: bool,
) -> list[tuple[str, int, int, str, str]]:
    if count <= 0:
        raise ValueError(f"--samples must be positive, got {count}")
    selected: list[tuple[str, int, int, str, str]] = []
    if positive_only:
        if count > len(positive_records):
            raise ValueError(
                f"requested {count} positives; only {len(positive_records)}"
            )
        return [
            ("positive", 1, index, header, sequence)
            for index, (header, sequence) in enumerate(positive_records[:count])
        ]
    pos_index = 0
    neg_index = 0
    while len(selected) < count:
        if pos_index < len(positive_records):
            header, sequence = positive_records[pos_index]
            selected.append(("positive", 1, pos_index, header, sequence))
            pos_index += 1
            if len(selected) == count:
                break
        if neg_index < len(negative_records):
            header, sequence = negative_records[neg_index]
            selected.append(("negative", 0, neg_index, header, sequence))
            neg_index += 1
        if pos_index >= len(positive_records) and neg_index >= len(negative_records):
            break
    if len(selected) != count:
        raise ValueError(f"requested {count} samples; only selected {len(selected)}")
    return selected


def calibrate(
    checkpoint: Path,
    positive_fasta: Path,
    negative_fasta: Path,
    sample_count: int,
    max_tokens: int,
    margin: float,
    positive_only: bool,
) -> dict[str, Any]:
    if max_tokens <= 0:
        raise ValueError(f"--max-tokens must be positive, got {max_tokens}")
    if margin <= 1.0:
        raise ValueError(f"--margin must exceed 1.0, got {margin}")

    positive_records = verify_positive_fasta(positive_fasta)
    negative_records = [] if positive_only else read_fasta(negative_fasta)
    samples = _select_samples(
        positive_records, negative_records, sample_count, positive_only
    )
    model, tokenizer, _ = build_verified_model(checkpoint)
    ranges = _new_layer_ranges(model.num_layers)
    sample_metadata: list[dict[str, Any]] = []
    max_instrumentation_error = 0.0
    started = time.perf_counter()

    with torch.inference_mode():
        for class_name, label, record_index, header, sequence in samples:
            prompt = gsr_prompt(sequence)
            ids = tokenizer.encode(
                prompt,
                max_len=min(max_tokens, model.max_len),
                pad=False,
                device=torch.device("cpu"),
            ).unsqueeze(0)
            x = model._embedding_impl(ids)
            for layer_index, block in enumerate(model.transformer.h):
                explicit_output, stats = _instrumented_block(block, x)
                upstream_output = block(x)
                error = float(torch.max(torch.abs(explicit_output - upstream_output)))
                max_instrumentation_error = max(max_instrumentation_error, error)
                if not torch.allclose(
                    explicit_output, upstream_output, rtol=2e-5, atol=2e-5
                ):
                    raise AssertionError(
                        "instrumented block diverged from upstream: "
                        f"layer={layer_index}, max_abs={error:.6g}"
                    )
                for field, tensor in stats.items():
                    ranges[layer_index][field].update(tensor)
                x = explicit_output
            sample_metadata.append(
                {
                    "class": class_name,
                    "label": label,
                    "record_index": record_index,
                    "record_header": header,
                    "record_sequence_sha256": sha256_text(sequence),
                    "prompt_sha256": sha256_text(prompt),
                    "token_count": int(ids.shape[1]),
                }
            )

    elapsed = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "evidence": "[V] plaintext real-weight calibration",
        "model": {
            "name": "dna_gpt0.1b_m",
            "checkpoint": repo_relative_or_absolute(checkpoint),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "dtype": "float32",
            "device": "cpu",
            "layers": model.num_layers,
            "heads": model.num_heads,
            "embedding_dim": model.embedding_dim,
        },
        "datasets": {
            "positive": {
                "path": repo_relative_or_absolute(positive_fasta),
                "sha256": POSITIVE_FASTA_SHA256,
            },
            "negative": None
            if positive_only
            else {
                "path": repo_relative_or_absolute(negative_fasta),
                "sha256": sha256_file(negative_fasta),
            },
        },
        "config": {
            "samples": sample_count,
            "selection": "first records, positive/negative interleaved"
            if not positive_only
            else "first positive records",
            "max_tokens": max_tokens,
            "domain_margin_multiplier": margin,
        },
        "sample_identities": sample_metadata,
        "per_layer_ranges": [
            {
                "layer": layer_index,
                **{field: values[field].as_json() for field in FIELDS},
            }
            for layer_index, values in enumerate(ranges)
        ],
        "recommended_fixed_public_domains": _recommend_domains(ranges, margin),
        "verification": {
            "instrumented_vs_upstream_rtol": 2e-5,
            "instrumented_vs_upstream_atol": 2e-5,
            "max_abs_error": max_instrumentation_error,
            "all_finite": True,
            "passed": True,
        },
        "elapsed_seconds": elapsed,
        "source": {
            "calibrate_sha256": sha256_file(Path(__file__)),
            "contract_sha256": sha256_file(Path(__file__).with_name("contract.py")),
        },
    }


def write_immutable_result(result: dict[str, Any], output_dir: Path, tag: str) -> Path:
    if not TAG_RE.fullmatch(tag):
        raise ValueError(f"tag must match [A-Za-z0-9][A-Za-z0-9_.-]*; got {tag!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{tag}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration result: {output}")
    result = {"tag": tag, **result}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="new immutable result tag")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--positive-fasta", type=Path, default=DEFAULT_POSITIVE_FASTA)
    parser.add_argument("--negative-fasta", type=Path, default=DEFAULT_NEGATIVE_FASTA)
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="total samples; interleaves positive and negative records",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="calibrate only the first positive records",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="upstream tokenizer truncation limit",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.25,
        help="outward multiplier for recommended domains (must exceed 1)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not TAG_RE.fullmatch(args.tag):
        raise ValueError(f"tag must match [A-Za-z0-9][A-Za-z0-9_.-]*; got {args.tag!r}")
    output_dir = args.output_dir.resolve()
    intended_output = output_dir / f"{args.tag}.json"
    if intended_output.exists():
        raise FileExistsError(
            f"refusing to overwrite calibration result: {intended_output}"
        )
    result = calibrate(
        args.checkpoint.resolve(),
        args.positive_fasta.resolve(),
        args.negative_fasta.resolve(),
        args.samples,
        args.max_tokens,
        args.margin,
        args.positive_only,
    )
    output = write_immutable_result(result, output_dir, args.tag)
    print(
        json.dumps(
            {
                "output": str(output),
                "samples": result["config"]["samples"],
                "elapsed_seconds": result["elapsed_seconds"],
                "recommended_fixed_public_domains": result[
                    "recommended_fixed_public_domains"
                ],
                "passed": result["verification"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
