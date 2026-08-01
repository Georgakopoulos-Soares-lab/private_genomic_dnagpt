"""Export a deterministic all-12-block DNAGPT T=103 plaintext/FHE fixture.

Fork discipline:
  This is an additive sibling of ``export_fixture.py`` (the frozen T=2
  12-block-plus-head exporter), not an edit of it. ``export_fixture.py``'s
  ``export_fixture()`` hard-rejects any ``token_count != 2``
  ("this bridge is intentionally specialized to T=2 non-degenerate
  attention"), so it cannot be reused unmodified for the T=103 composition
  target. Nothing else in that module is T=2-specific *except* the T=2-only
  calibration helpers in ``fhe/multiblock/manual.py``
  (``t2_attention_schedule``/``block_calibration``, which hard-assert
  ``attention_scores.shape[1:] == (2, 2)``). Those helpers exist only to
  produce Scheme-A-era Chebyshev public-domain calibration metadata, which
  Scheme B does not use (Scheme B evaluates every nonlinearity exactly at a
  client boundary, never via a fixed public polynomial domain). This fork
  therefore reuses every T-general helper from ``export_fixture.py``
  unchanged (block weight export, ``_write_array``, ``_canonical_content_
  identity``, ``_classification_summary``, ``_full_prompt_reference``) and
  simply omits the T=2-only calibration section, replacing it with an
  explicit "not computed" placeholder so no manifest reader can mistake its
  absence for silent data loss.

  At T=103 the exported prefix is the *complete* tokenized GSR prompt for
  this FASTA record (``full_prompt_token_count == exported_token_count_T
  == 103``), unlike the T=2 bridge, which is an explicitly-labeled
  "graph gate" that does not preserve full GSR task semantics. So this
  fixture's classification summary is real GSR task semantics, not a
  truncated-prefix proxy.

Measured finding (this fork, T=103): the frozen exporter's manual-vs-
upstream consistency check compares a float64 manual NumPy reimplementation
against the released model run in its Phase-A-locked float32 dtype
(``fhe/realweights/contract.py:build_verified_model`` hardcodes
``dtype=torch.float32`` and is not edited here). At T=2 that check passes
comfortably at rtol=atol=2e-5. At T=103, block-by-block divergence against
the float32 upstream grows to ~4.8e-4 by block 6 and plateaus near ~8e-4 by
block 11 -- this exceeds 2e-5 and the frozen T=2 exporter's assertion would
raise. Diagnosis (measured, see the deep-dive that produced this fork):
running an *additional*, float64-cast copy of the same released weights
through the same upstream torch forward pass reproduces the manual float64
chain to within ~5e-7 to ~5e-6 at every block, i.e. the divergence against
the float32 upstream is float32 rounding-accumulation over up to 103
attention/softmax terms per block compounded across blocks, not a defect in
the manual reimplementation. This fork's own consistency gate therefore
checks the manual chain against a float64-cast shadow copy of the upstream
model (built locally in this file, not by editing the frozen
``build_verified_model``, which stays float32 for Phase-A parity), and
additionally records the (larger, expected) float32 divergence per block as
diagnostic evidence rather than silently dropping it.

Large generated arrays live under the ignored ``checkpoints/fhe_exports``
tree. This module only writes the reproducible exporter.
"""

from __future__ import annotations

import argparse
import copy
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

from .export_fixture import (
    _canonical_content_identity,
    _classification_summary,
    _full_prompt_reference,
    _write_array,
    block_weight_keys,
)
from .manual import classifier_head_reference

SCHEMA = "dnagpt.multiblock.gsr_general.v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/classification.pth"
DEFAULT_POSITIVE_FASTA = REPO_ROOT / "data/gsr/Data/Human/PAS/hs_AATAAA_polyA.fa"
DEFAULT_TOKENS = 103


def _default_output(token_count: int) -> Path:
    identity = (
        f"gsr_pos0_multiblock_t{token_count}_"
        f"{CHECKPOINT_SHA256[:12]}_{POSITIVE_FASTA_SHA256[:12]}"
    )
    return REPO_ROOT / "checkpoints/fhe_exports" / identity


def _block_weights(state: dict[str, torch.Tensor], layer: int) -> dict[str, np.ndarray]:
    from .export_fixture import BLOCK_WEIGHT_SUFFIXES

    return {
        short_name: state[f"transformer.h.{layer}.{suffix}"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
        for short_name, suffix in BLOCK_WEIGHT_SUFFIXES.items()
    }


def export_fixture_general(
    checkpoint: Path,
    positive_fasta: Path,
    output_dir: Path,
    *,
    token_count: int,
) -> Path:
    """Create the immutable general-T 12-block+head fixture and return its manifest path.

    Unlike ``export_fixture.export_fixture`` this accepts any positive
    ``token_count`` up to the full tokenized prompt length. It does not
    compute the T=2-only Scheme-A calibration section.
    """
    if token_count <= 0:
        raise ValueError(f"T must be positive, got {token_count}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixture directory: {output_dir}")

    records = verify_positive_fasta(positive_fasta)
    record_header, sequence = records[0]
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
    token_pieces = tokenizer.tokenize(prompt)[:token_count]

    # Float64-cast shadow copy of the exact same released weights, used only
    # as a tighter internal consistency oracle (see module docstring). The
    # frozen build_verified_model()/Phase-A model stays float32, unedited.
    model64 = copy.deepcopy(model).to(dtype=torch.float64).eval()

    with torch.inference_mode():
        token_embeddings = model.transformer.wte(ids)
        positions = torch.arange(token_count, dtype=torch.long).unsqueeze(0)
        position_embeddings = model.transformer.wpe(positions)
        upstream_embeddings = model._embedding_impl(ids)
        if not torch.equal(upstream_embeddings, token_embeddings + position_embeddings):
            raise AssertionError("upstream embedding reproduction failed")
        upstream64_embeddings = model64._embedding_impl(ids)

    manual_x = upstream_embeddings[0].detach().cpu().numpy().astype(np.float64)
    upstream_x = upstream_embeddings
    upstream64_x = upstream64_embeddings
    block_oracles: list[dict[str, np.ndarray]] = []
    upstream_outputs: list[np.ndarray] = []
    block_gates: list[dict[str, Any]] = []
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
            upstream64_x = model64.transformer.h[layer](upstream64_x)
        upstream_np = upstream_x[0].detach().cpu().numpy().astype(np.float64)
        upstream64_np = upstream64_x[0].detach().cpu().numpy().astype(np.float64)
        max_abs_fp32 = float(np.max(np.abs(manual_x - upstream_np)))
        rel_inf_fp32 = relative_inf(upstream_np, manual_x)
        max_abs = float(np.max(np.abs(manual_x - upstream64_np)))
        rel_inf = relative_inf(upstream64_np, manual_x)
        passed = bool(np.allclose(manual_x, upstream64_np, rtol=2e-5, atol=2e-5))
        if not passed:
            raise AssertionError(
                f"manual 12-block lineage diverged from the float64 shadow "
                f"upstream at block {layer}: max_abs={max_abs:.6g}, "
                f"rel_inf={rel_inf:.6g}"
            )
        block_oracles.append(oracle)
        upstream_outputs.append(upstream_np)
        block_gates.append(
            {
                "block": layer,
                "max_abs_error_vs_float64_shadow_upstream": max_abs,
                "rel_inf_error_vs_float64_shadow_upstream": rel_inf,
                "rtol": 2e-5,
                "atol": 2e-5,
                "passed": True,
                "diagnostic_max_abs_error_vs_float32_upstream": max_abs_fp32,
                "diagnostic_rel_inf_error_vs_float32_upstream": rel_inf_fp32,
                "diagnostic_evidence": (
                    "[V] measured float32 accumulation drift over T=103 "
                    "attention/softmax terms compounded across blocks; not "
                    "gated on, see module docstring"
                ),
            }
        )

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
        upstream64_final = model64.transformer.ln_f(upstream64_x)
        upstream64_logits = model64.mlm_head(upstream64_final)[0].detach().cpu().numpy()
    head_max_abs_fp32 = float(np.max(np.abs(head_oracle["logits"] - upstream_logits)))
    head_rel_inf_fp32 = relative_inf(upstream_logits, head_oracle["logits"])
    head_max_abs = float(np.max(np.abs(head_oracle["logits"] - upstream64_logits)))
    head_rel_inf = relative_inf(upstream64_logits, head_oracle["logits"])
    if not np.allclose(head_oracle["logits"], upstream64_logits, rtol=2e-5, atol=2e-5):
        raise AssertionError(
            "manual classifier head diverged from the float64 shadow "
            f"upstream: max_abs={head_max_abs:.6g}, rel_inf={head_rel_inf:.6g}"
        )

    truncated_classification = _classification_summary(head_oracle["logits"], tokenizer)
    full_reference = _full_prompt_reference(model, tokenizer, full_ids)
    is_full_prompt = token_count == int(full_ids.numel())

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
                "[V] released real weights, public T={} prefix, independent "
                "NumPy float64 12-block lineage and classifier readout"
            ).format(token_count),
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
                "is_full_prompt": is_full_prompt,
                "token_ids": token_ids,
                "token_pieces": token_pieces,
            },
            "classification": {
                "graph_gate": {
                    **truncated_classification,
                    "evidence": (
                        "[V] manual NumPy float64; is_full_prompt={} -- when "
                        "true this is real GSR task semantics, not a "
                        "truncated-prefix graph-only proxy"
                    ).format(is_full_prompt),
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
                    "max_abs_error_vs_float64_shadow_upstream": head_max_abs,
                    "rel_inf_error_vs_float64_shadow_upstream": head_rel_inf,
                    "rtol": 2e-5,
                    "atol": 2e-5,
                    "passed": True,
                    "diagnostic_max_abs_error_vs_float32_upstream": head_max_abs_fp32,
                    "diagnostic_rel_inf_error_vs_float32_upstream": head_rel_inf_fp32,
                },
                "all_finite": True,
                "passed": True,
            },
            "public_calibration": {
                "evidence": (
                    "[A] not computed for this general-T fork. The T=2-only "
                    "helpers in fhe/multiblock/manual.py "
                    "(t2_attention_schedule/block_calibration) hard-assert "
                    "attention_scores.shape[1:]==(2,2) and only ever fed "
                    "Scheme-A-era fixed-Chebyshev-domain calibration, which "
                    "Scheme B does not use (every Scheme B nonlinearity is "
                    "evaluated exactly at a client boundary)."
                ),
                "computed": False,
            },
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
                "frozen_export_fixture_sha256": sha256_file(
                    Path(__file__).with_name("export_fixture.py")
                ),
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
    parser.add_argument("--tokens", "-T", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        args.output.resolve()
        if args.output is not None
        else _default_output(args.tokens).resolve()
    )
    manifest_path = export_fixture_general(
        args.checkpoint.resolve(),
        args.positive_fasta.resolve(),
        output,
        token_count=args.tokens,
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
                "is_full_prompt": result["tokenization"]["is_full_prompt"],
                "margin_n_minus_a": result["classification"]["graph_gate"][
                    "margin_n_minus_a"
                ],
                "binary_label": result["classification"]["graph_gate"]["binary_label"],
                "passed": result["oracle_gate"]["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
