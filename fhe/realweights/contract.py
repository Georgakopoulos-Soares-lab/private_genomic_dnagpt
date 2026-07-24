"""Frozen input and model contracts for the real-weight FHE path."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DNAGPT_DIR = REPO_ROOT / "DNAGPT"
if str(DNAGPT_DIR) not in sys.path:
    sys.path.insert(0, str(DNAGPT_DIR))

from dna_gpt.model import DNAGPT  # noqa: E402
from dna_gpt.tokenizer import KmerTokenizer  # noqa: E402

MODEL_NAME = "dna_gpt0.1b_m"
CHECKPOINT_SHA256 = "d63353abdc1adba18e076b8f8b1ee9cc4d55f6948361324a825ea2245caf55f5"
POSITIVE_FASTA_SHA256 = (
    "52d046d1fcf0f03bbaa4b971a1baa6b3fdbe29bc576bcdea8fe28f619efd2f11"
)
MOTIF_START = 300
MOTIF = "AATAAA"

# Copied verbatim from upstream DNAGPT/test.py:get_model.
SPECIAL_TOKENS = (
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    + ["+", "-", "*", "/", "=", "&", "|", "!"]
    + ["M", "B"]
    + ["P"]
    + ["R", "I", "K", "L", "O", "Q", "S", "U", "V"]
    + ["W", "Y", "X", "Z"]
)

EXPECTED_MISSING_KEYS = (
    "number_embedding.0.weight",
    "number_embedding.2.weight",
    "number_embedding.3.weight",
    "num_regression.0.weight",
    "num_regression.2.weight",
    "num_regression.3.weight",
)

BLOCK0_EXPORT_KEYS = {
    "ln1": "transformer.h.0.ln_1.weight",
    "attn_qkv": "transformer.h.0.attn.c_attn.weight",
    "attn_proj": "transformer.h.0.attn.c_proj.weight",
    "ln2": "transformer.h.0.ln_2.weight",
    "mlp_fc": "transformer.h.0.mlp.c_fc.weight",
    "mlp_proj": "transformer.h.0.mlp.c_proj.weight",
}


def expected_checkpoint_keys() -> tuple[str, ...]:
    """Return the exact released classification checkpoint key contract."""
    keys = ["transformer.wte.weight", "transformer.wpe.weight"]
    for layer in range(12):
        prefix = f"transformer.h.{layer}"
        keys.extend(
            (
                f"{prefix}.attn.c_attn.weight",
                f"{prefix}.attn.c_proj.weight",
                f"{prefix}.mlp.c_fc.weight",
                f"{prefix}.mlp.c_proj.weight",
                f"{prefix}.ln_1.weight",
                f"{prefix}.ln_2.weight",
            )
        )
    keys.extend(
        (
            "transformer.ln_f.weight",
            "mlm_head.0.weight",
            "mlm_head.2.weight",
            "mlm_head.3.weight",
        )
    )
    return tuple(keys)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def repo_relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def verify_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual} ({path})"
        )
    return actual


def make_tokenizer() -> KmerTokenizer:
    """Mirror upstream ``get_model('dna_gpt0.1b_m')`` exactly."""
    return KmerTokenizer(6, SPECIAL_TOKENS, dynamic_kmer=True)


def load_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    """Load weights safely and enforce the exact released-key contract."""
    verify_sha256(path, CHECKPOINT_SHA256, "classification checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if "model" in payload else payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state must be a mapping, got {type(state)!r}")
    expected = set(expected_checkpoint_keys())
    actual = set(state)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"checkpoint key contract mismatch: missing={missing}, unexpected={extra}"
        )
    if any(key.endswith(".bias") for key in actual):
        raise ValueError("bias-free checkpoint unexpectedly contains bias tensors")
    return state


def build_verified_model(
    checkpoint: Path,
) -> tuple[DNAGPT, KmerTokenizer, dict[str, torch.Tensor]]:
    """Build the upstream model, load the release, and verify missing keys."""
    state = load_checkpoint_state(checkpoint)
    tokenizer = make_tokenizer()
    model = DNAGPT.from_name(MODEL_NAME, len(tokenizer))
    incompat = model.load_state_dict(state, strict=False)
    if tuple(incompat.missing_keys) != EXPECTED_MISSING_KEYS:
        raise ValueError(
            "model missing-key contract mismatch: "
            f"expected {EXPECTED_MISSING_KEYS}, got {tuple(incompat.missing_keys)}"
        )
    if incompat.unexpected_keys:
        raise ValueError(f"unexpected model keys: {incompat.unexpected_keys}")
    model.to(device="cpu", dtype=torch.float32).eval()
    return model, tokenizer, state


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    pieces: list[str] = []
    with path.open(encoding="ascii") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(pieces).upper()))
                header = line[1:]
                pieces = []
            else:
                if header is None:
                    raise ValueError(f"FASTA sequence precedes header in {path}")
                pieces.append(line)
    if header is not None:
        records.append((header, "".join(pieces).upper()))
    if not records:
        raise ValueError(f"no FASTA records found in {path}")
    return records


def verify_positive_fasta(path: Path) -> list[tuple[str, str]]:
    verify_sha256(path, POSITIVE_FASTA_SHA256, "positive GSR FASTA")
    return read_fasta(path)


def strip_central_motif(sequence: str) -> str:
    """Apply the released GSR 606→600 transform with strict validation."""
    if len(sequence) != 606:
        raise ValueError(f"expected a 606 bp GSR record, got {len(sequence)} bp")
    observed = sequence[MOTIF_START : MOTIF_START + len(MOTIF)]
    if observed != MOTIF:
        raise ValueError(
            f"expected central motif {MOTIF!r}, got {observed!r} at {MOTIF_START}"
        )
    return sequence[:MOTIF_START] + sequence[MOTIF_START + len(MOTIF) :]


def gsr_prompt(sequence_606: str) -> str:
    """Match ``eval/common.py:classify`` and upstream classification scripts."""
    return f"<R>{strip_central_motif(sequence_606)}<=><R>"


def ensure_all_finite(tensors: Iterable[torch.Tensor], label: str) -> None:
    for tensor in tensors:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite values in {label}")
