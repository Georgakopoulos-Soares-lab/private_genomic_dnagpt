"""Float64 and static contracts for Scheme B client-boundary batching."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np


D = 768
T = 2
HEADS = 12
HEAD_DIM = 64
PACK_WIDTH = 1024
COPIES = 4
SLOTS = COPIES * PACK_WIDTH
MLP_DIM = 3072

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).with_name("src") / "real_dnagpt_fides_scheme_b.cpp"
FIXTURE = (
    ROOT
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_array(name: str) -> np.ndarray:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["arrays"][name]
    path = FIXTURE / metadata["file"]
    if _sha256(path) != metadata["sha256"]:
        raise ValueError(f"fixture hash mismatch for {name}")
    return np.fromfile(path, dtype="<f8").reshape(metadata["shape"])


def _active_transform(packed: np.ndarray, transform: object) -> np.ndarray:
    values = np.asarray(packed, dtype=np.float64).reshape(COPIES, PACK_WIDTH)
    result = np.zeros_like(values)
    result[:, :D] = transform(values[:, :D])
    return result.reshape(SLOTS)


def _rotate(values: np.ndarray, index: int) -> np.ndarray:
    """OpenFHE/FIDES logical left rotation."""
    return np.roll(np.asarray(values), -index)


def _replicate_copies(selected: np.ndarray) -> np.ndarray:
    return (
        selected
        + _rotate(selected, PACK_WIDTH)
        + _rotate(selected, -PACK_WIDTH)
        + _rotate(selected, 2 * PACK_WIDTH)
    )


class SchemeBBatchingContractTests(unittest.TestCase):
    def test_static_boundary_and_security_contract(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        evaluator = source.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]

        self.assertIn("attention_sigmoid_heads_batched", source)
        self.assertIn("_chunks_batched", source)
        self.assertIn("cross_boundary_active", source)
        self.assertIn("replicate_copies", source)
        self.assertIn("logical_boundary_instances", source)
        self.assertNotIn("Decrypt(", evaluator)
        self.assertNotIn("secretKey", evaluator)

        # The four unchanged LayerNorm instances, one 12-head attention batch,
        # and two four-chunk GELU batches represent 24 logical nonlinearities
        # in seven physical client crossings.
        self.assertEqual(2 + 1 + 2 + 2, 7)
        self.assertEqual(2 + HEADS + 2 + T * 4, 24)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=2 fixture absent")
    def test_attention_heads_pack_into_one_boundary(self) -> None:
        scores = _load_array("oracle.attention_scores")
        values = _load_array("oracle.value")
        expected = _load_array("oracle.attention_context_merged")

        deltas = scores[:, 1, 1] - scores[:, 1, 0]
        packed = np.zeros((COPIES, PACK_WIDTH), dtype=np.float64)
        for head in range(HEADS):
            packed[:, head * HEAD_DIM : (head + 1) * HEAD_DIM] = deltas[head]

        def sigmoid(value):
            return 1.0 / (1.0 + np.exp(-value))

        weights = _active_transform(packed, sigmoid).reshape(COPIES, PACK_WIDTH)
        merged_values = values.transpose(1, 0, 2).reshape(T, D)
        got = np.stack(
            (
                merged_values[0],
                merged_values[0]
                + weights[0, :D] * (merged_values[1] - merged_values[0]),
            )
        )

        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(weights[:, D:], 0.0)
        for copy in range(1, COPIES):
            np.testing.assert_allclose(weights[copy], weights[0], rtol=0.0, atol=0.0)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=2 fixture absent")
    def test_gelu_chunks_pack_and_restore_bsgs_copies(self) -> None:
        hidden = _load_array("oracle.mlp_fc")
        expected = _load_array("oracle.gelu_tanh")

        def gelu(value: np.ndarray) -> np.ndarray:
            return (
                0.5
                * value
                * (
                    1.0
                    + np.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3))
                )
            )

        for token in range(T):
            packed = np.zeros((COPIES, PACK_WIDTH), dtype=np.float64)
            for chunk in range(COPIES):
                packed[chunk, :D] = hidden[token, chunk * D : (chunk + 1) * D]

            activated = _active_transform(packed, gelu)
            restored_chunks = []
            for chunk in range(COPIES):
                mask = np.zeros((COPIES, PACK_WIDTH), dtype=np.float64)
                mask[chunk, :D] = 1.0
                selected = activated * mask.reshape(SLOTS)
                restored = _replicate_copies(selected).reshape(COPIES, PACK_WIDTH)
                for copy in range(1, COPIES):
                    np.testing.assert_allclose(
                        restored[copy], restored[0], rtol=0.0, atol=0.0
                    )
                np.testing.assert_array_equal(restored[:, D:], 0.0)
                restored_chunks.append(restored[0, :D])

            got = np.concatenate(restored_chunks)
            np.testing.assert_allclose(got, expected[token], rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
