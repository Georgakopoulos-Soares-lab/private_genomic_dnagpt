"""Contracts for the T=2 Scheme B block-0 -> block-1 refresh gate.

This is a pre-GPU contract. It proves the frozen single-block source remains
byte-identical, validates every fixture file the two-block driver consumes,
checks the full-hidden-state pack/unpack/replicate mapping, and independently
recomputes released block 1 from the refreshed block-0 oracle.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from fhe.realweights.manual import block_reference


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = ROOT / "src" / "real_dnagpt_fides_scheme_b_two_block_refresh.cpp"
FROZEN_SOURCE = ROOT / "src" / "real_dnagpt_fides_scheme_b.cpp"
FROZEN_SOURCE_SHA256 = (
    "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
)
FIXTURE_CONTRACT = ROOT / "fixture_two_block_t2.sha256"
RUN_SCRIPT = ROOT / "run_scheme_b_two_block_refresh.sh"
LAUNCH_SCRIPT = ROOT / "launch_brev_scheme_b_two_block_refresh.sh"
WAIT_SCRIPT = ROOT / "wait_and_run_scheme_b_two_block_refresh.sh"
CMAKE = ROOT / "CMakeLists.txt"
BUILD_SCRIPT = ROOT / "build_in_fideslib.sh"
FIXTURE = (
    REPO
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0"
)
FIXTURE_MANIFEST_SHA256 = (
    "3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c"
)

D = 768
T = 2
PACK_WIDTH = 1024
COPIES = 4
SLOTS = PACK_WIDTH * COPIES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_entries() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in FIXTURE_CONTRACT.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        entries[filename] = digest
    return entries


def _load_array(name: str) -> np.ndarray:
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    metadata = manifest["arrays"][name]
    path = FIXTURE / metadata["file"]
    if _sha256(path) != metadata["sha256"]:
        raise ValueError(f"fixture hash mismatch for {name}")
    return np.fromfile(path, dtype="<f8").reshape(metadata["shape"])


def _block_weights(block: int) -> dict[str, np.ndarray]:
    return {
        field: _load_array(f"weights.block{block}.{field}")
        for field in (
            "ln1",
            "attn_qkv",
            "attn_proj",
            "ln2",
            "mlp_fc",
            "mlp_proj",
        )
    }


def _pack_block_output(tokens: np.ndarray) -> np.ndarray:
    """Model frozen finish(): token 0 -> copy 0, token 1 -> copy 1."""
    packed = np.zeros(SLOTS, dtype=np.float64)
    for token in range(T):
        packed[token * PACK_WIDTH : token * PACK_WIDTH + D] = tokens[token]
    return packed


def _refresh_packed_output(packed: np.ndarray) -> np.ndarray:
    """Model the client boundary's unpack and per-token four-copy refresh."""
    refreshed = np.zeros((T, COPIES, PACK_WIDTH), dtype=np.float64)
    for token in range(T):
        values = packed[token * PACK_WIDTH : token * PACK_WIDTH + D]
        refreshed[token, :, :D] = values
    return refreshed


class TwoBlockRefreshSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.frozen = FROZEN_SOURCE.read_text(encoding="utf-8")

    def test_frozen_single_block_anchor_is_byte_identical(self) -> None:
        self.assertEqual(_sha256(FROZEN_SOURCE), FROZEN_SOURCE_SHA256)
        self.assertIn(FROZEN_SOURCE_SHA256, self.source)
        self.assertIn('#include "real_dnagpt_fides_scheme_b.cpp"', self.source)

    def test_driver_reuses_same_evaluator_for_both_released_blocks(self) -> None:
        self.assertEqual(self.source.count("EncryptedEvaluator block"), 2)
        self.assertIn(
            'block0_evaluator.evaluate(encrypted_inputs, "full")',
            self.source,
        )
        self.assertIn(
            'block1_evaluator.evaluate(refresh.refreshed, "full")',
            self.source,
        )
        self.assertIn("load_multiblock_fixture(options.fixture_dir, 0)", self.source)
        self.assertIn("load_multiblock_fixture(options.fixture_dir, 1)", self.source)

    def test_full_hidden_refresh_is_explicit_and_fail_closed(self) -> None:
        self.assertIn("refresh_full_hidden_state(", self.source)
        self.assertIn("Ct local = packed_block_output;", self.source)
        self.assertIn(
            "cc->Decrypt(keys.secretKey, local, &decoded)",
            self.source,
        )
        self.assertIn("result.refreshed = encrypt_token_states", self.source)
        self.assertIn(
            "fresh token-state encryption did not return level 0", self.source
        )
        self.assertIn("EXPECTED_ROUND_TRIPS_PER_BLOCK = 7", self.source)
        self.assertIn("EXPECTED_LOGICAL_BOUNDARIES_PER_BLOCK = 24", self.source)
        self.assertIn(
            "block 1 crossed an unexpected Scheme B boundary count", self.source
        )
        self.assertIn('"block_boundary_count\\": 1', self.source)
        self.assertIn('"block_boundary_decrypt_calls\\": 1', self.source)
        self.assertIn('"block_boundary_encrypt_calls\\": "', self.source)
        self.assertIn('"block1_input_level\\": 0', self.source)

    def test_protocol_has_no_bootstrap_or_deep_context(self) -> None:
        driver = self.source.split("int main(int argc, char** argv)", 1)[1]
        self.assertNotIn("EvalBootstrap", driver)
        self.assertNotIn("Enable(FHE)", driver)
        self.assertIn("parameters.SetMultiplicativeDepth(MULT_DEPTH)", driver)
        self.assertIn('"bootstraps\\": 0', self.source)

    def test_only_declared_refresh_and_final_measurement_decrypt_in_driver(
        self,
    ) -> None:
        self.assertEqual(self.source.count("cc->Decrypt("), 2)
        evaluator = self.frozen.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]
        self.assertNotIn("Decrypt(", evaluator)
        self.assertNotIn("secretKey", evaluator)

    def test_output_is_immutable_and_provenance_is_required(self) -> None:
        self.assertIn("refusing to overwrite immutable evidence", self.source)
        self.assertIn(
            "source and fixture contract identities are required", self.source
        )
        self.assertIn("frozen_single_block_source_sha256", self.source)
        self.assertIn(FIXTURE_MANIFEST_SHA256, self.source)

    def test_build_and_run_gates_reference_only_the_new_prototype(self) -> None:
        target = "real_dnagpt_fides_scheme_b_two_block_refresh"
        self.assertIn(target, CMAKE.read_text(encoding="utf-8"))
        self.assertIn(target, BUILD_SCRIPT.read_text(encoding="utf-8"))
        run_script = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(target, run_script)
        self.assertIn("_scheme_b_", run_script)
        self.assertIn("_two_block_refresh_", run_script)
        self.assertIn(FROZEN_SOURCE_SHA256, run_script)
        self.assertIn("sha256sum --check --status", run_script)
        launch_script = LAUNCH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_scheme_b_", launch_script)
        self.assertIn("_two_block_refresh_", launch_script)
        self.assertIn("gpucap_preflight_confirm_gpu", launch_script)
        self.assertIn("nohup sh -c", launch_script)
        self.assertIn("run_scheme_b_two_block_refresh.sh", launch_script)
        wait_script = WAIT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("gpucap_wait_for_capacity", wait_script)
        self.assertIn("launch_brev_scheme_b_two_block_refresh.sh", wait_script)
        self.assertIn("refusing to reuse orchestrator log", wait_script)
        self.assertIn("manual investigation required", wait_script)


@unittest.skipUnless(FIXTURE.exists(), "multiblock T=2 fixture absent")
class TwoBlockRefreshFixtureContractTests(unittest.TestCase):
    def test_every_consumed_fixture_file_matches_hash_contract(self) -> None:
        entries = _fixture_entries()
        self.assertEqual(len(entries), 16)
        for filename, digest in entries.items():
            self.assertEqual(_sha256(FIXTURE / filename), digest, filename)
        self.assertEqual(_sha256(FIXTURE / "manifest.json"), FIXTURE_MANIFEST_SHA256)

    def test_manifest_binds_released_two_block_t2_lineage(self) -> None:
        manifest = json.loads((FIXTURE / "manifest.json").read_text())
        self.assertEqual(manifest["schema"], "dnagpt.multiblock.gsr_t2.v1")
        self.assertEqual(manifest["model"]["layers"], 12)
        self.assertEqual(manifest["tokenization"]["exported_token_count_T"], T)
        self.assertEqual(manifest["tokenization"]["full_prompt_token_count"], 103)
        self.assertTrue(manifest["oracle_gate"]["passed"])

    def test_block0_output_is_exactly_block1_plaintext_input(self) -> None:
        block0 = _load_array("oracle.block0.block_output")
        block1_input = _load_array("oracle.block1.input_embeddings")
        np.testing.assert_array_equal(block0, block1_input)

    def test_pack_unpack_refresh_covers_every_value_once(self) -> None:
        block0 = _load_array("oracle.block0.block_output")
        packed = _pack_block_output(block0)
        refreshed = _refresh_packed_output(packed)
        for token in range(T):
            for copy in range(COPIES):
                np.testing.assert_array_equal(refreshed[token, copy, :D], block0[token])
                self.assertTrue(np.all(refreshed[token, copy, D:] == 0.0))

    def test_refreshed_block0_oracle_recomputes_released_block1(self) -> None:
        refreshed = _load_array("oracle.block0.block_output").copy()
        actual = block_reference(
            refreshed,
            _block_weights(1),
            num_heads=12,
            eps=1e-5,
        )["block_output"]
        expected = _load_array("oracle.block1.block_output")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
