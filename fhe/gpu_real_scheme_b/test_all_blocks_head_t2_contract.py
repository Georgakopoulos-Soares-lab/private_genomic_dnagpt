"""Pre-GPU contracts for Scheme B's complete 12-block + GSR-head graph at T=2.

This file is intentionally CPU/NumPy-only. It proves the released fixture's
eleven inter-block boundaries and block-11-to-head boundary, independently
recomputes every block and the encrypted N-minus-A readout, and freezes the
expected Scheme B boundary arithmetic before any C++/GPU implementation.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from fhe.multiblock.manual import layer_norm, silu
from fhe.realweights.manual import block_reference


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = ROOT / "src" / "real_dnagpt_fides_scheme_b_all_blocks_head_t2.cpp"
FROZEN_SOURCE = ROOT / "src" / "real_dnagpt_fides_scheme_b.cpp"
FROZEN_SOURCE_SHA256 = (
    "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
)
FIXTURE_CONTRACT = ROOT / "fixture_all_blocks_head_t2.sha256"
FIXTURE_CONTRACT_SHA256 = (
    "c72dd1a897cf2dc66768a1bbd077b40fb4e81c7c72247e811cdcb71465b488de"
)
RUN_SCRIPT = ROOT / "run_scheme_b_all_blocks_head_t2.sh"
CMAKE = ROOT / "CMakeLists.txt"
BUILD_SCRIPT = ROOT / "build_in_fideslib.sh"
FIXTURE = (
    REPO
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0"
)
MANIFEST = FIXTURE / "manifest.json"
MANIFEST_SHA256 = "3c8d7bbcbfe62d9dc7f3fa89571396b93c3f92cb89563fa5cb4cd7b831bd5e4c"

D = 768
T = 2
BLOCKS = 12
NONLINEARITY_ROUND_TRIPS_PER_BLOCK = 7
LOGICAL_NONLINEARITY_INSTANCES_PER_BLOCK = 24
INTER_BLOCK_FULL_HIDDEN_REFRESHES = BLOCKS - 1
BLOCK11_TO_HEAD_LAST_TOKEN_REFRESHES = 1
HEAD_NONLINEARITY_ROUND_TRIPS = 3
EXPECTED_DECLARED_CROSSINGS = (
    BLOCKS * NONLINEARITY_ROUND_TRIPS_PER_BLOCK
    + INTER_BLOCK_FULL_HIDDEN_REFRESHES
    + BLOCK11_TO_HEAD_LAST_TOKEN_REFRESHES
    + HEAD_NONLINEARITY_ROUND_TRIPS
)
EXPECTED_LOGICAL_INSTANCES = (
    BLOCKS * LOGICAL_NONLINEARITY_INSTANCES_PER_BLOCK
    + INTER_BLOCK_FULL_HIDDEN_REFRESHES * T
    + BLOCK11_TO_HEAD_LAST_TOKEN_REFRESHES
    + HEAD_NONLINEARITY_ROUND_TRIPS
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AllBlocksHeadT2SourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_evaluator_and_fixture_contract_are_pinned(self) -> None:
        self.assertEqual(sha256(FROZEN_SOURCE), FROZEN_SOURCE_SHA256)
        self.assertEqual(sha256(FIXTURE_CONTRACT), FIXTURE_CONTRACT_SHA256)
        self.assertIn(FROZEN_SOURCE_SHA256, self.source)
        self.assertIn(MANIFEST_SHA256, self.source)
        self.assertIn('#include "real_dnagpt_fides_scheme_b.cpp"', self.source)

    def test_one_context_and_evaluator_lineage_runs_all_twelve_blocks(self) -> None:
        self.assertEqual(self.source.count("GenCryptoContext(parameters)"), 1)
        self.assertEqual(self.source.count("SetDevices({options.gpu})"), 1)
        self.assertIn(
            "for (std::size_t block = 0; block < ALL_BLOCKS; ++block)",
            self.source,
        )
        self.assertIn("EncryptedEvaluator evaluator(cc, fixture, client)", self.source)
        self.assertIn('evaluator.evaluate(encrypted_inputs, "full")', self.source)
        self.assertIn(
            "EncryptedHeadEvaluator head_evaluator(\n"
            "            cc, head_fixture, client)",
            self.source,
        )

    def test_refreshes_are_explicit_and_runtime_counts_fail_closed(self) -> None:
        self.assertIn("EXPECTED_INTER_BLOCK_REFRESHES = ALL_BLOCKS - 1", self.source)
        self.assertIn("EXPECTED_HEAD_INPUT_REFRESHES = 1", self.source)
        self.assertIn("static_assert(EXPECTED_TOTAL_CROSSINGS == 99)", self.source)
        self.assertIn(
            "static_assert(EXPECTED_TOTAL_LOGICAL_INSTANCES == 314)", self.source
        )
        self.assertIn("refresh_all_hidden(", self.source)
        self.assertIn("refresh_last_token_for_head(", self.source)
        self.assertIn(
            "fresh token-state encryption did not return level 0", self.source
        )
        self.assertIn("head-input refresh did not return level 0", self.source)
        self.assertIn('"complete graph crossed an unexpected total "', self.source)
        self.assertIn('"Scheme B boundary count"', self.source)

    def test_head_schedule_stays_encrypted_between_declared_boundaries(self) -> None:
        head = self.source.split("class EncryptedHeadEvaluator", 1)[1].split(
            "struct BlockSummary", 1
        )[0]
        self.assertNotIn("Decrypt(", head)
        self.assertNotIn("secretKey", head)
        self.assertIn("layernorm(last_token", head)
        self.assertIn("matmul(baby_rotations(final_ln)", head)
        self.assertIn('"head_silu"', head)
        self.assertIn('"head_ln"', head)
        self.assertIn("fixture_.margin_n_minus_a", head)
        self.assertIn("sum_broadcast(weighted)", head)
        self.assertIn("stable_exact_silu", head)
        self.assertIn("value >= 0.0", self.source)

    def test_only_declared_refresh_and_final_decrypt_call_sites_exist(self) -> None:
        # decrypt_packed_hidden is reused for all 12 declared refreshes; the
        # second call site is the single final scalar task-output decrypt.
        self.assertEqual(self.source.count("cc->Decrypt("), 2)
        self.assertIn("decrypt_packed_hidden(", self.source)
        self.assertIn("decoded_margin", self.source)
        self.assertIn('"intermediate_decrypt_attempts\\": 0', self.source)
        self.assertIn('"evaluator_has_private_key\\": false', self.source)

    def test_direct_dot_oracle_and_label_gate_are_explicit(self) -> None:
        self.assertIn("direct_dot_oracle_margin(", self.source)
        self.assertIn("std::signbit(decrypted_margin)", self.source)
        self.assertIn("margin_rel_error <= TOL && label_matches", self.source)
        self.assertIn('"direct_dot_oracle_margin_n_minus_a\\": "', self.source)

    def test_build_run_and_immutability_gates_are_local_and_isolated(self) -> None:
        target = "real_dnagpt_fides_scheme_b_all_blocks_head_t2"
        self.assertIn(target, CMAKE.read_text(encoding="utf-8"))
        self.assertIn(target, BUILD_SCRIPT.read_text(encoding="utf-8"))
        run = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(target, run)
        self.assertIn(FROZEN_SOURCE_SHA256, run)
        self.assertIn(FIXTURE_CONTRACT_SHA256, run)
        self.assertIn("sha256sum --check --status", run)
        self.assertIn("refusing to overwrite immutable evidence", run)
        self.assertIn("_all_blocks_head_t2_", run)
        self.assertNotIn("brev ", run)
        self.assertNotIn("docker ", run)


@unittest.skipUnless(FIXTURE.exists(), "multiblock T=2 fixture absent")
class AllBlocksHeadT2ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def load(self, name: str) -> np.ndarray:
        item = self.manifest["arrays"][name]
        path = FIXTURE / item["file"]
        self.assertEqual(sha256(path), item["sha256"], name)
        return np.fromfile(path, dtype="<f8").reshape(item["shape"])

    def block_weights(self, block: int) -> dict[str, np.ndarray]:
        return {
            field: self.load(f"weights.block{block}.{field}")
            for field in (
                "ln1",
                "attn_qkv",
                "attn_proj",
                "ln2",
                "mlp_fc",
                "mlp_proj",
            )
        }

    def test_manifest_binds_complete_released_graph(self) -> None:
        self.assertEqual(sha256(MANIFEST), MANIFEST_SHA256)
        self.assertEqual(self.manifest["schema"], "dnagpt.multiblock.gsr_t2.v1")
        self.assertEqual(self.manifest["model"]["layers"], BLOCKS)
        self.assertEqual(self.manifest["model"]["embedding_dim"], D)
        self.assertEqual(self.manifest["tokenization"]["exported_token_count_T"], T)
        self.assertEqual(self.manifest["tokenization"]["full_prompt_token_count"], 103)
        self.assertTrue(self.manifest["oracle_gate"]["passed"])
        self.assertEqual(
            self.manifest["classification"]["encrypted_readout"],
            "last token -> final LN -> head linear -> SiLU -> head LN "
            "-> dot(head_readout_N - head_readout_A); sign selects N/A",
        )

    def test_every_consumed_file_matches_the_immutable_hash_contract(self) -> None:
        entries: dict[str, str] = {}
        for line in FIXTURE_CONTRACT.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            entries[filename] = digest
        self.assertEqual(len(entries), 90)
        for filename, digest in entries.items():
            self.assertEqual(sha256(FIXTURE / filename), digest, filename)

    def test_every_inter_block_boundary_is_byte_exact(self) -> None:
        for block in range(BLOCKS - 1):
            output = self.load(f"oracle.block{block}.block_output")
            next_input = self.load(f"oracle.block{block + 1}.input_embeddings")
            np.testing.assert_array_equal(output, next_input)

    def test_all_twelve_blocks_recompute_through_the_refresh_lineage(self) -> None:
        hidden = self.load("input.embeddings")
        for block in range(BLOCKS):
            expected_input = self.load(f"oracle.block{block}.input_embeddings")
            np.testing.assert_allclose(hidden, expected_input, rtol=0, atol=1e-12)
            actual = block_reference(
                hidden,
                self.block_weights(block),
                num_heads=12,
                eps=1e-5,
            )["block_output"]
            expected = self.load(f"oracle.block{block}.block_output")
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            # Model the declared client refresh: the exact recovered T x D state
            # becomes fresh level-0 input for the next block.
            hidden = actual.copy()

    def test_classifier_head_and_encrypted_margin_recompute(self) -> None:
        hidden = self.load("oracle.block11.block_output")
        final_ln, *_ = layer_norm(hidden, self.load("weights.final_ln"))
        np.testing.assert_allclose(
            final_ln,
            self.load("oracle.head.final_ln_output"),
            rtol=0,
            atol=1e-12,
        )

        head_linear = final_ln @ self.load("weights.head_linear").T
        np.testing.assert_allclose(
            head_linear,
            self.load("oracle.head.head_linear"),
            rtol=0,
            atol=1e-12,
        )

        head_silu = silu(head_linear)
        np.testing.assert_allclose(
            head_silu,
            self.load("oracle.head.head_silu"),
            rtol=0,
            atol=1e-12,
        )

        head_ln, *_ = layer_norm(head_silu, self.load("weights.head_ln"))
        oracle_head_ln = self.load("oracle.head.head_ln_output")
        np.testing.assert_allclose(
            head_ln,
            oracle_head_ln,
            rtol=0,
            atol=1e-12,
        )

        margin_row = self.load("weights.head_readout_margin_n_minus_a")
        margin = float(head_ln[-1] @ margin_row)
        direct_dot_oracle = float(oracle_head_ln[-1] @ margin_row)
        contract = self.manifest["classification"]["truncated_t2_graph_gate"]
        self.assertAlmostEqual(margin, direct_dot_oracle, places=12)
        # The manifest field is produced as N_logit - A_logit, while the
        # encrypted graph evaluates one dot against the pre-derived N-A row.
        # They are algebraically identical but differ slightly in float64
        # summation order. Correctness requires the direct-dot oracle and the
        # same classification sign, not bit equality between both schedules.
        self.assertLess(abs(margin - contract["margin_n_minus_a"]), 1e-7)
        self.assertEqual(np.signbit(margin), np.signbit(contract["margin_n_minus_a"]))
        self.assertLess(margin, 0.0)
        self.assertEqual(contract["binary_label"], "A")
        self.assertIn("graph gate", contract["evidence"])

    def test_boundary_arithmetic_is_frozen_before_cpp(self) -> None:
        self.assertEqual(EXPECTED_DECLARED_CROSSINGS, 99)
        self.assertEqual(EXPECTED_LOGICAL_INSTANCES, 314)
        self.assertEqual(INTER_BLOCK_FULL_HIDDEN_REFRESHES, 11)
        self.assertEqual(BLOCK11_TO_HEAD_LAST_TOKEN_REFRESHES, 1)
        self.assertEqual(HEAD_NONLINEARITY_ROUND_TRIPS, 3)


if __name__ == "__main__":
    unittest.main()
