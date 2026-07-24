"""Fail-closed static tests for the two-block encrypted lineage."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from .validate_contract import (
    DEFAULT_FIXTURE,
    DEFAULT_RANGE_CONTROL,
    RANGE_CONTROL_SHA256,
    validate,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "dnagpt_two_block_fides.cpp"


class TwoBlockStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_single_context_bootstrap_api_is_composed_without_readout(self) -> None:
        self.assertIn(
            "cc_->EvalBootstrap(conditioned[token], 1, 0, false)", self.source
        )
        self.assertIn("cc->EvalBootstrapSetup(", self.source)
        self.assertIn("cc->EvalBootstrapKeyGen(keys.secretKey, SLOTS)", self.source)
        self.assertIn("cc->Enable(FHE)", self.source)
        self.assertEqual(self.source.count("GenCryptoContext(parameters)"), 1)

    def test_two_released_blocks_share_one_ciphertext_lineage(self) -> None:
        self.assertIn("block(encrypted_inputs, 0)", self.source)
        self.assertIn("block(restored, 1)", self.source)
        self.assertIn("fixture_.blocks[block_index]", self.source)
        self.assertIn("oracle__block1__block_output.bin", self.source)
        self.assertIn("same_crypto_context_and_key_lineage", self.source)
        self.assertIn("client_round_trips", self.source)

    def test_exact_t2_sigmoid_schedule_is_used_in_both_blocks(self) -> None:
        self.assertIn("BLOCK_CONTRACTS[block_index]", self.source)
        self.assertIn("multiply(weight_1, value_delta)", self.source)
        self.assertRegex(
            self.source,
            re.compile(
                r"weight_1\s*=\s*cc_->EvalChebyshevSeries\(.*?"
                r"multiply\(weight_1,\s*value_delta\)",
                re.DOTALL,
            ),
        )
        for obsolete in ("DEG_EXP", "DEG_RECIPROCAL", "exp_coefficients_"):
            self.assertNotIn(obsolete, self.source)

    def test_refresh_condition_and_restore_are_public_and_fixed(self) -> None:
        self.assertIn("BLOCK0_REFRESH_BOUND = 20.0", self.source)
        self.assertIn("1.0 / BLOCK0_REFRESH_BOUND", self.source)
        self.assertIn("std::vector<double>(D, BLOCK0_REFRESH_BOUND)", self.source)
        self.assertIn("bootstrap_boundaries = 1", self.source)
        self.assertIn("bootstrap_primitive_calls = T", self.source)

    def test_evaluator_has_no_secret_key_or_decrypt(self) -> None:
        evaluator = self.source.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]
        self.assertNotIn("secretKey", evaluator)
        self.assertNotIn("PrivateKey", evaluator)
        self.assertNotIn("Decrypt(", evaluator)
        self.assertEqual(len(re.findall(r"\bDecrypt\s*\(", self.source)), 1)

    def test_depth_and_public_range_contracts_are_pinned(self) -> None:
        self.assertIn("MULT_DEPTH = 64", self.source)
        self.assertIn("SCALE_BITS = 59", self.source)
        self.assertIn(RANGE_CONTROL_SHA256, self.source)
        self.assertIn("block_index == 0 ? 8 : 10", self.source)
        self.assertIn("maximum_level(restored), 35", self.source)
        self.assertIn("refresh_output >= refresh_conditioned", self.source)
        self.assertIn("two-block final output packing", self.source)
        self.assertIn(
            "8dfa77fa298472824a86414efdc6a5d14c4fa57bce3447b61b9893fe37d547a4",
            self.source,
        )

    @unittest.skipUnless(DEFAULT_FIXTURE.exists(), "multiblock fixture absent")
    def test_consumed_fixture_and_range_control_contracts(self) -> None:
        result = validate(DEFAULT_FIXTURE, DEFAULT_RANGE_CONTROL)
        self.assertTrue(result["passed"])
        self.assertEqual(result["blocks"], [0, 1])
        self.assertEqual(result["consumed_fixture_files"], 15)


if __name__ == "__main__":
    unittest.main()
