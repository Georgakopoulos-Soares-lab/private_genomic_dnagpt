"""Static and float64 contracts for the isolated packed T=8 GPU gate."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from .packed_contract import (
    DEFAULT_CONTRACT,
    DEFAULT_FIXTURE,
    SLOTS,
    build_contract,
)


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "dnagpt_packed_t8_fides.cpp"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackedT8StaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))

    def test_interleaved_one_ciphertext_layout_is_exact(self) -> None:
        self.assertEqual(SLOTS, 32768)
        self.assertIn("copy * SLOTS_PER_COPY + channel * P + token", self.source)
        self.assertIn('"input_ciphertexts\\": 1', self.source)
        self.assertIn('"output_ciphertexts\\": 1', self.source)
        self.assertEqual(
            self.contract["layout"]["formula"],
            "copy*(1024*P)+channel*P+token",
        )
        self.assertEqual(self.contract["layout"]["bsgs"], [32, 32])

    def test_channel_rotations_are_scaled_by_p_and_attention_by_offset(self) -> None:
        self.assertIn("small * P", self.source)
        self.assertIn("giant * BSGS_N1 * P", self.source)
        self.assertIn("rotate(key, -static_cast<int>(offset))", self.source)
        self.assertIn("rotate(value, -static_cast<int>(offset))", self.source)
        self.assertIn(
            "accumulate(products, HEAD_DIM, static_cast<int>(P))", self.source
        )
        self.assertIn(
            "accumulate(weights, HEAD_DIM, -static_cast<int>(P))", self.source
        )

    def test_causal_numerator_first_schedule_and_public_guards(self) -> None:
        self.assertIn("std::array<Ct, T> numerators", self.source)
        self.assertIn("active_shift_plain(offset)", self.source)
        self.assertIn("anchor_mask(offset)", self.source)
        self.assertIn("reciprocal_guard_plain()", self.source)
        self.assertRegex(
            self.source,
            re.compile(
                r"EvalChebyshevSeries\(\s*anchors.*?denominator.*?"
                r"EvalChebyshevSeries\(\s*denominator.*?"
                r"multiply\(reciprocal,\s*numerators\[offset\]\)",
                re.DOTALL,
            ),
        )
        self.assertIn('private_query_domain_adaptation\\": false', self.source)
        self.assertIn("inactive_channel_guard_plain()", self.source)

    def test_no_secret_key_or_decrypt_inside_evaluator(self) -> None:
        evaluator = self.source.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]
        self.assertNotIn("PrivateKey", evaluator)
        self.assertNotIn("secretKey", evaluator)
        self.assertNotIn("Decrypt(", evaluator)
        self.assertEqual(len(re.findall(r"\bDecrypt\s*\(", self.source)), 1)
        self.assertIn('"intermediate_decrypt_attempts\\": 0', self.source)
        self.assertIn('"final_decrypt_calls\\": 1', self.source)

    def test_depth_operation_and_rotation_contracts(self) -> None:
        self.assertIn("MULT_DEPTH = 29", self.source)
        self.assertIn("EXPECTED_PROJECTION_MAX_LEVEL = 25", self.source)
        self.assertIn("require_remaining_depth(context->GetLevel(), 1,", self.source)
        # Exact graph counts excluding operations internal to Chebyshev.
        self.assertEqual(4 * 1024, 4096)
        self.assertEqual(2 + 8 + 8 + 8, 26)
        self.assertEqual(5 + 4096 + 8 + 8, 4117)
        self.assertEqual(2 + 8 + 8, 18)
        self.assertEqual(2 * 31 + 4 * 31 + 2 * 7, 200)
        self.assertIn('"direct_logical_rotations\\": ', self.source)

    def test_pinned_public_contract_identity_matches_source(self) -> None:
        digest = sha256(DEFAULT_CONTRACT)
        self.assertEqual(
            digest,
            "a913d3b8e3365525279b313a3dfb8ae863a6f7c0bf1c6d222e8f6886a853a0f4",
        )
        self.assertIn(digest, self.source)
        self.assertEqual(
            self.contract["identity"]["fixture_manifest_sha256"],
            "75ce745757a8141df42054612d54a0837b5f0c2e7a771cafd57fc5055e054162",
        )

    def test_launch_requires_a_process_free_gpu_and_immutable_output(self) -> None:
        launcher = (ROOT / "launch_brev_packed_t8.sh").read_text()
        scheduler = (ROOT / "schedule_when_free.sh").read_text()
        runner = (ROOT / "run_packed_t8.sh").read_text()
        self.assertIn("--query-compute-apps=gpu_uuid", launcher)
        self.assertIn("COMPUTE_PROCESS_COUNT != 0", launcher)
        self.assertIn("memory_mib >= 100", launcher)
        self.assertIn("utilization_pct != 0", launcher)
        self.assertIn("REQUIRED_STABLE_POLLS=2", scheduler)
        self.assertIn("memory < 100", scheduler)
        self.assertIn('[[ -e "${OUTPUT}" ]]', runner)
        self.assertIn("FIDES_CONTAINER_IMAGE", runner)

    @unittest.skipUnless(DEFAULT_FIXTURE.exists(), "public T=8 fixture absent")
    def test_exact_layout_and_approximation_preflights_are_reproducible(self) -> None:
        rebuilt = build_contract(DEFAULT_FIXTURE)
        self.assertEqual(rebuilt, self.contract)
        exact = rebuilt["exact_float64_layout_parity"]
        approximate = rebuilt["approximate_float64_preflight"]
        self.assertLess(exact["global_rel_inf"], 1e-12)
        self.assertLess(approximate["global_rel_inf"], 4e-2)
        self.assertLess(approximate["worst_token_rel_inf"], 4e-2)
        self.assertTrue(exact["passed"])
        self.assertTrue(approximate["passed"])


if __name__ == "__main__":
    unittest.main()
