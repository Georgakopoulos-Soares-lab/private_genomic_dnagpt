"""Static safety/optimization contract for the versioned sigmoid GPU graph."""

from __future__ import annotations

import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "real_dnagpt_fides_sigmoid.cpp"
FROZEN_FIXTURE = ROOT.parent / "gpu_real" / "fixture.sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SigmoidGPUStaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_fixture_contract_is_exactly_reused(self) -> None:
        self.assertEqual(sha256(ROOT / "fixture.sha256"), sha256(FROZEN_FIXTURE))

    def test_t2_sigmoid_identity_replaces_two_series_schedule(self) -> None:
        self.assertIn("DEG_SIGMOID = 13", self.source)
        self.assertIn("sigmoid_coefficients_", self.source)
        self.assertIn("const Ct value_delta", self.source)
        self.assertIn("multiply(weight_1, value_delta)", self.source)
        self.assertIn("EvalAdd(value_heads[head][0], weighted_delta)", self.source)
        for obsolete in (
            "DEG_EXP",
            "DEG_RECIPROCAL",
            "ATTN_DENOM",
            "exp_coefficients_",
            "reciprocal_coefficients_",
        ):
            self.assertNotIn(obsolete, self.source)

    def test_deeper_sigmoid_ciphertext_is_first_multiply_operand(self) -> None:
        self.assertRegex(
            self.source,
            re.compile(
                r"weight_1\s*=\s*cc_->EvalChebyshevSeries\(.*?"
                r"multiply\(weight_1,\s*value_delta\)",
                re.DOTALL,
            ),
        )

    def test_remaining_depth_guards_fail_closed(self) -> None:
        self.assertIn(
            'normalized2[token]->GetLevel(), 8,\n                "MLP token "',
            self.source,
        )
        self.assertIn(
            "require_remaining_depth(maximum_level(block_output), 1,",
            self.source,
        )
        self.assertIn("levels_needed > MULT_DEPTH - current_level", self.source)

    def test_predicted_level_contract_and_actual_trace_are_serialized(self) -> None:
        for name, level in (
            ("EXPECTED_CONTEXT_MAX_LEVEL", 20),
            ("EXPECTED_PROJECTION_MAX_LEVEL", 21),
            ("EXPECTED_LN2_MAX_LEVEL", 32),
            ("EXPECTED_BLOCK_MAX_LEVEL", 40),
            ("EXPECTED_PACKED_MAX_LEVEL", 41),
        ):
            self.assertIn(f"{name} = {level}", self.source)
        self.assertIn("predicted_level_maxima", self.source)
        self.assertIn("level_trace", self.source)
        self.assertIn("operation_counts", self.source)

    def test_evaluator_has_no_key_or_decrypt_and_main_decrypts_once(self) -> None:
        evaluator = self.source.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]
        self.assertNotIn("PrivateKey", evaluator)
        self.assertNotIn("secretKey", evaluator)
        self.assertNotIn("Decrypt(", evaluator)
        self.assertEqual(self.source.count("cc->Decrypt("), 1)
        self.assertIn("intermediate_decrypt_attempts", self.source)
        self.assertIn("final_decrypt_calls", self.source)
        self.assertIn("evaluator_has_private_key", self.source)

    def test_build_and_run_paths_are_versioned(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        runner = (ROOT / "run_sigmoid.sh").read_text(encoding="utf-8")
        launcher = (ROOT / "launch_brev_sigmoid.sh").read_text(encoding="utf-8")
        scheduler = (ROOT / "schedule_when_free.sh").read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_sigmoid", cmake)
        self.assertIn("real_dnagpt_fides_sigmoid", runner)
        self.assertIn("FIDES_REAL_SIGMOID_BUILD_DIR", runner)
        self.assertIn('"_sigmoid13_"', runner)
        self.assertIn("run_sigmoid.sh", launcher)
        self.assertIn("FIDES_REAL_SIGMOID_BUILD_DIR", launcher)
        self.assertIn('"_sigmoid13_"', launcher)
        self.assertIn("--query-compute-apps=gpu_uuid", launcher)
        self.assertIn("COMPUTE_PROCESS_COUNT != 0", launcher)
        self.assertIn("--query-compute-apps=gpu_uuid", scheduler)
        self.assertIn("REQUIRED_STABLE_POLLS=2", scheduler)
        self.assertIn("memory_mib < 100", scheduler)
        self.assertIn("/tmp/dnagpt-fhe-gpu-${gpu}.lock", scheduler)
        self.assertIn("trap cleanup_lock EXIT", scheduler)
        self.assertIn("launch_brev_sigmoid.sh", scheduler)
        self.assertIn("t2-sigmoid13-v1", self.source)


if __name__ == "__main__":
    unittest.main()
