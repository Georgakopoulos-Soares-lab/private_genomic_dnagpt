import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(name):
    return json.loads((RUNS / name).read_text())


class FHEGPUEvidenceTests(unittest.TestCase):
    def test_asymmetric_chebyshev_patch_closes_measured_failure(self):
        failed = load_run("fhe_gpu_asymmetric_chebyshev_prepatch_20260724.json")
        passed = load_run("fhe_gpu_asymmetric_chebyshev_asymfix2_20260724.json")
        source = ROOT / "fhe" / "gpu" / "probes" / "asymmetric_cheb.cpp"
        cmake = ROOT / "fhe" / "gpu" / "probes" / "CMakeLists.txt"
        dockerfile = ROOT / "docker" / "Dockerfile.fideslib"
        patch = ROOT / "docker" / "patches" / "fideslib-asymmetric-chebyshev.patch"

        self.assertFalse(failed["metrics"]["passed"])
        self.assertGreater(failed["metrics"]["rel_inf"], failed["metrics"]["tol"])
        self.assertTrue(passed["metrics"]["passed"])
        self.assertLessEqual(passed["metrics"]["rel_inf"], passed["metrics"]["tol"])
        self.assertLess(
            passed["metrics"]["rel_inf"], failed["metrics"]["rel_inf"] / 1000
        )
        self.assertEqual(passed["provenance"]["source_sha256"], sha256(source))
        self.assertEqual(passed["provenance"]["cmake_sha256"], sha256(cmake))
        self.assertEqual(passed["provenance"]["dockerfile_sha256"], sha256(dockerfile))
        self.assertEqual(passed["provenance"]["patch_sha256"], sha256(patch))

    def test_complete_gpu_toy_block_is_bound_to_source(self):
        result = load_run("fhe_fides_toy_a100_asymfix2_20260724.json")
        source = ROOT / "fhe" / "gpu" / "src" / "toy_dnagpt_fides.cpp"
        fixture = ROOT / "fhe" / "gpu" / "src" / "toy_fixture.hpp"

        self.assertTrue(result["passed"])
        self.assertLessEqual(result["global_rel_inf"], result["tol"])
        self.assertLessEqual(result["worst_token_rel_inf"], result["tol"])
        self.assertEqual(result["intermediate_decrypt_attempts"], 0)
        self.assertEqual(result["final_decrypt_calls"], 1)
        self.assertFalse(result["evaluator_has_private_key"])
        self.assertEqual(result["source_sha256"], sha256(source))
        self.assertEqual(result["fixture_sha256"], sha256(fixture))
        self.assertEqual(source.read_text().count("cc->Decrypt("), 1)

    def test_real_width_ln1_gate_is_bound_to_source_and_fixture(self):
        result = load_run("fhe_fides_real_d768_t2_ln1_a100_asymfix2_20260724.json")
        source = ROOT / "fhe" / "gpu_real" / "src" / "real_dnagpt_fides.cpp"
        fixture_contract = ROOT / "fhe" / "gpu_real" / "fixture.sha256"

        self.assertEqual(result["gate"], "ln1")
        self.assertTrue(result["passed"])
        self.assertEqual(result["config"]["D"], 768)
        self.assertEqual(result["config"]["T"], 2)
        self.assertEqual(result["config"]["batch_slots"], 4096)
        self.assertEqual(result["intermediate_decrypt_attempts"], 0)
        self.assertEqual(result["final_decrypt_calls"], 1)
        self.assertFalse(result["evaluator_has_private_key"])
        self.assertEqual(result["source_sha256"], sha256(source))
        self.assertEqual(result["fixture_contract_sha256"], sha256(fixture_contract))
        self.assertEqual(source.read_text().count("cc->Decrypt("), 1)

    def test_real_width_attention_gate_is_bound_to_source_and_fixture(self):
        result = load_run(
            "fhe_fides_real_d768_t2_attention_a100_asymfix2_20260724.json"
        )
        source = ROOT / "fhe" / "gpu_real" / "src" / "real_dnagpt_fides.cpp"
        fixture_contract = ROOT / "fhe" / "gpu_real" / "fixture.sha256"

        self.assertEqual(result["gate"], "attention")
        self.assertTrue(result["passed"])
        self.assertLessEqual(result["global_rel_inf"], result["tol"])
        self.assertLessEqual(result["worst_token_rel_inf"], result["tol"])
        self.assertEqual(result["config"]["D"], 768)
        self.assertEqual(result["config"]["T"], 2)
        self.assertEqual(result["config"]["heads"], 12)
        self.assertEqual(result["config"]["rotation_keys"], 63)
        self.assertEqual(result["intermediate_decrypt_attempts"], 0)
        self.assertEqual(result["final_decrypt_calls"], 1)
        self.assertFalse(result["evaluator_has_private_key"])
        self.assertEqual(result["source_sha256"], sha256(source))
        self.assertEqual(result["fixture_contract_sha256"], sha256(fixture_contract))
        self.assertEqual(source.read_text().count("cc->Decrypt("), 1)

    def test_original_full_block_depth_failure_is_preserved(self):
        result = load_run("fhe_fides_real_d768_t2_block0_depth43_FAIL_20260724.json")
        source = ROOT / "fhe" / "gpu_real" / "src" / "real_dnagpt_fides.cpp"

        self.assertIn("[V] FAIL", result["status"])
        self.assertEqual(result["config"]["multiplicative_depth"], 43)
        self.assertEqual(
            result["observed_level_trace"]["attention_projection"], [13, 26]
        )
        self.assertEqual(result["observed_level_trace"]["ln2"], [24, 37])
        self.assertEqual(result["failure"]["observed_intermediate_decrypts"], 0)
        self.assertEqual(result["failure"]["final_decrypt_calls"], 0)
        self.assertEqual(result["provenance"]["source_sha256"], sha256(source))


if __name__ == "__main__":
    unittest.main()
