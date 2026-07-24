import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(name):
    return json.loads((ROOT / "results" / "runs" / name).read_text())


class FHEEvidenceTests(unittest.TestCase):
    def test_toy_block_closes_the_no_mid_decrypt_gate(self):
        result = load_run("fhe_toy_block.json")
        self.assertTrue(result["passed"])
        self.assertLessEqual(result["global_rel_inf"], result["tol"])
        self.assertLessEqual(result["worst_token_rel_inf"], result["tol"])
        self.assertEqual(result["intermediate_decrypt_attempts"], 0)
        self.assertEqual(result["final_decrypt_calls"], 1)
        self.assertEqual(result["security"], "HEStd_128_classic")

        script = ROOT / "fhe" / "block" / "toy_block_ckks.py"
        oracle = ROOT / "fhe" / "oracle.py"
        self.assertEqual(result["provenance"]["script_sha256"], sha256(script))
        self.assertEqual(result["provenance"]["oracle_sha256"], sha256(oracle))
        self.assertEqual(script.read_text().count(".Decrypt("), 1)

    def test_extrapolation_is_bound_to_immutable_inputs(self):
        result = load_run("fhe_0p1b_extrapolation_20260724.json")
        measured = result["measured_inputs"]
        toy = ROOT / measured["toy_block_result"]
        bootstrap = ROOT / measured["bootstrap_result"]
        self.assertEqual(measured["toy_block_result_sha256"], sha256(toy))
        self.assertEqual(measured["bootstrap_result_sha256"], sha256(bootstrap))
        self.assertIn("[A/U]", result["status"])

    def test_local_optimization_screen_is_bound_to_source(self):
        result = load_run("fhe_local_optimizations_20260724.json")
        source = ROOT / "fhe" / "optimizations" / "local_ckks_probe.py"
        self.assertTrue(result["passed"])
        self.assertEqual(result["source_sha256"], sha256(source))
        self.assertEqual(
            result["probes"]["matmul"]["operation_reduction"][
                "naive_to_bsgs_rotations"
            ],
            {"from": 15, "to": 6},
        )
        self.assertEqual(result["probes"]["attention"]["levels_saved"], 2)

    def test_runtime_projection_is_bound_to_immutable_inputs(self):
        result = load_run("fhe_runtime_projection_20260724.json")
        inputs = result["inputs"]
        work = ROOT / inputs["work_count_result"]
        toy = ROOT / inputs["toy_block_result"]
        self.assertEqual(inputs["work_count_sha256"], sha256(work))
        self.assertEqual(inputs["toy_block_sha256"], sha256(toy))
        self.assertIn("[A/U]", result["status"])

    def test_measured_t2_runtime_boundary_is_narrow_and_input_bound(self):
        result = load_run("fhe_measured_t2_attention_boundary_20260724.json")
        measured = ROOT / result["input"]["result"]
        attention = json.loads(measured.read_text())

        self.assertEqual(result["input"]["sha256"], sha256(measured))
        self.assertTrue(attention["passed"])
        self.assertEqual(attention["gate"], "attention")
        self.assertEqual(result["target"]["sequence_length"], 2)
        self.assertAlmostEqual(
            result["derived"]["attention_only_repeated_seconds"],
            attention["timings_seconds"]["encrypted_evaluation"] * 12,
        )
        self.assertEqual(
            result["full_task_boundary"]["runtime"], "[U] not extrapolated"
        )
        self.assertIn("not a 12-block encrypted run", result["status"])


if __name__ == "__main__":
    unittest.main()
