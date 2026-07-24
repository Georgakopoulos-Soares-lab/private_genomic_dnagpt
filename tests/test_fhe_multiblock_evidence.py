import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "runs" / "fhe_multiblock_plaintext_contract_20260724.json"
RANGE_RESULT = (
    ROOT
    / "results"
    / "runs"
    / "fhe_range_control_t2_12block_optimized_v2_20260724.json"
)
FIXTURE = (
    ROOT
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_multiblock_t2_d63353abdc1a_52d046d1fcf0"
    / "manifest.json"
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FHEMultiblockEvidenceTests(unittest.TestCase):
    def test_result_is_bound_to_reproducible_sources(self):
        result = json.loads(RESULT.read_text())
        sources = {
            "export_fixture_sha256": ROOT / "fhe" / "multiblock" / "export_fixture.py",
            "manual_sha256": ROOT / "fhe" / "multiblock" / "manual.py",
            "realweights_contract_sha256": ROOT / "fhe" / "realweights" / "contract.py",
            "realweights_manual_sha256": ROOT / "fhe" / "realweights" / "manual.py",
        }

        self.assertTrue(result["oracle_gate"]["passed"])
        self.assertTrue(result["oracle_gate"]["all_finite"])
        self.assertFalse(
            result["composition_finding"]["block0_fixed_domains_compose_all_12"]
        )
        self.assertEqual(result["composition_finding"]["first_failed_block"], 1)
        for field, path in sources.items():
            self.assertEqual(result["provenance"][field], sha256(path))

    def test_generated_fixture_manifest_matches_when_present(self):
        if not FIXTURE.exists():
            self.skipTest("ignored 688 MB fixture has not been generated")
        result = json.loads(RESULT.read_text())
        manifest = json.loads(FIXTURE.read_text())

        self.assertEqual(result["fixture"]["manifest_sha256"], sha256(FIXTURE))
        self.assertEqual(
            result["fixture"]["content_identity_sha256"],
            manifest["content_identity_sha256"],
        )
        self.assertTrue(manifest["oracle_gate"]["passed"])
        self.assertFalse(
            manifest["public_calibration"]["block0_gpu_fixed_domains_compose_all_12"]
        )
        self.assertEqual(
            result["classification_boundary"]["full_prompt_margin_n_minus_a"],
            manifest["classification"]["complete_public_prompt_reference"][
                "margin_n_minus_a"
            ],
        )

    def test_range_control_result_is_bound_to_full_evidence_and_sources(self):
        result = json.loads(RANGE_RESULT.read_text())
        full_result = ROOT / result["provenance"]["full_result_path"]
        sources = {
            "schedule_sha256": ROOT / "fhe" / "range_control" / "schedule.py",
            "simulate_sha256": ROOT / "fhe" / "range_control" / "simulate.py",
            "tests_sha256": ROOT / "fhe" / "range_control" / "test_range_control.py",
            "readme_sha256": ROOT / "fhe" / "range_control" / "README.md",
        }

        self.assertTrue(result["metrics"]["all_12_blocks_passed"])
        self.assertTrue(result["metrics"]["head_passed"])
        self.assertEqual(result["metrics"]["domain_violation_count"], 0)
        self.assertTrue(result["metrics"]["classification_label_preserved"])
        self.assertLessEqual(
            result["metrics"]["worst_block_global_rel_inf"],
            result["config"]["acceptance_tol"],
        )
        self.assertLessEqual(
            result["metrics"]["worst_block_worst_token_rel_inf"],
            result["config"]["acceptance_tol"],
        )
        self.assertEqual(
            result["provenance"]["full_result_sha256"], sha256(full_result)
        )
        for field, path in sources.items():
            self.assertEqual(result["provenance"][field], sha256(path))


if __name__ == "__main__":
    unittest.main()
