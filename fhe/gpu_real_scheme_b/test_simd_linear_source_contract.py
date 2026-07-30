"""Static/local contract for the additive Token-SIMD C++ linear micro-gate."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).parent
SOURCE = ROOT / "src" / "real_dnagpt_fides_scheme_b_simd_linear_t103.cpp"
FROZEN_T32 = ROOT / "src" / "real_dnagpt_fides_scheme_b_general_attention_t32.cpp"
FROZEN_T32_SHA256 = "607e9c429bdea107149733e10f19765578175a5301b350556bd8604a1d384252"
FIXTURE_CONTRACT = ROOT / "fixture_t103.sha256"
FIXTURE_CONTRACT_SHA256 = (
    "061d53bd25bbcaf75d4c12067ec77f032c3ba0e35300a0a6168c2ce15f3672db"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenSimdLinearSourceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_parent_hash_is_unchanged_and_pinned(self) -> None:
        self.assertEqual(_sha256(FROZEN_T32), FROZEN_T32_SHA256)
        self.assertIn(FROZEN_T32_SHA256, self.source)
        self.assertIn("PINNED_PARENT_SOURCE_SHA256", self.source)
        self.assertIn("--parent-source-sha256", self.source)

    def test_fixture_contract_hash_is_unchanged_and_pinned(self) -> None:
        self.assertEqual(_sha256(FIXTURE_CONTRACT), FIXTURE_CONTRACT_SHA256)
        self.assertIn(FIXTURE_CONTRACT_SHA256, self.source)
        self.assertIn("PINNED_FIXTURE_CONTRACT_SHA256", self.source)
        self.assertIn(
            "options.fixture_contract_sha256 !=\n"
            "        PINNED_FIXTURE_CONTRACT_SHA256",
            self.source,
        )

    def test_t103_batch_geometry_and_partial_tail_are_static(self) -> None:
        for text in (
            "constexpr std::size_t T = 103;",
            "constexpr std::size_t TOKEN_BATCH = 8;",
            "constexpr std::size_t SLOTS = LOGICAL_SLOTS * TOKEN_BATCH;",
            "static_assert(SLOTS == 32768);",
            "static_assert(TOKEN_GROUPS == 13);",
            "static_assert(T % TOKEN_BATCH == 7);",
        ):
            self.assertIn(text, self.source)

    def test_slot_mapping_has_token_as_innermost_dimension(self) -> None:
        self.assertIn(
            "return (copy * PACK_WIDTH + feature) * TOKEN_BATCH + token_lane;",
            self.source,
        )
        self.assertIn("physical_slot(copy, dim, token_lane)", self.source)
        self.assertIn("physical_slot(0, dim, token_lane)", self.source)

    def test_every_bsgs_rotation_is_scaled_by_token_batch(self) -> None:
        self.assertIn("TOKEN_BATCH * small", self.source)
        self.assertIn("TOKEN_BATCH * BSGS_N1 * giant", self.source)
        self.assertIn(
            "add_accumulate_keys(indices, static_cast<int>(PACK_WIDTH),\n"
            "                        static_cast<int>(TOKEN_BATCH));",
            self.source,
        )
        self.assertIn(
            "cc_->AccumulateSum(input, static_cast<int>(PACK_WIDTH),\n"
            "                                    static_cast<int>(TOKEN_BATCH))",
            self.source,
        )
        evaluator = self.source.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]
        raw_rotate_calls = re.findall(r"EvalRotate\([^;]+;", evaluator, re.S)
        self.assertTrue(raw_rotate_calls)
        for call in raw_rotate_calls:
            self.assertIn("TOKEN_BATCH", call)

    def test_plaintext_coefficients_repeat_over_all_token_lanes(self) -> None:
        self.assertIn(
            "packed[physical_slot(copy, dim, token)] = values[dim];",
            self.source,
        )
        self.assertIn(
            "packed_values[physical_slot(\n"
            "                                    copy, row, token)] = coefficient;",
            self.source,
        )
        for text in (
            "Plaintext mean_scale = repeated_plain",
            "Plaintext ln1_weight = repeated_plain",
            "Plaintext diagonal_plain =",
        ):
            self.assertIn(text, self.source)

    def test_one_ciphertext_and_one_query_matmul_per_group(self) -> None:
        self.assertIn("std::vector<Ct> encrypted_inputs", self.source)
        self.assertIn("std::vector<Ct> packed_query", self.source)
        self.assertIn("encrypted_inputs.reserve(group_count)", self.source)
        self.assertIn(
            "result.packed_query.reserve(encrypted_inputs.size())", self.source
        )
        self.assertIn("result.matrix_products = matmul_count_;", self.source)
        self.assertIn("evaluation.matrix_products != group_count", self.source)
        self.assertIn("client.round_trips() != group_count", self.source)
        self.assertIn("client.logical_instances() != T", self.source)

    def test_packed_and_matched_serial_control_are_fail_closed(self) -> None:
        for text in (
            "--mode packed|serial_control",
            'options.mode != "packed"',
            'options.mode != "serial_control"',
            'return mode == "packed" ? TOKEN_GROUPS : T;',
            'if (mode == "serial_control")',
            '<< "  \\"mode\\": \\"" << json_escape(options.mode)',
            "mode_first_token(options.mode, group)",
            "mode_active_tokens(options.mode, group)",
        ):
            self.assertIn(text, self.source)
        self.assertIn(
            "evaluation.packed_query.size()",
            self.source,
        )

    def test_security_and_depth_parameters_match_frozen_scheme_b(self) -> None:
        for text in (
            "HEStd_128_classic",
            "constexpr std::uint32_t MULT_DEPTH = 16;",
            "constexpr std::uint32_t SCALE_BITS = 50;",
            "parameters.SetBatchSize(SLOTS);",
            "parameters.SetDevices({options.gpu});",
            "parameters.SetKeySwitchTechnique(HYBRID);",
        ):
            self.assertIn(text, self.source)

    def test_evaluator_has_no_secret_key_or_decrypt(self) -> None:
        evaluator = self.source.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]
        self.assertNotIn("secretKey", evaluator)
        self.assertNotIn("Decrypt(", evaluator)
        self.assertIn('"evaluator_has_private_key\\": false', self.source)

    def test_evidence_is_exclusive_and_hash_bound(self) -> None:
        self.assertIn("O_WRONLY | O_CREAT | O_EXCL", self.source)
        for text in (
            '\\"source_sha256\\"',
            '\\"parent_source_sha256\\"',
            '\\"fixture_manifest_sha256\\"',
            '\\"fixture_contract_sha256\\"',
            '\\"matrix_products\\"',
            '\\"frozen_equivalent_matrix_products\\"',
            '\\"dense_call_reduction\\"',
            '\\"server_linear_algebra_seconds\\"',
        ):
            self.assertIn(text, self.source)

    def test_build_and_capacity_gated_deployment_wiring_exists(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        target = "real_dnagpt_fides_scheme_b_simd_linear_t103"
        self.assertIn(target, cmake)
        self.assertIn(target, build)
        for script_name in (
            "run_scheme_b_simd_linear_t103.sh",
            "launch_brev_scheme_b_simd_linear_t103.sh",
            "wait_and_run_scheme_b_simd_linear_t103.sh",
            "wait_build_and_run_scheme_b_simd_linear_t103.sh",
        ):
            script = ROOT / script_name
            self.assertTrue(script.is_file(), script)
            text = script.read_text(encoding="utf-8")
            self.assertIn("_simd_linear_t103_", text)
            self.assertIn("_scheme_b_", text)
        launcher = (ROOT / "launch_brev_scheme_b_simd_linear_t103.sh").read_text(
            encoding="utf-8"
        )
        waiter = (ROOT / "wait_and_run_scheme_b_simd_linear_t103.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("gpucap_preflight_confirm_gpu", launcher)
        self.assertIn("gpucap_wait_for_capacity", waiter)
        self.assertIn("VRAM_LOG", launcher)


if __name__ == "__main__":
    unittest.main()
