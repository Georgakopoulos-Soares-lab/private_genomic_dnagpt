"""Fail-closed source contract for the complete T=103 Token-SIMD block."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from simd_layout import protocol_counts, server_operation_counts


ROOT = Path(__file__).parent
SOURCE = ROOT / "src" / "real_dnagpt_fides_scheme_b_simd_full_t103.cpp"
LINEAR_PARENT = ROOT / "src" / "real_dnagpt_fides_scheme_b_simd_linear_t103.cpp"
FROZEN_T103 = ROOT / "src" / "real_dnagpt_fides_scheme_b_general_attention_t103.cpp"
SCHEDULE = ROOT / "src" / "simd_t103_schedule.hpp"

LINEAR_PARENT_SHA256 = (
    "926be6b276006dd6435713efa168d189af02e02589d143684991380bae7e9233"
)
FROZEN_T103_SHA256 = "70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f"
SCHEDULE_SHA256 = "c6b221f365ba6326f615c5554458d7bd092990d23c4ba0d5106ca7577cb7c3aa"
FULL_SOURCE_SHA256 = "97727a5bb2145b32749b746c3fc1b576811dfcc3222088278962b029fc1aea4b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def class_body(source: str, class_name: str) -> str:
    start = source.index(f"class {class_name}")
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : index + 1]
    raise AssertionError(f"unterminated class {class_name}")


class TokenSimdFullT103ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_parents_and_schedule_are_unchanged(self) -> None:
        self.assertEqual(sha256(LINEAR_PARENT), LINEAR_PARENT_SHA256)
        self.assertEqual(sha256(FROZEN_T103), FROZEN_T103_SHA256)
        self.assertEqual(sha256(SCHEDULE), SCHEDULE_SHA256)
        for digest in (
            LINEAR_PARENT_SHA256,
            FROZEN_T103_SHA256,
            SCHEDULE_SHA256,
        ):
            self.assertIn(digest, self.source)

    def test_complete_fixture_and_oracle_are_loaded(self) -> None:
        for fixture_name in (
            "weights__ln1.bin",
            "weights__ln2.bin",
            "weights__attn_qkv.bin",
            "weights__attn_proj.bin",
            "weights__mlp_fc.bin",
            "weights__mlp_proj.bin",
            "oracle__block_output.bin",
        ):
            self.assertIn(fixture_name, self.source)
        self.assertIn(
            "measure_output(output, fixture.oracle_block_output)", self.source
        )

    def test_packed_geometry_and_partial_tail_are_static(self) -> None:
        for fragment in (
            "constexpr std::size_t T = 103;",
            "constexpr std::size_t TOKEN_BATCH = 8;",
            "constexpr std::size_t SLOTS = LOGICAL_SLOTS * TOKEN_BATCH;",
            "static_assert(SLOTS == 32768);",
            "static_assert(TOKEN_GROUPS == 13);",
            "static_assert(T % TOKEN_BATCH == 7);",
            "group * TOKEN_BATCH",
            "std::min(TOKEN_BATCH, T - first)",
        ):
            self.assertIn(fragment, self.source)

    def test_schedule_counts_match_the_proved_model(self) -> None:
        protocol = protocol_counts(103, 8)
        server = server_operation_counts(103, 8)
        self.assertEqual(protocol.full_block_crossings, 857)
        self.assertEqual(protocol.score_tile_decryptions, 91)
        self.assertEqual(protocol.weight_tile_encryptions, 727)
        self.assertEqual(server.matrix_products, 156)
        self.assertEqual(server.ciphertext_ciphertext_multiplications, 1506)
        self.assertEqual(server.ciphertext_plaintext_multiplications, 177734)
        self.assertEqual(server.explicit_rotations, 8173)
        self.assertEqual(server.accumulate_sum_calls, 8776)
        for value in (857, 91, 727, 156, 1506, 177734, 8173, 8776, 129162):
            self.assertRegex(
                self.source,
                rf"EXPECTED_[A-Z_]+ = {value};",
            )

    def test_attention_is_score_then_softmax_then_weight_value(self) -> None:
        begin = self.source.index("client_.begin_attention();")
        reduce = self.source.index("client_.reduce_score_tile(")
        finalize = self.source.index("client_.finalize_attention();")
        emit = self.source.index("client_.emit_weight_tile(")
        clear = self.source.index("client_.clear_attention();")
        self.assertLess(begin, reduce)
        self.assertLess(reduce, finalize)
        self.assertLess(finalize, emit)
        self.assertLess(emit, clear)
        for fragment in (
            "active_weight_shifts(query_group, key_group)",
            "build_group_shifts(key[key_group], key_group, true)",
            "build_group_shifts(value[key_group], key_group, false)",
            "score_isolation_plain(",
            "stable-softmax denominator",
        ):
            self.assertIn(fragment, self.source)

    def test_lane_shifts_are_memory_bounded_per_key_group(self) -> None:
        self.assertIn(
            "std::array<Ct, TOKEN_BATCH> build_group_shifts(",
            self.source,
        )
        self.assertNotIn(
            "std::array<std::array<Ct, TOKEN_BATCH>, TOKEN_GROUPS>",
            self.source,
        )
        self.assertIn("cached_key_lane_shifts_", self.source)
        self.assertIn("cached_value_lane_shifts_", self.source)

    def test_every_dense_family_and_full_tail_are_present(self) -> None:
        evaluator = class_body(self.source, "EncryptedEvaluator")
        for fragment in (
            "fixture_.qkv[0]",
            "fixture_.qkv[1]",
            "fixture_.qkv[2]",
            "fixture_.attention_projection",
            "fixture_.mlp_fc[chunk]",
            "fixture_.mlp_projection[chunk]",
            "client_.gelu_boundary(",
            "cc_->EvalAdd(residual1[group], mlp)",
        ):
            self.assertIn(fragment, evaluator)
        self.assertEqual(
            evaluator.count("matmul(") - 1,
            6,
            "source has six syntactic matmul call sites covering 12 logical families",
        )

    def test_evaluator_has_no_secret_key_or_decrypt(self) -> None:
        evaluator = class_body(self.source, "EncryptedEvaluator")
        self.assertNotIn("Decrypt(", evaluator)
        self.assertNotIn("secretKey", evaluator)
        self.assertNotIn("keys_", evaluator)
        self.assertEqual(
            self.source.count("cc_->Decrypt("),
            3,
            "three declared client decrypt sites",
        )
        self.assertEqual(self.source.count("cc->Decrypt("), 1)

    def test_security_depth_provenance_and_immutability_fail_closed(self) -> None:
        for fragment in (
            "parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);",
            "parameters.SetMultiplicativeDepth(MULT_DEPTH);",
            "parameters.SetBatchSize(SLOTS);",
            "parameters.SetDevices({options.gpu});",
            "std::filesystem::exists(options.output)",
            "O_WRONLY | O_CREAT | O_EXCL",
            '"intermediate_decrypt_attempts\\": 0',
            '"evaluator_has_private_key\\": false',
            "REAL_DNAGPT_TOKEN_SIMD_FULL_GATE_PASS",
        ):
            self.assertIn(fragment, self.source)

    def test_cmake_and_build_target_are_additive(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        target = "real_dnagpt_fides_scheme_b_simd_full_t103"
        self.assertIn(target, cmake)
        self.assertIn(target, build)
        self.assertIn(SOURCE.name, cmake)

    def test_run_launch_and_capacity_gates_are_fail_closed(self) -> None:
        run = (ROOT / "run_scheme_b_simd_full_t103.sh").read_text(encoding="utf-8")
        launch = (ROOT / "launch_brev_scheme_b_simd_full_t103.sh").read_text(
            encoding="utf-8"
        )
        wait = (ROOT / "wait_and_run_scheme_b_simd_full_t103.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(FULL_SOURCE_SHA256, run)
        for digest in (
            LINEAR_PARENT_SHA256,
            FROZEN_T103_SHA256,
            SCHEDULE_SHA256,
        ):
            self.assertIn(digest, run)
        self.assertIn("fixture_t103.sha256", run)
        self.assertIn('if [[ -e "${OUTPUT}" ]]', run)
        self.assertIn("run_scheme_b_simd_full_t103.sh", launch)
        self.assertIn("gpucap_preflight_confirm_gpu", launch)
        self.assertIn("nvidia-smi --query-compute-apps", launch)
        self.assertIn("launch_brev_scheme_b_simd_full_t103.sh", wait)
        self.assertIn("gpucap_wait_for_capacity", wait)
        for script in (run, launch, wait):
            self.assertIn("_simd_full_t103_", script)
            self.assertNotIn("_simd_linear_t103_", script)


if __name__ == "__main__":
    unittest.main()
