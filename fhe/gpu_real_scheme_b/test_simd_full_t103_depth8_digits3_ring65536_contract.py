"""Contract for the additive T=103 depth-8/digits-3/ring-65536 fork."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).parent
SOURCE = ROOT / "src"
BASE = SOURCE / "real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3.cpp"
RING65536 = (
    SOURCE / "real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536.cpp"
)

BASE_SHA256 = "c53e47123614bcddf6a577ca5cc1e22db786d234e953e5c31b6a488e3408c089"
RING65536_SHA256 = "65cc818cbb299af984cccdbe6a7df891cac5b7071f663edfac8c3f13be161365"

REPLACEMENTS = (
    (
        "// Additive depth-8/digits-3 complete Token-SIMD block-0 gate for Scheme B.",
        "// Additive depth-8/digits-3/ring-65536 complete Token-SIMD block-0 gate\n"
        "// for Scheme B.",
    ),
    (
        "//   direct parent: depth-8/digits-4 context-rejected prototype",
        "//   direct parent: depth-8/digits-3 context-rejected prototype",
    ),
    (
        "//     58c257522eefe7a928e6efaa486af6dd3ac53c289fc07878bb41d5d5f9350882",
        "//     c53e47123614bcddf6a577ca5cc1e22db786d234e953e5c31b6a488e3408c089",
    ),
    (
        "constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;\n"
        "constexpr std::uint32_t MULT_DEPTH = 8;",
        "constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;\n"
        "constexpr std::uint32_t RING_DIM = 65536;\n"
        "constexpr std::uint32_t MULT_DEPTH = 8;",
    ),
    (
        "static_assert(SLOTS == 32768);\nstatic_assert(TOKEN_GROUPS == 13);",
        "static_assert(SLOTS == 32768);\n"
        "static_assert(RING_DIM == 2 * SLOTS);\n"
        "static_assert(TOKEN_GROUPS == 13);",
    ),
    (
        '    "58c257522eefe7a928e6efaa486af6dd3ac53c289fc07878bb41d5d5f9350882";',
        '    "c53e47123614bcddf6a577ca5cc1e22db786d234e953e5c31b6a488e3408c089";',
    ),
    (
        "        std::cout\n"
        '                << "Token-SIMD T=103 depth-8/digits-3 complete '
        'block-0 gate\\n";',
        "        std::cout\n"
        '            << "Token-SIMD T=103 depth-8/digits-3/ring-65536 complete "\n'
        '               "block-0 gate\\n";',
    ),
    (
        '        << "  \\"task\\": \\"Scheme B Token-SIMD T=103 depth-8/digits-3 complete "\n'
        '           "released-weight DNAGPT block-0 gate\\",\\n"',
        '        << "  \\"task\\": \\"Scheme B Token-SIMD T=103 "\n'
        '           "depth-8/digits-3/ring-65536 complete "\n'
        '           "released-weight DNAGPT block-0 gate\\",\\n"',
    ),
    (
        '"\\"t103-scheme-b-token-simd-full-depth8-digits3-v1\\",\\n"',
        '"\\"t103-scheme-b-token-simd-full-depth8-digits3-ring65536-v1\\",\\n"',
    ),
    (
        "        parameters.SetBatchSize(SLOTS);\n"
        "        parameters.SetDevices({options.gpu});",
        "        parameters.SetBatchSize(SLOTS);\n"
        "        parameters.SetRingDim(RING_DIM);\n"
        "        parameters.SetDevices({options.gpu});",
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH8_DIGITS3_GATE_PASS\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH8_DIGITS3_RING65536_GATE_PASS\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH8_DIGITS3_GATE_FAIL\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH8_DIGITS3_RING65536_GATE_FAIL\\n"',
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TokenSimdFullT103Depth8Digits3Ring65536ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = BASE.read_text(encoding="utf-8")
        cls.ring65536 = RING65536.read_text(encoding="utf-8")

    def test_parent_and_fork_hashes_are_pinned(self) -> None:
        self.assertEqual(sha256(BASE), BASE_SHA256)
        self.assertEqual(sha256(RING65536), RING65536_SHA256)

    def test_fork_changes_only_declared_identity_and_ring_fields(self) -> None:
        expected = self.base
        for old, new in REPLACEMENTS:
            self.assertEqual(expected.count(old), 1, old)
            expected = expected.replace(old, new)
        self.assertEqual(self.ring65536, expected)

    def test_ring_is_explicit_and_other_crypto_fields_are_unchanged(
        self,
    ) -> None:
        for fragment in (
            "constexpr std::uint32_t MULT_DEPTH = 8;",
            "constexpr std::uint32_t LARGE_DIGITS = 3;",
            "constexpr std::uint32_t RING_DIM = 65536;",
            "constexpr std::uint32_t SCALE_BITS = 50;",
            "static_assert(SLOTS == 32768);",
            "static_assert(RING_DIM == 2 * SLOTS);",
            "parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);",
            "parameters.SetBatchSize(SLOTS);",
            "parameters.SetRingDim(RING_DIM);",
            "O_WRONLY | O_CREAT | O_EXCL",
            '"intermediate_decrypt_attempts\\": 0',
            '"evaluator_has_private_key\\": false',
        ):
            self.assertIn(fragment, self.ring65536)

    def test_build_target_is_additive(self) -> None:
        target = "real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3_ring65536"
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        self.assertIn(target, cmake)
        self.assertIn(target, build)
        self.assertIn(RING65536.name, cmake)

    def test_run_script_pins_source_parent_and_fixture(self) -> None:
        run = (
            ROOT / "run_scheme_b_simd_full_t103_depth8_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            RING65536_SHA256,
            BASE_SHA256,
            RING65536.name,
            BASE.name,
            "fixture_t103.sha256",
            'if [[ -e "${OUTPUT}" ]]',
            "_simd_full_t103_depth8_digits3_ring65536_",
        ):
            self.assertIn(fragment, run)

    def test_launch_and_wait_scripts_keep_capacity_and_vram_gates(self) -> None:
        launch = (
            ROOT / "launch_brev_scheme_b_simd_full_t103_depth8_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        wait = (
            ROOT / "wait_and_run_scheme_b_simd_full_t103_depth8_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("gpucap_preflight_confirm_gpu", launch)
        self.assertIn("nvidia-smi --query-compute-apps", launch)
        self.assertIn("gpucap_wait_for_capacity", wait)
        self.assertIn("SCHEME_B_SIMD_FULL_DEPTH8_DIGITS3_RING65536_RUN_TAG", wait)
        for script in (launch, wait):
            self.assertIn("_simd_full_t103_depth8_digits3_ring65536_", script)
            self.assertNotIn('"run_scheme_b_simd_full_t103_depth8_digits3.sh"', script)
            self.assertNotIn(
                '"launch_brev_scheme_b_simd_full_t103_depth8_digits3.sh"',
                script,
            )

    def test_scripts_parse_and_are_executable(self) -> None:
        for name in (
            "run_scheme_b_simd_full_t103_depth8_digits3_ring65536.sh",
            "launch_brev_scheme_b_simd_full_t103_depth8_digits3_ring65536.sh",
            "wait_and_run_scheme_b_simd_full_t103_depth8_digits3_ring65536.sh",
        ):
            path = ROOT / name
            self.assertTrue(path.stat().st_mode & 0o111, name)


if __name__ == "__main__":
    unittest.main()
