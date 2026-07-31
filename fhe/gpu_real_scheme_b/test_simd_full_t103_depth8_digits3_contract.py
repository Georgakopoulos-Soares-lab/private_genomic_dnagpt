"""Fail-closed contract for the additive T=103 depth-8/digits-3 fork."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).parent
SOURCE = ROOT / "src"
BASE = SOURCE / "real_dnagpt_fides_scheme_b_simd_full_t103_depth8.cpp"
DIGITS3 = SOURCE / "real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3.cpp"

BASE_SHA256 = "58c257522eefe7a928e6efaa486af6dd3ac53c289fc07878bb41d5d5f9350882"
DIGITS3_SHA256 = "c53e47123614bcddf6a577ca5cc1e22db786d234e953e5c31b6a488e3408c089"

REPLACEMENTS = (
    (
        "// Additive depth-8 complete Token-SIMD block-0 gate for Scheme B.",
        "// Additive depth-8/digits-3 complete Token-SIMD block-0 gate for Scheme B.",
    ),
    (
        "//   direct parent: passing T=103 Token-SIMD complete block",
        "//   direct parent: depth-8/digits-4 context-rejected prototype",
    ),
    (
        "//     97727a5bb2145b32749b746c3fc1b576811dfcc3222088278962b029fc1aea4b",
        "//     58c257522eefe7a928e6efaa486af6dd3ac53c289fc07878bb41d5d5f9350882",
    ),
    (
        "constexpr std::uint32_t LARGE_DIGITS = 4;",
        "constexpr std::uint32_t LARGE_DIGITS = 3;",
    ),
    (
        '    "97727a5bb2145b32749b746c3fc1b576811dfcc3222088278962b029fc1aea4b";',
        '    "58c257522eefe7a928e6efaa486af6dd3ac53c289fc07878bb41d5d5f9350882";',
    ),
    (
        'std::cout << "Token-SIMD T=103 depth-8 complete block-0 gate\\n";',
        "std::cout\n"
        '                << "Token-SIMD T=103 depth-8/digits-3 complete '
        'block-0 gate\\n";',
    ),
    (
        '<< "  \\"task\\": \\"Scheme B Token-SIMD T=103 depth-8 complete "',
        '<< "  \\"task\\": \\"Scheme B Token-SIMD T=103 depth-8/digits-3 complete "',
    ),
    (
        '"\\"t103-scheme-b-token-simd-full-depth8-v1\\",\\n"',
        '"\\"t103-scheme-b-token-simd-full-depth8-digits3-v1\\",\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH8_GATE_PASS\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH8_DIGITS3_GATE_PASS\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH8_GATE_FAIL\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH8_DIGITS3_GATE_FAIL\\n"',
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TokenSimdFullT103Depth8Digits3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = BASE.read_text(encoding="utf-8")
        cls.digits3 = DIGITS3.read_text(encoding="utf-8")

    def test_parent_and_fork_hashes_are_pinned(self) -> None:
        self.assertEqual(sha256(BASE), BASE_SHA256)
        self.assertEqual(sha256(DIGITS3), DIGITS3_SHA256)

    def test_fork_changes_only_declared_identity_and_depth_fields(self) -> None:
        expected = self.base
        for old, new in REPLACEMENTS:
            self.assertEqual(expected.count(old), 1, old)
            expected = expected.replace(old, new)
        self.assertEqual(self.digits3, expected)

    def test_depth8_keeps_scale_slots_security_and_fail_closed_output(self) -> None:
        for fragment in (
            "constexpr std::uint32_t MULT_DEPTH = 8;",
            "constexpr std::uint32_t LARGE_DIGITS = 3;",
            "constexpr std::uint32_t SCALE_BITS = 50;",
            "static_assert(SLOTS == 32768);",
            "parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);",
            "parameters.SetBatchSize(SLOTS);",
            "O_WRONLY | O_CREAT | O_EXCL",
            '"intermediate_decrypt_attempts\\": 0',
            '"evaluator_has_private_key\\": false',
        ):
            self.assertIn(fragment, self.digits3)

    def test_build_target_is_additive(self) -> None:
        target = "real_dnagpt_fides_scheme_b_simd_full_t103_depth8_digits3"
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        self.assertIn(target, cmake)
        self.assertIn(target, build)
        self.assertIn(DIGITS3.name, cmake)

    def test_run_script_pins_source_parent_and_fixture(self) -> None:
        run = (ROOT / "run_scheme_b_simd_full_t103_depth8_digits3.sh").read_text(
            encoding="utf-8"
        )
        for fragment in (
            DIGITS3_SHA256,
            BASE_SHA256,
            DIGITS3.name,
            BASE.name,
            "fixture_t103.sha256",
            'if [[ -e "${OUTPUT}" ]]',
            "_simd_full_t103_depth8_digits3_",
        ):
            self.assertIn(fragment, run)

    def test_launch_and_wait_scripts_keep_capacity_and_vram_gates(self) -> None:
        launch = (
            ROOT / "launch_brev_scheme_b_simd_full_t103_depth8_digits3.sh"
        ).read_text(encoding="utf-8")
        wait = (
            ROOT / "wait_and_run_scheme_b_simd_full_t103_depth8_digits3.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("gpucap_preflight_confirm_gpu", launch)
        self.assertIn("nvidia-smi --query-compute-apps", launch)
        self.assertIn("gpucap_wait_for_capacity", wait)
        self.assertIn("SCHEME_B_SIMD_FULL_DEPTH8_DIGITS3_RUN_TAG", wait)
        for script in (launch, wait):
            self.assertIn("_simd_full_t103_depth8_digits3_", script)
            self.assertNotIn("run_scheme_b_simd_full_t103_depth8.sh", script)
            self.assertNotIn("launch_brev_scheme_b_simd_full_t103_depth8.sh", script)

    def test_scripts_parse_and_are_executable(self) -> None:
        for name in (
            "run_scheme_b_simd_full_t103_depth8_digits3.sh",
            "launch_brev_scheme_b_simd_full_t103_depth8_digits3.sh",
            "wait_and_run_scheme_b_simd_full_t103_depth8_digits3.sh",
        ):
            path = ROOT / name
            self.assertTrue(path.stat().st_mode & 0o111, name)


if __name__ == "__main__":
    unittest.main()
