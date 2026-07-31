"""Contract for the diagnostic-only T=103 depth-10-diag/digits-3/ring-65536 fork.

Parent is the depth-10/digits-3/ring-65536 fork, which got past all 91 score
tiles, attention projection, residual, and the LN2 client boundary, then hit
its own require_remaining_depth() guard: "insufficient depth before
Token-SIMD MLP group 0" -- see docs/hybrid/tasks.md and
docs/hybrid/simd_current_state.txt (2026-07-30). The exact consumed level at
that point was never logged. This fork keeps MULT_DEPTH=10 UNCHANGED and adds
exactly one diagnostic print of normalized2->GetLevel() immediately before
the guard, so the next real fork can jump directly to the correct
MULT_DEPTH instead of guessing with further +1 increments. It does not patch
the frozen FIDESlib library and does not change any crypto parameter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).parent
SOURCE = ROOT / "src"
BASE = (
    SOURCE / "real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536.cpp"
)
DIAG = (
    SOURCE
    / "real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.cpp"
)

BASE_SHA256 = "703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2"
DIAG_SHA256 = "258e0da1ece28ef8f42652097ed8c7b61218b2a76c2d297f879a860410c72e82"

REPLACEMENTS = (
    (
        "// Additive depth-10/digits-3/ring-65536 complete Token-SIMD block-0 gate",
        "// Additive depth-10-diag/digits-3/ring-65536 complete Token-SIMD block-0\n"
        "// gate (diagnostic-only: adds one level print, no crypto/schedule change)",
    ),
    (
        "//   direct parent: depth-9/digits-3/ring-65536 prototype (fixed the\n"
        "//   depth-8 crash, but then failed differently: see Reason below)\n"
        "//   Reason: the depth-9/digits-3/ring-65536 parent got past all 91\n"
        "//   causal score tiles (fixing the depth-8 vector-underflow crash),\n"
        "//   but then OpenFHE's Decode() failed with \"approximation error is\n"
        '//   too high" while decrypting the last/deepest score tile\n'
        "//   (q=12 k=12) at the client boundary (Client::reduce_score_tile).\n"
        "//   Read as one more level of remaining precision still needed at\n"
        "//   that one worst-case ciphertext. This fork only changes\n"
        "//   MULT_DEPTH to 10 for that extra level; it does not patch the\n"
        "//   frozen FIDESlib library.",
        "//   direct parent: depth-10/digits-3/ring-65536 prototype (fixed the\n"
        "//   depth-9 decode-precision failure, but then failed a third way:\n"
        "//   see Reason below)\n"
        "//   Reason: the depth-10 parent got past all 91 score tiles, attention\n"
        "//   projection, residual, and the LN2 client boundary, then hit its own\n"
        '//   self-diagnosing require_remaining_depth() guard: "insufficient\n'
        "//   depth before Token-SIMD MLP group 0\" (normalized2's level leaves\n"
        "//   fewer than 4 remaining levels under MULT_DEPTH=10). The exact\n"
        "//   consumed level was never logged. This fork keeps MULT_DEPTH=10\n"
        "//   UNCHANGED and adds exactly one diagnostic print of\n"
        "//   normalized2->GetLevel() immediately before that guard, so the next\n"
        "//   fork can jump directly to the correct MULT_DEPTH instead of\n"
        "//   guessing with further +1 increments. It does not patch the frozen\n"
        "//   FIDESlib library and does not change any crypto parameter.",
    ),
    (
        "//     738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235",
        "//     703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2",
    ),
    (
        '    "738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235";',
        '    "703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2";',
    ),
    (
        '            << "Token-SIMD T=103 depth-10/digits-3/ring-65536 complete "\n'
        '               "block-0 gate\\n";',
        '            << "Token-SIMD T=103 depth-10-diag/digits-3/ring-65536 complete "\n'
        '               "block-0 gate\\n";',
    ),
    (
        "            Ct normalized2 = layernorm(\n"
        '                residual1[group], active_tokens[group], "ln2");\n'
        "            require_remaining_depth(\n"
        "                normalized2->GetLevel(), 4,\n"
        '                "Token-SIMD MLP group " + std::to_string(group));',
        "            Ct normalized2 = layernorm(\n"
        '                residual1[group], active_tokens[group], "ln2");\n'
        '            std::cout << "[diag] ln2 group=" << group\n'
        "                      << \" level=\" << normalized2->GetLevel() << '\\n';\n"
        "            require_remaining_depth(\n"
        "                normalized2->GetLevel(), 4,\n"
        '                "Token-SIMD MLP group " + std::to_string(group));',
    ),
    (
        '           "depth-10/digits-3/ring-65536 complete "',
        '           "depth-10-diag/digits-3/ring-65536 complete "',
    ),
    (
        '"\\"t103-scheme-b-token-simd-full-depth10-digits3-ring65536-v1\\",\\n"',
        '"\\"t103-scheme-b-token-simd-full-depth10-diag-digits3-ring65536-v1\\",\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH10_DIGITS3_RING65536_GATE_PASS\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH10_DIAG_DIGITS3_RING65536_GATE_PASS\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH10_DIGITS3_RING65536_GATE_FAIL\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH10_DIAG_DIGITS3_RING65536_GATE_FAIL\\n"',
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TokenSimdFullT103Depth10DiagDigits3Ring65536ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = BASE.read_text(encoding="utf-8")
        cls.diag = DIAG.read_text(encoding="utf-8")

    def test_parent_and_fork_hashes_are_pinned(self) -> None:
        self.assertEqual(sha256(BASE), BASE_SHA256)
        self.assertEqual(sha256(DIAG), DIAG_SHA256)

    def test_fork_changes_only_declared_identity_and_diag_print(self) -> None:
        expected = self.base
        for old, new in REPLACEMENTS:
            self.assertEqual(expected.count(old), 1, old)
            expected = expected.replace(old, new)
        self.assertEqual(self.diag, expected)

    def test_mult_depth_unchanged_and_other_crypto_fields_intact(self) -> None:
        for fragment in (
            "constexpr std::uint32_t MULT_DEPTH = 10;",
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
            '"[diag] ln2 group="',
        ):
            self.assertIn(fragment, self.diag)

    def test_build_target_is_additive(self) -> None:
        target = (
            "real_dnagpt_fides_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536"
        )
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        self.assertIn(target, cmake)
        self.assertIn(target, build)
        self.assertIn(DIAG.name, cmake)

    def test_run_script_pins_source_parent_and_fixture(self) -> None:
        run = (
            ROOT / "run_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            DIAG_SHA256,
            BASE_SHA256,
            DIAG.name,
            BASE.name,
            "fixture_t103.sha256",
            'if [[ -e "${OUTPUT}" ]]',
            "_simd_full_t103_depth10_diag_digits3_ring65536_",
        ):
            self.assertIn(fragment, run)

    def test_launch_and_wait_scripts_keep_capacity_and_vram_gates(self) -> None:
        launch = (
            ROOT
            / "launch_brev_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        wait = (
            ROOT
            / "wait_and_run_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("gpucap_preflight_confirm_gpu", launch)
        self.assertIn("nvidia-smi --query-compute-apps", launch)
        self.assertIn("gpucap_wait_for_capacity", wait)
        self.assertIn("SCHEME_B_SIMD_FULL_DEPTH10_DIAG_DIGITS3_RING65536_RUN_TAG", wait)
        for script in (launch, wait):
            self.assertIn("_simd_full_t103_depth10_diag_digits3_ring65536_", script)
            self.assertNotIn(
                '"run_scheme_b_simd_full_t103_depth10_digits3_ring65536.sh"',
                script,
            )
            self.assertNotIn(
                '"launch_brev_scheme_b_simd_full_t103_depth10_digits3_ring65536.sh"',
                script,
            )

    def test_scripts_parse_and_are_executable(self) -> None:
        for name in (
            "run_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.sh",
            "launch_brev_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.sh",
            "wait_and_run_scheme_b_simd_full_t103_depth10_diag_digits3_ring65536.sh",
        ):
            path = ROOT / name
            self.assertTrue(path.stat().st_mode & 0o111, name)


if __name__ == "__main__":
    unittest.main()
