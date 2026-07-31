"""Contract for the additive T=103 depth-13/digits-3/ring-65536 fork.

Parent is the depth-10/digits-3/ring-65536 fork, which got past all 91
causal score tiles, attention projection, residual, and the LN2 client
boundary, then hit its own require_remaining_depth() guard: "insufficient
depth before Token-SIMD MLP group 0". A diagnostic-only sibling fork
(depth10_diag, MULT_DEPTH unchanged, one added print) revealed the exact
level at that point: normalized2->GetLevel() == 9 -- see docs/hybrid/tasks.md
and docs/hybrid/simd_current_state.txt (2026-07-31). The guard requires
4 <= MULT_DEPTH - level, so the minimum passing MULT_DEPTH is 9 + 4 = 13.
This fork changes only MULT_DEPTH to 13; it does not patch the frozen
FIDESlib library.
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
DEPTH13 = (
    SOURCE / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"
)

BASE_SHA256 = "703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2"
DEPTH13_SHA256 = "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e"

REPLACEMENTS = (
    (
        "// Additive depth-10/digits-3/ring-65536 complete Token-SIMD block-0 gate",
        "// Additive depth-13/digits-3/ring-65536 complete Token-SIMD block-0 gate",
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
        '//   require_remaining_depth() guard: "insufficient depth before\n'
        '//   Token-SIMD MLP group 0". A diagnostic-only sibling fork\n'
        "//   (depth10_diag, MULT_DEPTH unchanged, one added print) revealed the\n"
        "//   exact level at that point: normalized2->GetLevel() == 9. The guard\n"
        "//   requires 4 <= MULT_DEPTH - level, so the minimum passing\n"
        "//   MULT_DEPTH is 9 + 4 = 13. This fork changes only MULT_DEPTH to 13;\n"
        "//   it does not patch the frozen FIDESlib library.",
    ),
    (
        "//     738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235",
        "//     703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2",
    ),
    (
        "constexpr std::uint32_t MULT_DEPTH = 10;",
        "constexpr std::uint32_t MULT_DEPTH = 13;",
    ),
    (
        '    "738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235";',
        '    "703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2";',
    ),
    (
        '            << "Token-SIMD T=103 depth-10/digits-3/ring-65536 complete "\n'
        '               "block-0 gate\\n";',
        '            << "Token-SIMD T=103 depth-13/digits-3/ring-65536 complete "\n'
        '               "block-0 gate\\n";',
    ),
    (
        '           "depth-10/digits-3/ring-65536 complete "',
        '           "depth-13/digits-3/ring-65536 complete "',
    ),
    (
        '"\\"t103-scheme-b-token-simd-full-depth10-digits3-ring65536-v1\\",\\n"',
        '"\\"t103-scheme-b-token-simd-full-depth13-digits3-ring65536-v1\\",\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH10_DIGITS3_RING65536_GATE_PASS\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_GATE_PASS\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH10_DIGITS3_RING65536_GATE_FAIL\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH13_DIGITS3_RING65536_GATE_FAIL\\n"',
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TokenSimdFullT103Depth13Digits3Ring65536ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = BASE.read_text(encoding="utf-8")
        cls.depth13 = DEPTH13.read_text(encoding="utf-8")

    def test_parent_and_fork_hashes_are_pinned(self) -> None:
        self.assertEqual(sha256(BASE), BASE_SHA256)
        self.assertEqual(sha256(DEPTH13), DEPTH13_SHA256)

    def test_fork_changes_only_declared_identity_and_depth_fields(self) -> None:
        expected = self.base
        for old, new in REPLACEMENTS:
            self.assertEqual(expected.count(old), 1, old)
            expected = expected.replace(old, new)
        self.assertEqual(self.depth13, expected)

    def test_depth_is_explicit_and_other_crypto_fields_are_unchanged(
        self,
    ) -> None:
        for fragment in (
            "constexpr std::uint32_t MULT_DEPTH = 13;",
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
            self.assertIn(fragment, self.depth13)

    def test_build_target_is_additive(self) -> None:
        target = "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536"
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        self.assertIn(target, cmake)
        self.assertIn(target, build)
        self.assertIn(DEPTH13.name, cmake)

    def test_run_script_pins_source_parent_and_fixture(self) -> None:
        run = (
            ROOT / "run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            DEPTH13_SHA256,
            BASE_SHA256,
            DEPTH13.name,
            BASE.name,
            "fixture_t103.sha256",
            'if [[ -e "${OUTPUT}" ]]',
            "_simd_full_t103_depth13_digits3_ring65536_",
        ):
            self.assertIn(fragment, run)

    def test_launch_and_wait_scripts_keep_capacity_and_vram_gates(self) -> None:
        launch = (
            ROOT / "launch_brev_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        wait = (
            ROOT / "wait_and_run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("gpucap_preflight_confirm_gpu", launch)
        self.assertIn("nvidia-smi --query-compute-apps", launch)
        self.assertIn("gpucap_wait_for_capacity", wait)
        self.assertIn("SCHEME_B_SIMD_FULL_DEPTH13_DIGITS3_RING65536_RUN_TAG", wait)
        for script in (launch, wait):
            self.assertIn("_simd_full_t103_depth13_digits3_ring65536_", script)
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
            "run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh",
            "launch_brev_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh",
            "wait_and_run_scheme_b_simd_full_t103_depth13_digits3_ring65536.sh",
        ):
            path = ROOT / name
            self.assertTrue(path.stat().st_mode & 0o111, name)


if __name__ == "__main__":
    unittest.main()
