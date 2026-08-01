"""Contract for the additive T=103 depth-10/digits-3/ring-65536 fork.

Parent is the depth-9/digits-3/ring-65536 fork, which fixed the depth-8
vector-underflow crash (got past all 91 causal score tiles) but then failed
differently: OpenFHE's Decode() raised "approximation error is too high"
while decrypting the last/deepest score tile (q=12 k=12) at the client
boundary (Client::reduce_score_tile) -- see docs/hybrid/tasks.md
(2026-07-30). This fork changes only
MULT_DEPTH (9 -> 10) to give that one worst-case ciphertext one more level
of remaining precision; it does not patch the frozen FIDESlib library.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).parent
SOURCE = ROOT / "src"
BASE = SOURCE / "real_dnagpt_fides_scheme_b_simd_full_t103_depth9_digits3_ring65536.cpp"
DEPTH10 = (
    SOURCE / "real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536.cpp"
)

BASE_SHA256 = "738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235"
DEPTH10_SHA256 = "703a72d8dcc76cbe93234e04dcc953cce7daf0b48b64011ac1239129333d5ca2"

REPLACEMENTS = (
    (
        "// Additive depth-9/digits-3/ring-65536 complete Token-SIMD block-0 gate",
        "// Additive depth-10/digits-3/ring-65536 complete Token-SIMD block-0 gate",
    ),
    (
        "//   direct parent: depth-8/digits-3/ring-65536 prototype (real-evaluation\n"
        "//   crash, not context-rejected: see Reason below)\n"
        "//   Reason: the depth-8/digits-3/ring-65536 parent crashed inside the\n"
        "//   linked FIDESlib library (unguarded level underflow in\n"
        "//   RNSPoly::rescale/Ciphertext::rescale) at the final causal\n"
        "//   score tile, whose partial-group mask multiply consumes one more\n"
        "//   level than MULT_DEPTH=8 budgets. This fork only changes\n"
        "//   MULT_DEPTH to 9 for that one extra level of headroom; it does not\n"
        "//   patch the frozen FIDESlib library.",
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
    ),
    (
        "//     65cc818cbb299af984cccdbe6a7df891cac5b7071f663edfac8c3f13be161365",
        "//     738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235",
    ),
    (
        "constexpr std::uint32_t MULT_DEPTH = 9;",
        "constexpr std::uint32_t MULT_DEPTH = 10;",
    ),
    (
        '    "65cc818cbb299af984cccdbe6a7df891cac5b7071f663edfac8c3f13be161365";',
        '    "738da0c0c32bd14407ee14877925c6fe7e45fe17d6863f402d46f6bdf98ae235";',
    ),
    (
        '            << "Token-SIMD T=103 depth-9/digits-3/ring-65536 complete "\n'
        '               "block-0 gate\\n";',
        '            << "Token-SIMD T=103 depth-10/digits-3/ring-65536 complete "\n'
        '               "block-0 gate\\n";',
    ),
    (
        '           "depth-9/digits-3/ring-65536 complete "',
        '           "depth-10/digits-3/ring-65536 complete "',
    ),
    (
        '"\\"t103-scheme-b-token-simd-full-depth9-digits3-ring65536-v1\\",\\n"',
        '"\\"t103-scheme-b-token-simd-full-depth10-digits3-ring65536-v1\\",\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH9_DIGITS3_RING65536_GATE_PASS\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH10_DIGITS3_RING65536_GATE_PASS\\n"',
    ),
    (
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH9_DIGITS3_RING65536_GATE_FAIL\\n"',
        '"REAL_DNAGPT_TOKEN_SIMD_FULL_DEPTH10_DIGITS3_RING65536_GATE_FAIL\\n"',
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TokenSimdFullT103Depth10Digits3Ring65536ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = BASE.read_text(encoding="utf-8")
        cls.depth10 = DEPTH10.read_text(encoding="utf-8")

    def test_parent_and_fork_hashes_are_pinned(self) -> None:
        self.assertEqual(sha256(BASE), BASE_SHA256)
        self.assertEqual(sha256(DEPTH10), DEPTH10_SHA256)

    def test_fork_changes_only_declared_identity_and_depth_fields(self) -> None:
        expected = self.base
        for old, new in REPLACEMENTS:
            self.assertEqual(expected.count(old), 1, old)
            expected = expected.replace(old, new)
        self.assertEqual(self.depth10, expected)

    def test_depth_is_explicit_and_other_crypto_fields_are_unchanged(
        self,
    ) -> None:
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
        ):
            self.assertIn(fragment, self.depth10)

    def test_build_target_is_additive(self) -> None:
        target = "real_dnagpt_fides_scheme_b_simd_full_t103_depth10_digits3_ring65536"
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        self.assertIn(target, cmake)
        self.assertIn(target, build)
        self.assertIn(DEPTH10.name, cmake)

    def test_run_script_pins_source_parent_and_fixture(self) -> None:
        run = (
            ROOT / "run_scheme_b_simd_full_t103_depth10_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            DEPTH10_SHA256,
            BASE_SHA256,
            DEPTH10.name,
            BASE.name,
            "fixture_t103.sha256",
            'if [[ -e "${OUTPUT}" ]]',
            "_simd_full_t103_depth10_digits3_ring65536_",
        ):
            self.assertIn(fragment, run)

    def test_launch_and_wait_scripts_keep_capacity_and_vram_gates(self) -> None:
        launch = (
            ROOT / "launch_brev_scheme_b_simd_full_t103_depth10_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        wait = (
            ROOT / "wait_and_run_scheme_b_simd_full_t103_depth10_digits3_ring65536.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("gpucap_preflight_confirm_gpu", launch)
        self.assertIn("nvidia-smi --query-compute-apps", launch)
        self.assertIn("gpucap_wait_for_capacity", wait)
        self.assertIn("SCHEME_B_SIMD_FULL_DEPTH10_DIGITS3_RING65536_RUN_TAG", wait)
        for script in (launch, wait):
            self.assertIn("_simd_full_t103_depth10_digits3_ring65536_", script)
            self.assertNotIn(
                '"run_scheme_b_simd_full_t103_depth9_digits3_ring65536.sh"',
                script,
            )
            self.assertNotIn(
                '"launch_brev_scheme_b_simd_full_t103_depth9_digits3_ring65536.sh"',
                script,
            )

    def test_scripts_parse_and_are_executable(self) -> None:
        for name in (
            "run_scheme_b_simd_full_t103_depth10_digits3_ring65536.sh",
            "launch_brev_scheme_b_simd_full_t103_depth10_digits3_ring65536.sh",
            "wait_and_run_scheme_b_simd_full_t103_depth10_digits3_ring65536.sh",
        ):
            path = ROOT / name
            self.assertTrue(path.stat().st_mode & 0o111, name)


if __name__ == "__main__":
    unittest.main()
