"""Static/structural contract for the additive T=103 12-block+head
Token-SIMD CONFIG-3 "all-optimizations" driver (CPU-side diagonal-vector
cache variant).

Mirrors ``test_all_blocks_head_t103_simd_source_contract.py`` exactly, but
for the cpudiagcache end-to-end driver, which is an additive fork of the
config-1 e2e driver that #includes the ..._cpudiagcache per-block source
instead of the plain depth-13 block. Because the cpudiagcache block is a
strict superset of the depth-13 block (same EncryptedEvaluator/Client/
Fixture/evaluate() symbol surface, same op/boundary counts, differing only
by an internal class-private host-side vector cache), the composition logic
is byte-for-byte identical to the config-1 e2e driver; this test pins that.

This file does not require FIDESlib/CUDA (it reads and greps source text,
not compiles it) and passes on the Mac dev machine.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DRIVER = (
    SRC / "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache.cpp"
)
CPUDIAGCACHE_PARENT = (
    SRC
    / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp"
)
E2E_FORK_PARENT = SRC / "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head.cpp"
TWO_BLOCK_REFRESH_PARENT = SRC / "real_dnagpt_fides_scheme_b_two_block_refresh.cpp"

DRIVER_SHA256 = "afb8d4bae5fdd0f157bfdde1fdfdd6ac8981806fdbb74177f03e924dd2a70c48"
CPUDIAGCACHE_PARENT_SHA256 = (
    "686c8b766436fde0fd4422f6a29f80361f8c8a2565768577857e9264b2976fd5"
)
E2E_FORK_PARENT_SHA256 = (
    "355f1857bec77471192f3d81010f88b478d9b3707f63ebe5898526f99791a5da"
)
TWO_BLOCK_REFRESH_PARENT_SHA256 = (
    "5cb82f81dc46efe2f4bfd7a1c9e9c1bf90cea3d629432b027b42e5df89093767"
)
FIXTURE_MANIFEST_SHA256 = (
    "ab55533eaf8779b64a67c5cdd8ac67152419389e33f1c44addd4a092e0ff962d"
)
FIXTURE_CONTRACT_SHA256 = (
    "5256d3f59e215631f9ded7dba68f13e16c12c8fde927c53277bc1752171f7cca"
)

TARGET_NAME = "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache"

# The per-block source this driver #includes exactly once.
PER_BLOCK_INCLUDE = '#include "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp"'
MAIN_RENAME = "#define main real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_main_unused"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DRIVER.read_text(encoding="utf-8")

    def test_pinned_hashes(self) -> None:
        self.assertEqual(sha256(DRIVER), DRIVER_SHA256)
        self.assertEqual(sha256(CPUDIAGCACHE_PARENT), CPUDIAGCACHE_PARENT_SHA256)
        self.assertEqual(sha256(E2E_FORK_PARENT), E2E_FORK_PARENT_SHA256)
        self.assertEqual(
            sha256(TWO_BLOCK_REFRESH_PARENT), TWO_BLOCK_REFRESH_PARENT_SHA256
        )

    def test_pinned_identities_appear_in_source(self) -> None:
        for fragment in (
            CPUDIAGCACHE_PARENT_SHA256,
            E2E_FORK_PARENT_SHA256,
            TWO_BLOCK_REFRESH_PARENT_SHA256,
            FIXTURE_MANIFEST_SHA256,
            FIXTURE_CONTRACT_SHA256,
        ):
            self.assertIn(fragment, self.source)

    def test_includes_cpudiagcache_per_block_source_exactly_once_unedited(self) -> None:
        self.assertEqual(self.source.count(PER_BLOCK_INCLUDE), 1)
        # It must NOT include the plain (non-cpudiagcache) depth-13 block.
        self.assertEqual(
            self.source.count(
                '#include "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"'
            ),
            0,
        )
        # The included source's own main() must be renamed away, never left
        # to collide with this file's main().
        self.assertIn(MAIN_RENAME, self.source)
        self.assertIn("#undef main", self.source)

    def test_swaps_only_the_per_block_source_from_the_config1_e2e_parent(self) -> None:
        # The cpudiagcache source-sha CLI flag/pin replaces the depth13 one;
        # no depth13-source token may survive.
        self.assertIn("PINNED_CPUDIAGCACHE_SOURCE_SHA256", self.source)
        self.assertIn("--cpudiagcache-source-sha256", self.source)
        self.assertNotIn("PINNED_DEPTH13_SOURCE_SHA256", self.source)
        self.assertNotIn("--depth13-source-sha256", self.source)
        # New PASS/FAIL markers.
        self.assertIn(
            "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_PASS",
            self.source,
        )
        self.assertIn(
            "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_FAIL",
            self.source,
        )
        # The old SIMD marker must not linger.
        self.assertNotIn(
            "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_SIMD_PASS", self.source
        )


class OneContextTwelveBlocksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DRIVER.read_text(encoding="utf-8")

    def test_exactly_one_shared_crypto_context(self) -> None:
        # GenCryptoContext appears once in THIS file's own text (the
        # included per-block source's own call site is inside a renamed,
        # never-invoked main() and is irrelevant to the composed graph).
        self.assertEqual(self.source.count("GenCryptoContext(parameters)"), 1)
        self.assertEqual(self.source.count("SetDevices({options.gpu})"), 1)
        self.assertEqual(self.source.count("Client client(cc, keys);"), 1)

    def test_exactly_one_twelve_block_loop(self) -> None:
        self.assertEqual(
            self.source.count(
                "for (std::size_t block = 0; block < ALL_BLOCKS; ++block)"
            ),
            1,
        )
        self.assertIn("constexpr std::size_t ALL_BLOCKS = 12;", self.source)
        self.assertEqual(
            self.source.count("EncryptedEvaluator evaluator(cc, fixture, client)"), 1
        )
        self.assertEqual(
            self.source.count("evaluator.evaluate(encrypted_inputs, active_counts)"), 1
        )

    def test_per_block_weight_selection_uses_the_loop_variable(self) -> None:
        # load_block_fixture must be called with the loop variable `block`,
        # not a hardcoded index, and must build a per-block filename prefix.
        self.assertEqual(
            self.source.count("load_block_fixture(options.fixture_dir, block)"), 1
        )
        self.assertIn(
            'const std::string prefix = "weights__block" + std::to_string(block_index) + "__";',
            self.source,
        )
        self.assertIn(
            'directory /\n            ("oracle__block" + std::to_string(block_index) + "__block_output.bin")'.replace(
                "\n            ", " "
            ),
            self.source.replace("\n            ", " "),
        )

    def test_exactly_eleven_inter_block_refreshes_and_one_head_input_refresh(
        self,
    ) -> None:
        self.assertIn(
            "constexpr std::size_t INTER_BLOCK_REFRESHES = ALL_BLOCKS - 1;", self.source
        )
        self.assertIn("constexpr std::size_t HEAD_INPUT_REFRESHES = 1;", self.source)
        # One call site for each refresh kind, invoked conditionally inside
        # the single 12-iteration loop -- 11 executions of the full-hidden
        # branch (blocks 0..10) and exactly 1 execution of the last-token
        # branch (block 11), not 11/1 duplicated call sites.
        self.assertEqual(
            self.source.count("refresh_all_hidden_token_simd("), 2
        )  # def + call
        self.assertEqual(
            self.source.count("refresh_last_token_for_head("), 2
        )  # def + call
        self.assertEqual(self.source.count("if (block + 1 < ALL_BLOCKS)"), 1)
        self.assertIn("constexpr std::size_t EXPECTED_TOTAL_CROSSINGS =", self.source)
        self.assertIn("static_assert(EXPECTED_TOTAL_CROSSINGS == 10299);", self.source)
        self.assertIn(
            "static_assert(EXPECTED_TOTAL_LOGICAL_INSTANCES == 1551081);", self.source
        )

    def test_refresh_decrypts_all_token_groups_and_re_encrypts_fresh_level_zero(
        self,
    ) -> None:
        self.assertIn(
            "std::vector<double> decrypt_token_simd_groups(Cc cc, Keys& keys,",
            self.source,
        )
        self.assertIn("if (groups.size() != TOKEN_GROUPS) {", self.source)
        self.assertIn(
            "fresh Token-SIMD group encryption did not return level 0", self.source
        )
        self.assertIn("head-input refresh did not return level 0", self.source)

    def test_head_applied_exactly_once_after_block_eleven(self) -> None:
        self.assertEqual(self.source.count("EncryptedHeadEvaluator head_evaluator("), 1)
        self.assertEqual(self.source.count("head_evaluator.evaluate(head_input)"), 1)
        # The head evaluator construction/call must appear textually AFTER
        # the 12-block loop closes, not inside it.
        loop_start = self.source.index(
            "for (std::size_t block = 0; block < ALL_BLOCKS; ++block)"
        )
        head_call = self.source.index("EncryptedHeadEvaluator head_evaluator(")
        self.assertLess(loop_start, head_call)
        # And the else-branch (block 11 only) is what feeds it.
        else_branch = self.source.index("} else {", loop_start)
        self.assertLess(else_branch, head_call)

    def test_last_token_position_is_pinned(self) -> None:
        self.assertIn("constexpr std::size_t LAST_TOKEN = T - 1;", self.source)
        self.assertIn(
            "constexpr std::size_t LAST_TOKEN_GROUP = LAST_TOKEN / TOKEN_BATCH;",
            self.source,
        )
        self.assertIn(
            "constexpr std::size_t LAST_TOKEN_LANE = LAST_TOKEN % TOKEN_BATCH;",
            self.source,
        )
        self.assertIn(
            "static_assert(LAST_TOKEN_GROUP == TOKEN_GROUPS - 1);", self.source
        )


class NoPrematureDecryptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DRIVER.read_text(encoding="utf-8")

    def test_decrypt_call_sites_are_exactly_the_three_declared_boundaries(self) -> None:
        # 1) decrypt_token_simd_groups (shared by both refresh kinds)
        # 2) silu_boundary (the one head crossing with no Client method)
        # 3) the single final margin decrypt in main()
        self.assertEqual(self.source.count("->Decrypt("), 3)
        self.assertIn("cc->Decrypt(keys.secretKey, local, &decoded);", self.source)
        self.assertIn("cc_->Decrypt(keys_.secretKey, local, &plaintext);", self.source)
        self.assertIn(
            "cc->Decrypt(keys.secretKey, head.encrypted_margin, &decoded_margin);",
            self.source,
        )

    def test_no_secret_key_reaches_the_evaluator_classes(self) -> None:
        # EncryptedEvaluator is entirely reused from the included per-block
        # parent (never redefined here); this file's own new classes/functions
        # must only touch secretKey inside the three declared boundary
        # call sites checked above, never inside matmul/layernorm/BSGS
        # helper code.
        head_class = self.source.split("class EncryptedHeadEvaluator", 1)[1].split(
            "struct BlockSummary", 1
        )[0]
        self.assertEqual(head_class.count("secretKey"), 1)  # only silu_boundary
        self.assertIn("silu_boundary(const Ct& ciphertext) {", head_class)

    def test_output_is_written_exclusively_and_never_overwritten(self) -> None:
        self.assertIn("write_exclusive(options.output, evidence);", self.source)
        self.assertIn("refusing to overwrite immutable evidence: ", self.source)

    def test_declared_privacy_fields_are_honest(self) -> None:
        self.assertIn('"intermediate_decrypt_attempts\\": 0', self.source)
        self.assertIn('"evaluator_has_private_key\\": false', self.source)


class ProgressLoggingTests(unittest.TestCase):
    """This run may take many hours; a partial failure must be diagnosable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DRIVER.read_text(encoding="utf-8")

    def test_stage_logging_after_every_block(self) -> None:
        self.assertIn('"[stage] block=" << block << " begin\\n"', self.source)
        self.assertIn(
            '"[stage] block=" << block\n                      << " evaluation done evaluation_seconds=" << block_seconds'.replace(
                "\n                      ", " "
            ),
            self.source.replace("\n                      ", " "),
        )
        self.assertIn('"[stage] block=" << block << " end\\n"', self.source)

    def test_boundary_logging_at_every_refresh(self) -> None:
        self.assertIn('"[boundary] refresh block=" << block << "->"', self.source)
        self.assertIn('"[boundary] head-input refresh begin\\n"', self.source)
        self.assertIn(
            '"[boundary] head-input refresh done global_rel_inf="', self.source
        )

    def test_head_stage_logging(self) -> None:
        self.assertIn('"[stage] head begin\\n"', self.source)
        self.assertIn('"[stage] head end evaluation_seconds="', self.source)
        self.assertIn(
            '"[stage] final decrypt done decrypted_margin_n_minus_a="', self.source
        )


class BuildWiringTests(unittest.TestCase):
    def test_cmake_and_build_scripts_reference_the_new_target(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        binaries = (
            ROOT.parent.parent / "kimon" / "env" / "build_binaries.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(TARGET_NAME, cmake)
        self.assertIn(DRIVER.name, cmake)
        self.assertIn(TARGET_NAME, build)
        self.assertIn(TARGET_NAME, binaries)

    def test_build_wiring_does_not_touch_concurrent_tracks(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        build = (ROOT / "build_in_fideslib.sh").read_text(encoding="utf-8")
        # The config-1 e2e driver and the standalone per-block cpudiagcache
        # target and the sharded tracks own these substrings; this driver
        # must add alongside, never drop them from either build entry point.
        for reserved in (
            "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head ",
            "shard_writer",
            "shard_reader",
        ):
            self.assertIn(reserved, cmake)  # still present in CMake
            self.assertIn(reserved, build)  # still present in build script
            # and this new target's own name must not collide with them
            self.assertNotIn(reserved, TARGET_NAME)


if __name__ == "__main__":
    unittest.main()
