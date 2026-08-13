"""Static/structural contract for the additive T=103 12-block+head
Token-SIMD CONFIG-3 "all-optimizations" driver, T123_V3 per-block lever.

Mirrors ``test_all_blocks_head_t103_cpudiagcache_source_contract.py`` exactly,
but for the T123_V3 end-to-end driver, which is an additive fork of the
config-3 cpudiagcache e2e driver that #includes the ..._cpudiagcache_t123_v3
per-block source instead of the plain cpudiagcache block, plus one addition
in its own main(): the same post-LoadContext ClearEvalMultKeys()/
ClearEvalAutomorphismKeys() call T123_V3 makes in its own (unused) main().
Because the T123_V3 block is a strict superset of the plain cpudiagcache
block (same EncryptedEvaluator/Client/Fixture/evaluate() symbol surface, same
op/boundary counts, differing only by an internal, class-private encoded-
Plaintext template cache flushed at each intra-block stage boundary and
implicitly at every block boundary because a fresh EncryptedEvaluator is
constructed per block), the composition logic is byte-for-byte identical to
the config-3 cpudiagcache e2e driver aside from the one free-host-keys
addition; this test pins that.

This file does not require FIDESlib/CUDA (it reads and greps source text,
not compiles it) and passes on the Mac dev machine / any machine with only
Python 3 available.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DRIVER = (
    SRC
    / "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3.cpp"
)
T123_V3_PARENT = (
    SRC
    / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3.cpp"
)
T123_PARENT = (
    SRC
    / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp"
)
CPUDIAGCACHE_GRANDPARENT = (
    SRC
    / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp"
)
CPUDIAGCACHE_DRIVER_PARENT = (
    SRC / "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache.cpp"
)
TWO_BLOCK_REFRESH_PARENT = SRC / "real_dnagpt_fides_scheme_b_two_block_refresh.cpp"

T123_V3_PARENT_SHA256 = (
    "cbba5b6c2b13e0a9fbe9a6ca1db724782bbf0af0b4651fa17f1586fa6b03534b"
)
CPUDIAGCACHE_DRIVER_PARENT_SHA256 = (
    "588fde54d38bf17b34b676e683e4434589a0d837f59f63a55effcb9a0ee54b2e"
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

TARGET_NAME = (
    "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache_t123_v3"
)

# The per-block source this driver #includes exactly once.
PER_BLOCK_INCLUDE = (
    '#include "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v3.cpp"'
)
MAIN_RENAME = (
    "#define main real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_"
    "ring65536_cpudiagcache_t123_v3_main_unused"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DRIVER.read_text(encoding="utf-8")

    def test_pinned_hashes(self) -> None:
        self.assertEqual(sha256(T123_V3_PARENT), T123_V3_PARENT_SHA256)
        self.assertEqual(
            sha256(CPUDIAGCACHE_DRIVER_PARENT), CPUDIAGCACHE_DRIVER_PARENT_SHA256
        )
        self.assertEqual(
            sha256(TWO_BLOCK_REFRESH_PARENT), TWO_BLOCK_REFRESH_PARENT_SHA256
        )

    def test_pinned_identities_appear_in_source(self) -> None:
        for fragment in (
            T123_V3_PARENT_SHA256,
            CPUDIAGCACHE_DRIVER_PARENT_SHA256,
            TWO_BLOCK_REFRESH_PARENT_SHA256,
            FIXTURE_MANIFEST_SHA256,
            FIXTURE_CONTRACT_SHA256,
        ):
            self.assertIn(fragment, self.source)

    def test_includes_t123_v3_per_block_source_exactly_once_unedited(self) -> None:
        self.assertEqual(self.source.count(PER_BLOCK_INCLUDE), 1)
        # It must NOT include the plain (non-T123_V3) cpudiagcache block, nor
        # the T123 (pre-v3) block, nor the plain depth-13 block, nor the
        # abandoned t123_v2 block.
        for forbidden in (
            '#include "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"',
            '#include "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp"',
            '#include "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123.cpp"',
            '#include "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_t123_v2.cpp"',
        ):
            self.assertEqual(self.source.count(forbidden), 0)
        # The included source's own main() must be renamed away, never left
        # to collide with this file's main().
        self.assertIn(MAIN_RENAME, self.source)
        self.assertIn("#undef main", self.source)

    def test_never_touches_the_abandoned_t123_v2_variant(self) -> None:
        self.assertNotIn("t123_v2", self.source)
        self.assertNotIn("T123_V2", self.source)

    def test_swaps_only_the_per_block_source_from_the_cpudiagcache_driver_parent(
        self,
    ) -> None:
        self.assertIn("PINNED_CPUDIAGCACHE_T123_V3_SOURCE_SHA256", self.source)
        self.assertIn("--cpudiagcache-t123-v3-source-sha256", self.source)
        self.assertIn("PINNED_CPUDIAGCACHE_DRIVER_PARENT_SOURCE_SHA256", self.source)
        self.assertIn("--cpudiagcache-driver-parent-source-sha256", self.source)
        self.assertNotIn("PINNED_CPUDIAGCACHE_SOURCE_SHA256", self.source)
        self.assertNotIn("--cpudiagcache-source-sha256 SHA", self.source)
        self.assertNotIn("PINNED_DEPTH13_SOURCE_SHA256", self.source)
        self.assertNotIn("--depth13-source-sha256", self.source)
        # New PASS/FAIL markers, distinct from the plain-cpudiagcache driver's.
        self.assertIn(
            "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_T123_V3_PASS",
            self.source,
        )
        self.assertIn(
            "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_T123_V3_FAIL",
            self.source,
        )
        # The old (plain cpudiagcache) marker must not linger.
        self.assertNotIn(
            "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_CPUDIAGCACHE_PASS\"",
            self.source,
        )
        self.assertNotIn(
            "REAL_DNAGPT_FIDES_SCHEME_B_ALL_BLOCKS_HEAD_T103_SIMD_PASS", self.source
        )


class FreeHostKeysTests(unittest.TestCase):
    """T123_V3's own contribution beyond the T123 tiers: a one-time,
    global, post-LoadContext free of the CPU-side OpenFHE key maps. Must
    appear EXACTLY once in this driver's own main() (not per block)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = DRIVER.read_text(encoding="utf-8")

    def test_free_host_keys_calls_appear_exactly_once_each(self) -> None:
        self.assertEqual(
            self.source.count(
                "lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalMultKeys();"
            ),
            1,
        )
        self.assertEqual(
            self.source.count(
                "lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalAutomorphismKeys();"
            ),
            1,
        )

    def test_free_host_keys_calls_are_outside_the_block_loop_and_after_load_context(
        self,
    ) -> None:
        load_context = self.source.index("cc->LoadContext(keys.publicKey);")
        clear_mult = self.source.index(
            "lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalMultKeys();"
        )
        clear_rot = self.source.index(
            "lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::ClearEvalAutomorphismKeys();"
        )
        block_loop = self.source.index(
            "for (std::size_t block = 0; block < ALL_BLOCKS; ++block)"
        )
        # Both calls textually follow this driver's own LoadContext call...
        self.assertLess(load_context, clear_mult)
        self.assertLess(load_context, clear_rot)
        # ...and both precede the 12-block loop (one-time setup, not per-block).
        self.assertLess(clear_mult, block_loop)
        self.assertLess(clear_rot, block_loop)

    def test_includes_raw_openfhe_header_for_the_clear_calls(self) -> None:
        # T123_V3's own comment explains this needs OpenFHE's raw header
        # directly; the #include is pulled in transitively via the #include
        # of the T123_V3 per-block source, which this test also checks.
        t123_v3_source = T123_V3_PARENT.read_text(encoding="utf-8")
        self.assertIn("#include <openfhe.h>", t123_v3_source)


class TierCacheCompositionTests(unittest.TestCase):
    """The T123 Tier1/2/3 encoded-Plaintext template cache is a per-
    EncryptedEvaluator-instance member; this driver (like its cpudiagcache
    parent) must construct a FRESH EncryptedEvaluator per block so no cached
    template state (or any other T123-introduced state) crosses a block or a
    client-refresh boundary."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.driver_source = DRIVER.read_text(encoding="utf-8")
        cls.t123_v3_source = T123_V3_PARENT.read_text(encoding="utf-8")

    def test_fresh_evaluator_constructed_inside_the_block_loop(self) -> None:
        self.assertEqual(
            self.driver_source.count(
                "EncryptedEvaluator evaluator(cc, fixture, client);"
            ),
            1,
        )
        block_loop = self.driver_source.index(
            "for (std::size_t block = 0; block < ALL_BLOCKS; ++block)"
        )
        evaluator_ctor = self.driver_source.index(
            "EncryptedEvaluator evaluator(cc, fixture, client);"
        )
        block_end_log = self.driver_source.index(
            '"[stage] block=" << block << " end\\n"'
        )
        self.assertLess(block_loop, evaluator_ctor)
        self.assertLess(evaluator_ctor, block_end_log)

    def test_diagonal_plain_cache_is_a_private_member_flushed_within_evaluate(
        self,
    ) -> None:
        self.assertIn("diagonal_plain_cache_", self.t123_v3_source)
        self.assertIn("void flush_diagonal_plains() { diagonal_plain_cache_.clear(); }", self.t123_v3_source)
        # Flushed at both stage boundaries inside evaluate(), i.e. before the
        # object itself goes out of scope at the end of a block iteration.
        self.assertEqual(self.t123_v3_source.count("flush_diagonal_plains();"), 2)


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
        self.assertEqual(
            self.source.count("load_block_fixture(options.fixture_dir, block)"), 1
        )
        self.assertIn(
            'const std::string prefix = "weights__block" + std::to_string(block_index) + "__";',
            self.source,
        )

    def test_exactly_eleven_inter_block_refreshes_and_one_head_input_refresh(
        self,
    ) -> None:
        self.assertIn(
            "constexpr std::size_t INTER_BLOCK_REFRESHES = ALL_BLOCKS - 1;", self.source
        )
        self.assertIn("constexpr std::size_t HEAD_INPUT_REFRESHES = 1;", self.source)
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

    def test_head_applied_exactly_once_after_block_eleven(self) -> None:
        self.assertEqual(self.source.count("EncryptedHeadEvaluator head_evaluator("), 1)
        self.assertEqual(self.source.count("head_evaluator.evaluate(head_input)"), 1)
        loop_start = self.source.index(
            "for (std::size_t block = 0; block < ALL_BLOCKS; ++block)"
        )
        head_call = self.source.index("EncryptedHeadEvaluator head_evaluator(")
        self.assertLess(loop_start, head_call)
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
        self.assertEqual(self.source.count("->Decrypt("), 3)
        self.assertIn("cc->Decrypt(keys.secretKey, local, &decoded);", self.source)
        self.assertIn("cc_->Decrypt(keys_.secretKey, local, &plaintext);", self.source)
        self.assertIn(
            "cc->Decrypt(keys.secretKey, head.encrypted_margin, &decoded_margin);",
            self.source,
        )

    def test_no_secret_key_reaches_the_evaluator_classes(self) -> None:
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
        self.assertIn('"[stage] block=" << block << " end\\n"', self.source)

    def test_boundary_logging_at_every_refresh(self) -> None:
        self.assertIn('"[boundary] refresh block=" << block << "->"', self.source)
        self.assertIn('"[boundary] head-input refresh begin\\n"', self.source)

    def test_head_stage_logging(self) -> None:
        self.assertIn('"[stage] head begin\\n"', self.source)
        self.assertIn('"[stage] head end evaluation_seconds="', self.source)


class BuildWiringTests(unittest.TestCase):
    def test_cmake_and_build_scripts_reference_the_new_target(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        binaries = (
            ROOT.parent.parent / "kimon" / "env" / "build_binaries.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(TARGET_NAME, cmake)
        self.assertIn(DRIVER.name, cmake)
        self.assertIn(TARGET_NAME, binaries)

    def test_build_wiring_does_not_touch_concurrent_tracks(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        binaries = (
            ROOT.parent.parent / "kimon" / "env" / "build_binaries.sh"
        ).read_text(encoding="utf-8")
        # CMakeLists always follows a target name with a space before the
        # next token (PRIVATE/EXCLUDE_FROM_ALL/newline-indent), so a
        # trailing space distinguishes the bare 12blocks_head target from
        # its _cpudiagcache and _cpudiagcache_t123_v3 siblings.
        for reserved in (
            "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache\n",
            "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head ",
            "shard_writer",
            "shard_reader",
        ):
            self.assertIn(reserved.strip("\n"), cmake)
        # build_binaries.sh lists bare target names one per line (no
        # trailing-space convention), so check line-exact membership there.
        binaries_lines = {line.strip() for line in binaries.splitlines()}
        for reserved_line in (
            "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head",
            "real_dnagpt_fides_scheme_b_simd_full_t103_12blocks_head_cpudiagcache",
            "real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13",
            "real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13",
        ):
            self.assertIn(reserved_line, binaries_lines)

    def test_runner_script_exists_and_references_the_binary(self) -> None:
        runner = (
            ROOT / "run_scheme_b_all_blocks_head_t103_cpudiagcache_t123_v3.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(TARGET_NAME, runner)
        self.assertIn("--cpudiagcache-t123-v3-source-sha256", runner)
        self.assertIn("--cpudiagcache-driver-parent-source-sha256", runner)


if __name__ == "__main__":
    unittest.main()
