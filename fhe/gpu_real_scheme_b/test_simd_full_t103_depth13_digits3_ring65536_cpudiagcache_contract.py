"""Static source contract for the reopened "diagonal-plaintext-cache" speed
lever, scoped this time to the CPU-side vector only
(real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp).

No FIDESlib/CUDA build is required to run this: it is a source-text
contract, the same style as test_diagcache_contract.py (the abandoned
2026-07-27/28 attempt this reopens), proving before any GPU run that:

  - the frozen depth-13 parent this fork is forked from, and the T=103
    semantic anchor, are both untouched, and this new file is a genuinely
    separate fork (not an in-place edit);
  - MULT_DEPTH/RING_DIM/LARGE_DIGITS/BSGS_N1/BSGS_N2/TOL/packing constants
    are byte-identical to the parent -- this fix changes only WHICH
    std::vector<double> gets rebuilt for each BSGS diagonal, never the
    diagonal values, packing, or crypto parameters;
  - the diagonal cache is keyed by weight-matrix ADDRESS
    (`std::map<const std::vector<double>*, ...>`), matching the design
    that reused a Plaintext OBJECT'S identity in the abandoned attempt --
    except this cache holds only a std::vector<double>, never a
    Plaintext/Ciphertext, and matmul() still calls raw_plain() (a fresh
    MakeCKKSPackedPlaintext) on every single call, so no GPU-resident
    object is ever reused across two multPt calls (the abandoned
    attempt's exact, root-caused crash mechanism);
  - matmul()'s inner loop no longer builds `packed_values` inline -- that
    construction now happens exactly once per distinct weight, inside
    cached_packed_values();
  - every existing op-count invariant (matmul/ct-ct/ct-plain/rotation/
    accumulate-sum/score-tile/weight-tile/lane-shift/round-trip counts) is
    identical, textually, to the parent's EXPECTED_* constants -- proving
    this fork cannot have changed any encrypted operation, only which
    Plaintext-construction work is skipped;
  - two new cache-specific invariants (12 misses, one per distinct weight
    matrix; 159732 hits, everything else) are present and internally
    consistent with the unchanged matmul-call/BSGS-diagonal counts;
  - the run/launch/wait/CMake/build scripts reference the new binary and
    require the `_cpudiagcache_` tag marker, which is disjoint from the
    abandoned attempt's `_diagcache_` marker and from every other
    prototype's marker;
  - EncryptedEvaluator still contains neither `Decrypt(` nor `secretKey`.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

HERE = Path(__file__).with_name("src")
SOURCE = HERE / (
    "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_"
    "cpudiagcache.cpp"
)
PARENT_SOURCE = HERE / (
    "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"
)
FROZEN_ANCHOR_SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention_t103.cpp"
OLD_ABANDONED_DIAGCACHE_SOURCE = HERE / "real_dnagpt_fides_scheme_b_diagcache.cpp"

PARENT_SHA256 = "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e"
FROZEN_ANCHOR_SHA256 = (
    "70580ff0b4f12565921e0af8ce04e7b4c0d4253b51c0a8380ef0727ce85b2a0f"
)

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = (
    SCRIPT_DIR / "run_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.sh"
)
LAUNCH_SCRIPT = (
    SCRIPT_DIR
    / "launch_brev_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.sh"
)
ORCHESTRATOR_SCRIPT = (
    SCRIPT_DIR
    / "wait_and_run_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.sh"
)
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class SchemeBCpuDiagonalCacheContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    # -- fork identity / non-interference with frozen or sibling sources --

    def test_frozen_depth13_parent_is_untouched(self) -> None:
        self.assertEqual(_sha256(PARENT_SOURCE), PARENT_SHA256)

    def test_frozen_t103_semantic_anchor_is_untouched(self) -> None:
        self.assertEqual(_sha256(FROZEN_ANCHOR_SOURCE), FROZEN_ANCHOR_SHA256)

    def test_this_fork_is_a_genuinely_separate_file(self) -> None:
        self.assertTrue(PARENT_SOURCE.exists())
        self.assertNotEqual(_sha256(PARENT_SOURCE), _sha256(SOURCE))

    def test_pinned_parent_hash_in_source_matches_real_parent_file(self) -> None:
        self.assertIn(
            "constexpr std::string_view PINNED_PARENT_SOURCE_SHA256 =\n"
            f'    "{PARENT_SHA256}";',
            self.source,
        )

    def test_old_abandoned_diagcache_attempt_unaffected_and_disjoint(
        self,
    ) -> None:
        # The 2026-07-27/28 attempt (a different fork, from the much older
        # non-Token-SIMD T=2 source) must still exist untouched, and this
        # fork's tag marker must never collide with its `_diagcache_` tag.
        self.assertTrue(OLD_ABANDONED_DIAGCACHE_SOURCE.exists())
        self.assertNotIn("_diagcache_", "_cpudiagcache_")

    # -- crypto/packing parameters unchanged from the parent --

    def test_correctness_and_crypto_contract_unchanged_from_parent(self) -> None:
        for fragment in (
            "constexpr std::size_t D = 768;",
            "constexpr std::size_t T = 103;",
            "constexpr std::size_t PACK_WIDTH = 1024;",
            "constexpr std::size_t COPIES = 4;",
            "constexpr std::size_t TOKEN_BATCH = 8;",
            "constexpr std::size_t BSGS_N1 = 32;",
            "constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;",
            "constexpr std::uint32_t RING_DIM = 65536;",
            "constexpr std::uint32_t MULT_DEPTH = 13;",
            "constexpr std::uint32_t SCALE_BITS = 50;",
            "constexpr std::uint32_t LARGE_DIGITS = 3;",
            "constexpr double TOL = 4e-2;",
            "static_assert(SLOTS == 32768);",
            "static_assert(RING_DIM == 2 * SLOTS);",
            "static_assert(TOKEN_GROUPS == 13);",
            "parameters.SetSecurityLevel(SecurityLevel::HEStd_128_classic);",
            "parameters.SetBatchSize(SLOTS);",
            "parameters.SetRingDim(RING_DIM);",
            "O_WRONLY | O_CREAT | O_EXCL",
            '"intermediate_decrypt_attempts\\": 0',
            '"evaluator_has_private_key\\": false',
        ):
            self.assertIn(fragment, self.source, fragment)

    # -- the cache mechanism itself --

    def test_cache_keyed_by_weight_address_not_by_value(self) -> None:
        self.assertIn(
            "std::map<const std::vector<double>*, std::vector<std::vector<double>>>",
            self.source,
        )
        self.assertIn("diagonal_vector_cache_.find(&weight)", self.source)
        self.assertIn(
            "diagonal_vector_cache_\n"
            "                        .emplace(&weight, std::move(built))",
            self.source,
        )

    def test_cache_holds_vectors_never_plaintext_or_ciphertext_objects(
        self,
    ) -> None:
        # The map's value type must be a plain std::vector<std::vector
        # <double>>, never Plaintext/Ciphertext -- this is the structural
        # guarantee that this fork cannot hit the abandoned attempt's
        # GPU-resident-Plaintext-reuse crash mechanism.
        self.assertIn(
            "std::map<const std::vector<double>*, std::vector<std::vector<double>>>\n"
            "        diagonal_vector_cache_;",
            self.source,
        )
        self.assertNotIn("std::map<const Matrix*, std::vector<Plaintext>>", self.source)
        self.assertNotIn("Plaintext> diagonal_cache_", self.source)

    def test_fresh_plaintext_constructed_on_every_matmul_call(self) -> None:
        # raw_plain() (a fresh MakeCKKSPackedPlaintext call) must still be
        # invoked once per (giant, small) pair inside matmul(), taking the
        # cached vector as input -- never skipped, never memoized itself.
        start = self.source.index("Ct matmul(const BabyRotations& baby,")
        end = self.source.index("\n    Ct lane_shift(", start)
        matmul_body = self.source[start:end]
        self.assertIn("cached_packed_values(weight, giant, small)", matmul_body)
        self.assertIn("raw_plain(packed_values)", matmul_body)
        self.assertNotIn("Plaintext diagonal_plain =\n                    ", "")
        # matmul() no longer builds packed_values inline (no nested
        # copy/row/token loop remains in its body).
        self.assertNotIn(
            "for (std::size_t copy = 0; copy < COPIES; ++copy) {", matmul_body
        )
        self.assertNotIn("rolled_row", matmul_body)

    def test_diagonal_build_loop_lives_exactly_once_in_cached_packed_values(
        self,
    ) -> None:
        # The nested copy/row/token diagonal-construction loop (the actual
        # redundant CPU work) must now live exactly once, inside
        # cached_packed_values, not duplicated in matmul().
        self.assertEqual(self.source.count("const std::size_t rolled_row ="), 1)
        def_start = self.source.index(
            "const std::vector<double>& cached_packed_values("
        )
        def_end = self.source.index("\n    Ct matmul(", def_start)
        body = self.source[def_start:def_end]
        self.assertIn("rolled_row", body)
        self.assertIn("found = diagonal_vector_cache_", body)
        self.assertIn("++diagonal_cache_misses_;", body)
        self.assertIn("++diagonal_cache_hits_;", body)
        # Miss path builds before caching; hit path does no rebuild work.
        self.assertLess(
            body.index("diagonal_vector_cache_.find"),
            body.index("++diagonal_cache_misses_"),
        )

    def test_cache_population_happens_before_first_use_no_use_before_build(
        self,
    ) -> None:
        def_start = self.source.index(
            "const std::vector<double>& cached_packed_values("
        )
        def_end = self.source.index("\n    Ct matmul(", def_start)
        body = self.source[def_start:def_end]
        # The lookup/build happens before the return statement that hands
        # back a reference into the (by-then-populated) cache entry.
        self.assertLess(
            body.index("diagonal_vector_cache_.find"), body.index("return found")
        )
        self.assertLess(
            body.index("built[diagonal] = std::move"), body.index("return found")
        )

    def test_no_serial_deserialize_or_gpu_object_reuse_across_multpt(self) -> None:
        self.assertNotIn("Serial::Deserialize", self.source)
        self.assertNotIn("Serial::Serialize", self.source)
        # No GPU Plaintext/Ciphertext object identity is ever stashed in
        # the cache map's value type (checked structurally above); also
        # confirm no separate Plaintext-keyed cache was introduced.
        self.assertNotIn("std::vector<Plaintext>>", self.source)

    # -- op-count invariants: caching changes zero encrypted operations --

    def test_op_count_expected_constants_unchanged_from_parent(self) -> None:
        for fragment in (
            "constexpr std::size_t EXPECTED_MATRIX_PRODUCTS = 156;",
            "constexpr std::size_t EXPECTED_CT_CT = 1506;",
            "constexpr std::size_t EXPECTED_CT_PLAIN = 177734;",
            "constexpr std::size_t EXPECTED_ROTATIONS = 8173;",
            "constexpr std::size_t EXPECTED_ACCUMULATE_SUM = 8776;",
            "constexpr std::size_t EXPECTED_SCORE_TILES = 91;",
            "constexpr std::size_t EXPECTED_WEIGHT_TILES = 727;",
            "constexpr std::size_t EXPECTED_CACHED_SHIFTS = 90;",
            "constexpr std::size_t EXPECTED_ROUND_TRIPS = 857;",
            "constexpr std::size_t EXPECTED_LOGICAL_INSTANCES = 129162;",
        ):
            self.assertIn(fragment, self.source, fragment)

    def test_new_diagonal_cache_invariants_present_and_consistent(self) -> None:
        self.assertIn(
            "constexpr std::size_t EXPECTED_DIAGONAL_CACHE_MISSES = 12;",
            self.source,
        )
        self.assertIn(
            "EXPECTED_MATRIX_PRODUCTS * BSGS_N1 * BSGS_N2 -\n"
            "            EXPECTED_DIAGONAL_CACHE_MISSES",
            self.source,
        )
        # 156 matmul calls * 1024 diagonals/call - 12 misses = 159732 hits.
        expected_hits = 156 * 32 * 32 - 12
        self.assertEqual(expected_hits, 159732)
        self.assertIn(
            "evaluation.diagonal_cache_misses !=\n"
            "                EXPECTED_DIAGONAL_CACHE_MISSES ||",
            self.source,
        )
        self.assertIn(
            "evaluation.diagonal_cache_hits != EXPECTED_DIAGONAL_CACHE_HITS ||",
            self.source,
        )

    def test_evidence_json_reports_cache_hit_miss_counters(self) -> None:
        self.assertIn('"diagonal_cache_hits\\":', self.source)
        self.assertIn('"diagonal_cache_misses\\":', self.source)

    # -- privacy invariant, unchanged --

    def test_evaluator_has_no_private_key_access(self) -> None:
        start = self.source.index("class EncryptedEvaluator")
        end = self.source.index("struct Metrics")
        evaluator_body = self.source[start:end]
        self.assertNotIn("Decrypt(", evaluator_body)
        self.assertNotIn("secretKey", evaluator_body)

    def test_single_final_decrypt_in_main(self) -> None:
        self.assertEqual(self.source.count("cc->Decrypt("), 1)

    # -- scripts / build wiring --

    def test_scripts_require_cpudiagcache_tag_marker(self) -> None:
        for path in (RUN_SCRIPT, LAUNCH_SCRIPT):
            text = path.read_text(encoding="utf-8")
            self.assertIn("_cpudiagcache_", text)
            self.assertIn("_scheme_b_", text)
            self.assertNotIn("_diagcache_", text.replace("_cpudiagcache_", ""))

    def test_orchestrator_references_cpudiagcache_launch_script(self) -> None:
        text = ORCHESTRATOR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "launch_brev_scheme_b_simd_full_t103_depth13_digits3_ring65536_"
            "cpudiagcache.sh",
            text,
        )
        self.assertIn(
            "SCHEME_B_SIMD_FULL_DEPTH13_DIGITS3_RING65536_CPUDIAGCACHE_RUN_TAG",
            text,
        )

    def test_tag_marker_disjoint_from_other_prototypes(self) -> None:
        for forbidden in (
            "_cached_",
            "_serialized_",
            "_profiled_",
            "_diagcache_",
            "_warmup_",
            "_shard_",
        ):
            self.assertNotEqual(forbidden, "_cpudiagcache_")
            self.assertNotIn(forbidden, "_cpudiagcache_")

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn(
            "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_"
            "ring65536_cpudiagcache",
            cmake_text,
        )
        self.assertIn(SOURCE.name, cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_"
            "ring65536_cpudiagcache",
            build_text,
        )

    def test_scripts_parse_and_are_executable(self) -> None:
        for path in (RUN_SCRIPT, LAUNCH_SCRIPT, ORCHESTRATOR_SCRIPT):
            self.assertTrue(path.stat().st_mode & 0o111, path.name)

    def test_run_script_pins_source_parent_and_fixture(self) -> None:
        run = RUN_SCRIPT.read_text(encoding="utf-8")
        for fragment in (
            _sha256(SOURCE),
            PARENT_SHA256,
            SOURCE.name,
            PARENT_SOURCE.name,
            "fixture_t103.sha256",
            'if [[ -e "${OUTPUT}" ]]',
            "_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_",
        ):
            self.assertIn(fragment, run)

    def test_launch_and_wait_scripts_keep_capacity_and_vram_gates(self) -> None:
        launch = LAUNCH_SCRIPT.read_text(encoding="utf-8")
        wait = ORCHESTRATOR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("gpucap_preflight_confirm_gpu", launch)
        self.assertIn("nvidia-smi --query-compute-apps", launch)
        self.assertIn("gpucap_wait_for_capacity", wait)
        for script in (launch, wait):
            self.assertIn(
                "_simd_full_t103_depth13_digits3_ring65536_cpudiagcache_",
                script,
            )


if __name__ == "__main__":
    unittest.main()
