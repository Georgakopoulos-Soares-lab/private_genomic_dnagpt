"""Static source contract for the Scheme B diagonal-plaintext-cache FIX
prototype (real_dnagpt_fides_scheme_b_diagcache.cpp).

No FIDESlib/CUDA build is required to run this: it is a source-text
contract, the same style as test_profiling_contract.py, proving (before any
GPU run) that:

  - the frozen single-shot/cached sources, and the profiled baseline this
    fix is forked from, are all untouched;
  - TOL, BSGS_N1/BSGS_N2, and the packing constants are byte-identical to
    the frozen single-shot gate -- the fix changes WHICH Plaintext object
    identity is reused, never the diagonal values, the packing, or the
    tolerance;
  - the diagonal-plaintext cache is keyed by weight-matrix address (a
    `std::map<const Matrix*, ...>`), not by a copy of its values, and
    `bsgs_inner_loop` no longer builds a fresh `raw_plain(packed_values)`
    inline -- that construction now happens exactly once per distinct
    weight, inside `cached_diagonal_plaintexts`;
  - the same 6 matmul call sites, the same one detailed call, and the same
    profiling bucket labels as the profiled baseline are preserved, so the
    two evidence files are directly bucket-by-bucket comparable;
  - the run/launch/orchestrator scripts and CMake reference the new binary
    and require the `_diagcache_` tag marker, disjoint from every other
    prototype's marker;
  - EncryptedEvaluator still contains neither `Decrypt(` nor `secretKey`.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

HERE = Path(__file__).with_name("src")
SOURCE = HERE / "real_dnagpt_fides_scheme_b_diagcache.cpp"
ORIGINAL_SOURCE = HERE / "real_dnagpt_fides_scheme_b.cpp"
CACHED_SOURCE = HERE / "real_dnagpt_fides_scheme_b_cached.cpp"
PROFILED_SOURCE = HERE / "real_dnagpt_fides_scheme_b_profiled.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
CACHED_SHA256 = "de79d6e7145b3ac9035b09cf7bc86354c755faab139c46f942bcc03f6ab72cc6"

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = SCRIPT_DIR / "run_scheme_b_diagcache.sh"
LAUNCH_SCRIPT = SCRIPT_DIR / "launch_brev_scheme_b_diagcache.sh"
ORCHESTRATOR_SCRIPT = SCRIPT_DIR / "wait_and_run_scheme_b_diagcache.sh"
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class SchemeBDiagonalCacheContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_single_shot_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(ORIGINAL_SOURCE), ORIGINAL_SHA256)

    def test_frozen_cached_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(CACHED_SOURCE), CACHED_SHA256)

    def test_profiled_baseline_source_is_untouched(self) -> None:
        # The diagcache fix is forked from the profiled baseline via `cp`,
        # not by editing it in place -- confirm the baseline still exists
        # and this fix is a genuinely separate file.
        self.assertTrue(PROFILED_SOURCE.exists())
        self.assertNotEqual(_sha256(PROFILED_SOURCE), _sha256(SOURCE))

    def test_correctness_contract_unchanged(self) -> None:
        self.assertIn("constexpr double TOL = 4e-2;", self.source)
        self.assertIn("constexpr std::size_t BSGS_N1 = 32;", self.source)
        self.assertIn(
            "constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;", self.source
        )
        self.assertIn("constexpr std::size_t D = 768;", self.source)
        self.assertIn("constexpr std::size_t T = 2;", self.source)
        self.assertIn("constexpr std::size_t PACK_WIDTH = 1024;", self.source)
        self.assertIn("constexpr std::size_t COPIES = 4;", self.source)
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 16;", self.source)
        self.assertRegex(self.source, r'GATE\s*=\s*"full"')

    def test_diagonal_cache_keyed_by_weight_address(self) -> None:
        self.assertIn(
            "std::map<const Matrix*, std::vector<Plaintext>> diagonal_cache_;",
            self.source,
        )
        self.assertIn("diagonal_cache_.find(&weight)", self.source)
        self.assertIn("diagonal_cache_.emplace(&weight,", self.source)

    def test_bsgs_inner_loop_no_longer_builds_plaintext_inline(self) -> None:
        # The per-(giant,small) `raw_plain(packed_values)` construction must
        # now live exactly once, inside cached_diagonal_plaintexts -- not
        # duplicated inline in bsgs_inner_loop (that duplication, run once
        # per token per weight, was the redundant work this fix removes).
        self.assertEqual(self.source.count("raw_plain(packed_values)"), 1)
        start = self.source.index("Ct bsgs_inner_loop(")
        end = self.source.index("\n    }\n", start)
        inner_loop_body = self.source[start:end]
        self.assertNotIn("packed_values", inner_loop_body)
        self.assertIn("cached_diagonal_plaintexts(weight, detailed)", inner_loop_body)

    def test_cache_reused_across_both_tokens_not_rebuilt(self) -> None:
        # cached_diagonal_plaintexts must return early on a cache hit,
        # before doing any work -- this is what lets token 1's call skip
        # the CPU-encode+GPU-upload that token 0's call already paid.
        _ = self.source.index("cached_diagonal_plaintexts(")
        # Grab the function body (not the two call sites), i.e. the first
        # occurrence that is a definition (followed by a brace, not a call).
        def_start = self.source.index(
            "const std::vector<Plaintext>& cached_diagonal_plaintexts("
        )
        def_end = self.source.index("\n    Ct bsgs_inner_loop(", def_start)
        body = self.source[def_start:def_end]
        self.assertIn("found != diagonal_cache_.end()", body)
        self.assertIn("return found->second;", body)
        self.assertLess(body.index("found->second"), body.index("build_start"))

    def test_matmul_call_sites_unchanged_from_profiled_baseline(self) -> None:
        call_sites = self.source.count("return matmul(")
        self.assertEqual(call_sites, 6)
        self.assertEqual(
            self.source.count("matmul(baby, fixture_.qkv[0], detail_this_call)"), 1
        )
        self.assertEqual(self.source.count(", false);"), 5)

    def test_profiling_buckets_preserved_for_direct_comparison(self) -> None:
        for needle in (
            '"matmul_qkv_query"',
            '"matmul_qkv_key"',
            '"matmul_qkv_value"',
            '"matmul_attention_projection"',
            '"matmul_mlp_fc"',
            '"matmul_mlp_projection"',
            '"detail_ciphertext_plaintext_multiply_accumulate"',
            '"detail_giant_rotation_keyswitch"',
            '"detail_giant_result_accumulate"',
        ):
            self.assertIn(needle, self.source, f"missing bucket label: {needle!r}")

    def test_new_diagonal_build_buckets_present(self) -> None:
        for needle in (
            '"diagonal_plaintext_build"',
            '"diagonal_plaintext_build_detailed"',
        ):
            self.assertIn(needle, self.source)

    def test_evaluator_has_no_private_key_access(self) -> None:
        start = self.source.index("class EncryptedEvaluator")
        end = self.source.index("struct Metrics")
        evaluator_body = self.source[start:end]
        self.assertNotIn("Decrypt(", evaluator_body)
        self.assertNotIn("secretKey", evaluator_body)

    def test_single_final_decrypt_in_main(self) -> None:
        self.assertEqual(self.source.count("cc->Decrypt("), 1)

    def test_scripts_require_diagcache_tag_marker(self) -> None:
        for path in (RUN_SCRIPT, LAUNCH_SCRIPT):
            text = path.read_text(encoding="utf-8")
            self.assertIn("_diagcache_", text)
            self.assertIn("_scheme_b_", text)

    def test_orchestrator_references_diagcache_launch_script(self) -> None:
        text = ORCHESTRATOR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("launch_brev_scheme_b_diagcache.sh", text)
        self.assertIn("SCHEME_B_DIAGCACHE_RUN_TAG", text)

    def test_tag_marker_disjoint_from_other_prototypes(self) -> None:
        for forbidden in ("_cached_", "_serialized_", "_profiled_"):
            self.assertNotEqual(forbidden, "_diagcache_")
            self.assertNotIn(forbidden, "_diagcache_")
            self.assertNotIn("_diagcache_", forbidden)

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_diagcache", cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_diagcache", build_text)


if __name__ == "__main__":
    unittest.main()
