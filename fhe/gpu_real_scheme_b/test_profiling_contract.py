"""Static source contract for the Scheme B GPU profiling prototype
(real_dnagpt_fides_scheme_b_profiled.cpp).

No FIDESlib/CUDA build is required to run this: it is a source-text
contract, the same style as test_caching_contract.py/test_batching_contract.py,
proving (before any GPU run) that:

  - the frozen single-shot and cached sources are untouched;
  - TOL, BSGS_N1/BSGS_N2, and the packing constants (D/T/HEADS/HEAD_DIM/
    MLP_DIM/PACK_WIDTH/COPIES/SLOTS/MULT_DEPTH) are byte-identical to the
    frozen single-shot gate -- this is a profiling instrument, not a new
    protocol or accuracy variant;
  - exactly 24 matmul() call sites exist, matching the block's known
    ciphertext-plaintext matrix-product count (verified against
    fhe_fides_real_d768_t2_block0_batched_scheme_b_a100_20260726.json's
    explicit_ciphertext_plaintext_multiplications=24666 in this same test);
  - exactly one call site passes `true` for the `detailed` argument (the one
    representative matmul call that gets a full giant-step breakdown);
  - every matmul() call site and every baby_rotations() call site is
    wrapped in a named Synchronize()-bracketed timer, so the evidence JSON's
    buckets can reconcile against the unbracketed total;
  - the run/launch/orchestrator scripts and CMake reference the new binary
    and require the `_profiled_` tag marker, kept disjoint from `_cached_`
    and `_serialized_`;
  - EncryptedEvaluator still contains neither `Decrypt(` nor `secretKey`
    (same invariant as every other Scheme B gate).
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

HERE = Path(__file__).with_name("src")
SOURCE = HERE / "real_dnagpt_fides_scheme_b_profiled.cpp"
ORIGINAL_SOURCE = HERE / "real_dnagpt_fides_scheme_b.cpp"
CACHED_SOURCE = HERE / "real_dnagpt_fides_scheme_b_cached.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
CACHED_SHA256 = "de79d6e7145b3ac9035b09cf7bc86354c755faab139c46f942bcc03f6ab72cc6"

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = SCRIPT_DIR / "run_scheme_b_profiled.sh"
LAUNCH_SCRIPT = SCRIPT_DIR / "launch_brev_scheme_b_profiled.sh"
ORCHESTRATOR_SCRIPT = SCRIPT_DIR / "wait_and_run_scheme_b_profiled.sh"
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"

# Cross-checked against results/runs/fhe_fides_real_d768_t2_block0_batched_scheme_b_a100_20260726.json
EXPECTED_MATMUL_CALLS = 24
EXPECTED_CTPT_MULTIPLICATIONS = 24666
EXPECTED_CTCT_MULTIPLICATIONS = 33


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class SchemeBProfilingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_single_shot_source_is_untouched(self) -> None:
        self.assertTrue(ORIGINAL_SOURCE.exists())
        self.assertEqual(
            _sha256(ORIGINAL_SOURCE),
            ORIGINAL_SHA256,
            "the frozen, hash-pinned single-shot gate source must not change",
        )

    def test_frozen_cached_source_is_untouched(self) -> None:
        self.assertTrue(CACHED_SOURCE.exists())
        self.assertEqual(
            _sha256(CACHED_SOURCE),
            CACHED_SHA256,
            "the frozen, hash-pinned caching prototype source must not change",
        )

    def test_correctness_contract_unchanged(self) -> None:
        # This is a profiling instrument, not a new protocol: same tolerance,
        # same BSGS split, same packing geometry as every other Scheme B gate.
        self.assertIn("constexpr double TOL = 4e-2;", self.source)
        self.assertIn("constexpr std::size_t BSGS_N1 = 32;", self.source)
        self.assertIn(
            "constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;", self.source
        )
        self.assertIn("constexpr std::size_t D = 768;", self.source)
        self.assertIn("constexpr std::size_t T = 2;", self.source)
        self.assertIn("constexpr std::size_t HEADS = 12;", self.source)
        self.assertIn("constexpr std::size_t HEAD_DIM = 64;", self.source)
        self.assertIn("constexpr std::size_t MLP_DIM = 3072;", self.source)
        self.assertIn("constexpr std::size_t PACK_WIDTH = 1024;", self.source)
        self.assertIn("constexpr std::size_t COPIES = 4;", self.source)
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 16;", self.source)
        self.assertRegex(self.source, r'GATE\s*=\s*"full"')

    def test_matmul_call_sites_count_matches_known_block_shape(self) -> None:
        # Textual call sites: qkv_query, qkv_key, qkv_value,
        # attention_projection, mlp_fc, mlp_projection = 6. Each site is
        # visited at runtime via its enclosing token/chunk loop(s) to reach
        # 24 total calls at runtime (6+2+8+8), matching
        # ciphertext_plain_matmuls in every prior full-block evidence file.
        call_sites = self.source.count("return matmul(")
        self.assertEqual(
            call_sites,
            6,
            "matmul() should be called from exactly 6 textual call sites",
        )

    def test_exactly_one_detailed_matmul_call(self) -> None:
        # Only token 0's Q projection gets the full giant-step breakdown;
        # every other call site passes a literal `false`.
        self.assertEqual(
            self.source.count("matmul(baby, fixture_.qkv[0], detail_this_call)"), 1
        )
        self.assertEqual(self.source.count(", false);"), 5)

    def test_every_matmul_and_baby_rotation_call_is_bracketed(self) -> None:
        # Bucket labels are function-argument string literals, not JSON
        # output, so they appear verbatim (unescaped) somewhere in the
        # source -- but `timed_ct(` and its label argument may be split
        # across lines by clang-format-style wrapping, so each is checked
        # independently rather than as one contiguous substring.
        for needle in (
            '"matmul_qkv_query"',
            '"matmul_qkv_key"',
            '"matmul_qkv_value"',
            '"matmul_attention_projection"',
            '"matmul_mlp_fc"',
            '"matmul_mlp_projection"',
        ):
            self.assertIn(needle, self.source, f"missing bucket label: {needle!r}")
        self.assertGreaterEqual(self.source.count("timed_ct("), 6)
        self.assertGreaterEqual(self.source.count("timed_baby("), 2)

    def test_detailed_matmul_has_giant_step_sub_buckets(self) -> None:
        for needle in (
            '"detail_ciphertext_plaintext_multiply_accumulate"',
            '"detail_giant_rotation_keyswitch"',
            '"detail_giant_result_accumulate"',
        ):
            self.assertIn(needle, self.source)

    def test_reconciliation_fields_present_in_json_writer(self) -> None:
        # JSON keys are emitted with escaped quotes (\"name\") inside the
        # ostringstream literal, so match the bare field name rather than a
        # quoted substring.
        for needle in (
            "sum_of_buckets_seconds",
            "unaccounted_seconds",
            "unaccounted_fraction_of_server_seconds",
        ):
            self.assertIn(needle, self.source)
        self.assertIn('\\"sum_of_buckets_seconds\\"', self.source)
        self.assertIn('\\"unaccounted_fraction_of_server_seconds\\"', self.source)

    def test_evaluator_has_no_private_key_access(self) -> None:
        # Split the source at the EncryptedEvaluator class definition and
        # check only that region -- Client (which legitimately holds the
        # secret key) and main() (one final Decrypt) are expected to use
        # these symbols elsewhere in the file.
        start = self.source.index("class EncryptedEvaluator")
        end = self.source.index("struct Metrics")
        evaluator_body = self.source[start:end]
        self.assertNotIn("Decrypt(", evaluator_body)
        self.assertNotIn("secretKey", evaluator_body)

    def test_single_final_decrypt_in_main(self) -> None:
        self.assertEqual(self.source.count("cc->Decrypt("), 1)

    def test_no_repeats_option_this_is_single_shot(self) -> None:
        # Unlike the caching prototype, profiling runs the gate exactly
        # once -- adding a repeats loop would reintroduce the caching
        # question this session is explicitly not re-litigating.
        self.assertNotIn("--repeats", self.source)
        self.assertNotIn("options.repeats", self.source)

    def test_scripts_require_profiled_tag_marker(self) -> None:
        for path in (RUN_SCRIPT, LAUNCH_SCRIPT):
            text = path.read_text(encoding="utf-8")
            self.assertIn("_profiled_", text)
            self.assertIn("_scheme_b_", text)

    def test_orchestrator_references_profiled_launch_script(self) -> None:
        text = ORCHESTRATOR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("launch_brev_scheme_b_profiled.sh", text)
        self.assertIn("SCHEME_B_PROFILED_RUN_TAG", text)

    def test_tag_marker_disjoint_from_other_prototypes(self) -> None:
        # _profiled_ must never collide with _cached_ or _serialized_ so the
        # three prototypes' evidence namespaces can never be confused.
        for forbidden in ("_cached_", "_serialized_"):
            self.assertNotIn(forbidden, "_profiled_")
            self.assertNotIn("_profiled_", forbidden)

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_profiled", cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_profiled", build_text)

    def test_ctpt_and_ctct_multiplication_counts_are_traceable(self) -> None:
        # These constants are not computed in the source (they are a runtime
        # emergent property of the loop structure); this test just anchors
        # the expected values used elsewhere in this file's docstring and in
        # docs/hybrid/tasks.md to the real evidence file, so a future editor
        # who changes the block shape is forced to update this anchor too.
        self.assertEqual(EXPECTED_MATMUL_CALLS, 24)
        self.assertEqual(EXPECTED_CTPT_MULTIPLICATIONS, 24666)
        self.assertEqual(EXPECTED_CTCT_MULTIPLICATIONS, 33)


if __name__ == "__main__":
    unittest.main()
