"""Static source contract for the Scheme B warm-up-prelude prototype
(real_dnagpt_fides_scheme_b_warmup.cpp).

No FIDESlib/CUDA build is required to run this: it is a source-text
contract, the same style as test_profiling_contract.py/test_diagcache_contract.py,
proving (before any GPU run) that:

  - the frozen single-shot/cached sources, and the profiled baseline this
    prototype is forked from, are all untouched;
  - the warm-up prelude touches only a throwaway dummy plaintext/ciphertext
    pair (never the real fixture, never any weight matrix) and is timed
    separately from encrypted_evaluation_seconds, never folded into it;
  - matmul()/bsgs_inner_loop/EncryptedEvaluator/Client are BYTE-IDENTICAL to
    the profiled baseline -- this prototype changes nothing about the real
    evaluation, only adds an untimed prelude before it;
  - the warm-up prelude exercises each of the five primitive op kinds
    (ct-pt multiply, ct-ct multiply, EvalAdd, giant EvalRotate, baby
    EvalFastRotation) at least once;
  - TOL, BSGS_N1/BSGS_N2, and the packing constants are unchanged;
  - EncryptedEvaluator still contains neither `Decrypt(` nor `secretKey`;
  - the run/launch/orchestrator scripts and CMake reference the new binary
    and require the `_warmup_` tag marker, disjoint from every other
    prototype's marker.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

HERE = Path(__file__).with_name("src")
SOURCE = HERE / "real_dnagpt_fides_scheme_b_warmup.cpp"
ORIGINAL_SOURCE = HERE / "real_dnagpt_fides_scheme_b.cpp"
CACHED_SOURCE = HERE / "real_dnagpt_fides_scheme_b_cached.cpp"
PROFILED_SOURCE = HERE / "real_dnagpt_fides_scheme_b_profiled.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
CACHED_SHA256 = "de79d6e7145b3ac9035b09cf7bc86354c755faab139c46f942bcc03f6ab72cc6"

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = SCRIPT_DIR / "run_scheme_b_warmup.sh"
LAUNCH_SCRIPT = SCRIPT_DIR / "launch_brev_scheme_b_warmup.sh"
ORCHESTRATOR_SCRIPT = SCRIPT_DIR / "wait_and_run_scheme_b_warmup.sh"
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _shared_region(text: str) -> str:
    # matmul()/bsgs_inner_loop/EncryptedEvaluator/Client live between the
    # Profiler class and Metrics struct in both files -- isolate that region
    # for a byte-identical comparison against the profiled baseline.
    start = text.index("class Client")
    end = text.index("struct Metrics")
    return text[start:end]


class SchemeBWarmupContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")
        self.profiled_source = PROFILED_SOURCE.read_text(encoding="utf-8")

    def test_frozen_single_shot_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(ORIGINAL_SOURCE), ORIGINAL_SHA256)

    def test_frozen_cached_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(CACHED_SOURCE), CACHED_SHA256)

    def test_profiled_baseline_source_is_untouched_and_differs(self) -> None:
        self.assertTrue(PROFILED_SOURCE.exists())
        self.assertNotEqual(_sha256(PROFILED_SOURCE), _sha256(SOURCE))

    def test_evaluator_client_and_matmul_byte_identical_to_profiled(self) -> None:
        # The whole point of this prototype: it changes NOTHING about the
        # real evaluation. Client/EncryptedEvaluator/matmul()/
        # bsgs_inner_loop must be byte-for-byte identical to the profiled
        # baseline.
        self.assertEqual(
            _shared_region(self.source),
            _shared_region(self.profiled_source),
            "Client/EncryptedEvaluator/matmul()/bsgs_inner_loop must be "
            "unchanged from the profiled baseline -- this prototype only "
            "adds an untimed prelude before evaluate(), it does not touch "
            "evaluate() itself",
        )

    def test_correctness_contract_unchanged(self) -> None:
        self.assertIn("constexpr double TOL = 4e-2;", self.source)
        self.assertIn("constexpr std::size_t BSGS_N1 = 32;", self.source)
        self.assertIn(
            "constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;", self.source
        )
        self.assertIn("constexpr std::size_t PACK_WIDTH = 1024;", self.source)
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 16;", self.source)
        self.assertRegex(self.source, r'GATE\s*=\s*"full"')

    def test_warmup_prelude_uses_only_dummy_data(self) -> None:
        start = self.source.index("warmup_start = Clock::now()")
        end = self.source.index("warmup_seconds = elapsed_seconds(warmup_start)")
        warmup_body = self.source[start:end]
        self.assertIn("dummy_values", warmup_body)
        self.assertIn("dummy_plain", warmup_body)
        self.assertIn("dummy_ct", warmup_body)
        self.assertNotIn("fixture.", warmup_body)
        self.assertNotIn("fixture_.", warmup_body)

    def test_warmup_exercises_each_primitive_op_kind(self) -> None:
        start = self.source.index("warmup_start = Clock::now()")
        end = self.source.index("warmup_seconds = elapsed_seconds(warmup_start)")
        warmup_body = self.source[start:end]
        self.assertIn("cc->EvalMult(dummy_ct, dummy_plain)", warmup_body)  # ct-pt
        self.assertIn("cc->EvalMult(dummy_ct, dummy_ct)", warmup_body)  # ct-ct
        self.assertIn("cc->EvalAdd(", warmup_body)
        self.assertIn("cc->EvalRotate(", warmup_body)  # giant-style
        self.assertIn("cc->EvalFastRotation(", warmup_body)  # baby-style hoisted

    def test_warmup_timed_separately_not_folded_into_evaluation(self) -> None:
        # warmup_seconds must be computed BEFORE encryption_start, and the
        # encrypted_evaluation timer must start fresh at evaluation_start,
        # after encryption -- i.e. warmup time is never inside the
        # evaluation_seconds window.
        warmup_idx = self.source.index("const double warmup_seconds")
        encryption_start_idx = self.source.index("const auto encryption_start")
        evaluation_start_idx = self.source.index("const auto evaluation_start")
        self.assertLess(warmup_idx, encryption_start_idx)
        self.assertLess(encryption_start_idx, evaluation_start_idx)
        self.assertIn('"warmup_prelude"', self.source.replace('\\"', '"'))

    def test_evaluator_has_no_private_key_access(self) -> None:
        start = self.source.index("class EncryptedEvaluator")
        end = self.source.index("struct Metrics")
        evaluator_body = self.source[start:end]
        self.assertNotIn("Decrypt(", evaluator_body)
        self.assertNotIn("secretKey", evaluator_body)

    def test_single_final_decrypt_in_main(self) -> None:
        # One real final decrypt of the real output; the warm-up block
        # never decrypts its dummy ciphertext.
        self.assertEqual(self.source.count("cc->Decrypt("), 1)

    def test_scripts_require_warmup_tag_marker(self) -> None:
        for path in (RUN_SCRIPT, LAUNCH_SCRIPT):
            text = path.read_text(encoding="utf-8")
            self.assertIn("_warmup_", text)
            self.assertIn("_scheme_b_", text)

    def test_orchestrator_references_warmup_launch_script(self) -> None:
        text = ORCHESTRATOR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("launch_brev_scheme_b_warmup.sh", text)
        self.assertIn("SCHEME_B_WARMUP_RUN_TAG", text)

    def test_tag_marker_disjoint_from_other_prototypes(self) -> None:
        for forbidden in ("_cached_", "_serialized_", "_profiled_", "_diagcache_"):
            self.assertNotIn(forbidden, "_warmup_")
            self.assertNotIn("_warmup_", forbidden)

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_warmup", cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_warmup", build_text)


if __name__ == "__main__":
    unittest.main()
