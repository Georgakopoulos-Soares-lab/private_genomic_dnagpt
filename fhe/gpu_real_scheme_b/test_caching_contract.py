"""Static source contract for the Scheme B in-process context/key caching
prototype (real_dnagpt_fides_scheme_b_cached.cpp).

No FIDESlib/CUDA build is required to run this: it is a source-text
contract, the same style as test_batching_contract.py, proving (before any
GPU run) that:

  - context/key setup (GenCryptoContext/KeyGen/EvalMultKeyGen/
    EvalRotateKeyGen/LoadContext) happens exactly once, textually outside
    any loop;
  - rotation keys are requested exactly once, for the covering "full" gate,
    before that one-time setup -- not recomputed per iteration;
  - a loop constructs a fresh Client/EncryptedEvaluator and calls
    evaluate() every iteration (the concrete "no state leakage between
    repeated evaluations" property: each iteration's counters start at
    zero by construction, not by an explicit reset);
  - --repeats < 2 is rejected (this binary exists to prove reuse across
    more than one evaluation, so a single-shot invocation is refused);
  - the evidence JSON schema records the one-time setup and per-iteration
    results separately, plus an explicit honesty label;
  - EncryptedEvaluator still contains neither a decrypt call nor secret-key
    access (same invariant as the single-shot gate).
"""

from __future__ import annotations

import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("src") / "real_dnagpt_fides_scheme_b_cached.cpp"
ORIGINAL_SOURCE = Path(__file__).with_name("src") / "real_dnagpt_fides_scheme_b.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class SchemeBCachingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_original_frozen_source_is_untouched(self) -> None:
        self.assertTrue(
            ORIGINAL_SOURCE.exists(),
            "the frozen single-shot Scheme B source must still exist",
        )
        self.assertEqual(
            _sha256(ORIGINAL_SOURCE),
            ORIGINAL_SHA256,
            "the frozen, hash-pinned single-shot gate source must not change: "
            "its SHA-256 is recorded as provenance in existing immutable "
            "evidence JSONs",
        )

    def test_setup_calls_occur_exactly_once(self) -> None:
        for needle in (
            "GenCryptoContext(",
            "->KeyGen()",
            "EvalMultKeyGen(",
            "EvalRotateKeyGen(",
            "LoadContext(",
        ):
            self.assertEqual(
                self.source.count(needle),
                1,
                f"{needle!r} must occur exactly once (one-time setup, "
                "not per iteration)",
            )

    def test_rotation_keys_requested_once_for_the_covering_gate(self) -> None:
        # One function definition + exactly one call site == not recomputed
        # per iteration.
        self.assertEqual(self.source.count("required_rotation_keys("), 2)
        self.assertRegex(self.source, r'GATE\s*=\s*"full"')

        setup_index = self.source.index("const std::vector<int> rotation_keys =")
        keygen_index = self.source.index("cc->EvalRotateKeyGen(")
        loop_index = self.source.index(
            "for (std::size_t iteration = 0; iteration < options.repeats"
        )
        self.assertLess(
            setup_index,
            keygen_index,
            "rotation keys must be computed before EvalRotateKeyGen",
        )
        self.assertLess(
            keygen_index,
            loop_index,
            "rotation-key generation must happen before the repeat loop, "
            "since FIDESlib throws if EvalRotateKeyGen runs after LoadContext",
        )

    def test_loop_constructs_fresh_client_and_evaluator_each_iteration(self) -> None:
        loop_start = self.source.index(
            "for (std::size_t iteration = 0; iteration < options.repeats"
        )
        # main() has one loop; slice from the loop to the evidence-writing
        # call that follows it.
        loop_end = self.source.index("const std::string evidence = make_json_cached(")
        self.assertLess(loop_start, loop_end)
        loop_body = self.source[loop_start:loop_end]

        self.assertIn("Client client(cc, keys);", loop_body)
        self.assertIn("EncryptedEvaluator evaluator(cc, fixture, client);", loop_body)
        self.assertIn("evaluator.evaluate(", loop_body)
        self.assertIn("iterations.push_back", loop_body)

        # Exactly one textual construction site each -- reused at runtime by
        # the surrounding loop, not duplicated per gate/iteration in source.
        self.assertEqual(self.source.count("Client client(cc, keys);"), 1)
        self.assertEqual(
            self.source.count("EncryptedEvaluator evaluator(cc, fixture, client);"), 1
        )

    def test_repeats_below_two_is_rejected(self) -> None:
        self.assertIn("options.repeats < 2", self.source)
        self.assertIn("must be at least 2", self.source)

    def test_encrypted_evaluator_has_no_decrypt_or_secret_key(self) -> None:
        evaluator = self.source.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]
        self.assertNotIn("Decrypt(", evaluator)
        self.assertNotIn("secretKey", evaluator)

    def test_evidence_schema_separates_setup_from_iterations(self) -> None:
        writer = self.source.split("make_json_cached(", 1)[1]
        # The JSON is built as an escaped C++ string literal, so the quotes
        # around each key appear in the source as literal backslash-quote
        # pairs (\"key\").
        self.assertRegex(writer, r'\\"setup\\":\s*\{')
        self.assertIn("context_keygen_load_seconds", writer)
        self.assertRegex(writer, r'\\"iterations\\":\s*\[')
        self.assertRegex(writer, r'\\"label\\":\s*\\"')
        self.assertIn("not evidence of distinct multi-block correctness", writer)
        self.assertIn("not evidence of cross-process (serialized) caching", writer)

    def test_output_immutability_and_pin_guards_unchanged(self) -> None:
        self.assertIn("O_WRONLY | O_CREAT | O_EXCL", self.source)
        self.assertIn('"786c7600fb2f16b724e0acf73df367b27b8afed6"', self.source)
        self.assertIn(
            '"8d20a2841ee29a7a386171f1bd173b7189144359ce8f91ea5eb21cc60c0a78fe"',
            self.source,
        )

    def test_cmake_and_run_script_reference_the_new_binary(self) -> None:
        cmake = Path(__file__).with_name("CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_cached", cmake)

        run_script = (
            Path(__file__)
            .with_name("run_scheme_b_cached.sh")
            .read_text(encoding="utf-8")
        )
        self.assertIn("_scheme_b_", run_script)
        self.assertIn("_cached_", run_script)
        self.assertRegex(run_script, r"REPEATS.*<\s*2")


if __name__ == "__main__":
    unittest.main()
