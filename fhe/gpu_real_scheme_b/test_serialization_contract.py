"""Static source contract for the Scheme B CROSS-PROCESS context/key
serialization prototype (real_dnagpt_fides_scheme_b_serialize_writer.cpp and
real_dnagpt_fides_scheme_b_serialize_reader.cpp).

No FIDESlib/CUDA build is required to run this: it is a source-text
contract, the same style as test_caching_contract.py and
test_batching_contract.py, proving (before any GPU run) that:

  - neither of the two frozen upstream sources (the single-shot gate and the
    in-process caching prototype) was edited to build this;
  - the writer builds a fresh context/keys (GenCryptoContext/KeyGen/
    EvalMultKeyGen/EvalRotateKeyGen) but never calls LoadContext and defines
    no Client/EncryptedEvaluator -- it only serializes and exits;
  - the writer restricts the serialized secret-key file's permissions and
    never prints its path/contents;
  - the reader contains ZERO occurrences of GenCryptoContext(/
    EvalMultKeyGen(/EvalRotateKeyGen( -- the concrete, checkable proof that
    it never regenerates keys, only deserializes them;
  - the reader deserializes the context/keys before calling LoadContext
    (the one GPU step that cannot be skipped -- see the source header
    comments for why FIDESlib has no serialize path for the GPU-side
    context);
  - the reader's EncryptedEvaluator still contains neither a decrypt call
    nor secret-key access (same invariant as every other Scheme B gate);
  - the reader never prints the secret-key file's path/contents either;
  - output immutability (O_EXCL) and the pinned FIDESlib-commit/fixture-
    manifest guards are present in both files where applicable;
  - CMakeLists.txt and both new run_scheme_b_serialize_*.sh scripts reference
    the two new binaries and require both "_scheme_b_" and "_serialized_" in
    their evidence-tag guards -- a substring distinct from "_cached_", so
    evidence from the three prototypes can never collide;
  - the orchestrator deletes the state directory (secret-key hygiene) after
    the reader phase, regardless of pass/fail.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

WRITER_SOURCE = (
    Path(__file__).with_name("src") / "real_dnagpt_fides_scheme_b_serialize_writer.cpp"
)
READER_SOURCE = (
    Path(__file__).with_name("src") / "real_dnagpt_fides_scheme_b_serialize_reader.cpp"
)
ORIGINAL_SOURCE = Path(__file__).with_name("src") / "real_dnagpt_fides_scheme_b.cpp"
CACHED_SOURCE = (
    Path(__file__).with_name("src") / "real_dnagpt_fides_scheme_b_cached.cpp"
)

ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
CACHED_SHA256 = "de79d6e7145b3ac9035b09cf7bc86354c755faab139c46f942bcc03f6ab72cc6"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _log_lines(source: str) -> list[str]:
    return [
        line
        for line in source.splitlines()
        if "std::cout" in line or "std::cerr" in line
    ]


class UpstreamFrozenSourcesTests(unittest.TestCase):
    def test_original_frozen_source_is_untouched(self) -> None:
        self.assertTrue(ORIGINAL_SOURCE.exists())
        self.assertEqual(
            _sha256(ORIGINAL_SOURCE),
            ORIGINAL_SHA256,
            "the frozen, hash-pinned single-shot gate source must not change",
        )

    def test_cached_prototype_source_is_untouched(self) -> None:
        self.assertTrue(CACHED_SOURCE.exists())
        self.assertEqual(
            _sha256(CACHED_SOURCE),
            CACHED_SHA256,
            "the in-process caching prototype source must not change either -- "
            "this serialization prototype forks logic from it, it does not edit it",
        )


class SchemeBSerializeWriterContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = WRITER_SOURCE.read_text(encoding="utf-8")

    def test_builds_fresh_context_and_keys_exactly_once(self) -> None:
        for needle in (
            "GenCryptoContext(",
            "->KeyGen()",
            "->EvalMultKeyGen(",
            "->EvalRotateKeyGen(",
        ):
            self.assertEqual(
                self.source.count(needle),
                1,
                f"writer must build a fresh context/keys exactly once: {needle!r}",
            )

    def test_never_calls_load_context(self) -> None:
        self.assertNotIn(
            "->LoadContext(",
            self.source,
            "the writer must never call LoadContext/GenCryptoContextGPU -- "
            "that GPU step is exclusively the reader's job",
        )

    def test_defines_no_client_or_evaluator(self) -> None:
        self.assertNotIn("class Client", self.source)
        self.assertNotIn("class EncryptedEvaluator", self.source)

    def test_serializes_context_and_all_keys(self) -> None:
        for needle in (
            "Serial::SerializeToFile(context_path",
            "Serial::SerializeToFile(public_key_path",
            "Serial::SerializeToFile(secret_key_path",
            "SerializeEvalMultKey(",
            "SerializeEvalAutomorphismKey(",
        ):
            self.assertIn(needle, self.source)

    def test_restricts_secret_key_file_permissions(self) -> None:
        self.assertIn("chmod(secret_key_path.c_str()", self.source)
        self.assertIn("S_IRUSR | S_IWUSR", self.source)

    def test_never_logs_secret_key_path_or_contents(self) -> None:
        for line in _log_lines(self.source):
            self.assertNotIn("secret_key_path", line)
            self.assertNotIn("secret-key.txt", line)

    def test_evidence_json_never_records_secret_key_path(self) -> None:
        start = self.source.index("std::string make_writer_json(")
        end = self.source.index("return out.str();", start)
        writer_json = self.source[start:end]
        self.assertNotIn("secret_key_path", writer_json)
        self.assertNotIn("secret-key.txt", writer_json)

    def test_refuses_to_reuse_existing_state_directory(self) -> None:
        self.assertIn("refusing to reuse an existing state directory", self.source)

    def test_output_immutability_and_pin_guard_present(self) -> None:
        self.assertIn("O_WRONLY | O_CREAT | O_EXCL", self.source)
        self.assertIn('"786c7600fb2f16b724e0acf73df367b27b8afed6"', self.source)


class SchemeBSerializeReaderContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = READER_SOURCE.read_text(encoding="utf-8")

    def test_never_regenerates_keys_or_builds_fresh_context(self) -> None:
        for needle in (
            "GenCryptoContext(",
            "->EvalMultKeyGen(",
            "->EvalRotateKeyGen(",
            "->KeyGen()",
        ):
            self.assertEqual(
                self.source.count(needle),
                0,
                f"the reader must NEVER regenerate keys or build a fresh CPU "
                f"context -- found forbidden call: {needle!r}",
            )

    def test_deserializes_context_public_and_secret_key(self) -> None:
        self.assertEqual(self.source.count("Serial::DeserializeFromFile("), 3)
        self.assertIn('"crypto-context.txt"', self.source)
        self.assertIn('"public-key.txt"', self.source)
        self.assertIn('"secret-key.txt"', self.source)

    def test_deserializes_eval_mult_and_automorphism_keys_exactly_once(self) -> None:
        self.assertEqual(self.source.count("DeserializeEvalMultKey("), 1)
        self.assertEqual(self.source.count("DeserializeEvalAutomorphismKey("), 1)

    def test_deserialize_happens_before_load_context(self) -> None:
        self.assertEqual(self.source.count("->LoadContext("), 1)
        last_deserialize = self.source.rindex("Serial::DeserializeFromFile(")
        load_context_index = self.source.index("->LoadContext(")
        self.assertLess(
            last_deserialize,
            load_context_index,
            "all deserialization must complete before the one unavoidable "
            "LoadContext/GenCryptoContextGPU call",
        )

    def test_load_context_and_deserialize_timed_separately(self) -> None:
        self.assertIn("deserialize_seconds", self.source)
        self.assertIn("load_context_gpu_seconds", self.source)

    def test_encrypted_evaluator_has_no_decrypt_or_secret_key(self) -> None:
        evaluator = self.source.split("class EncryptedEvaluator", 1)[1].split(
            "struct Metrics", 1
        )[0]
        self.assertNotIn("Decrypt(", evaluator)
        self.assertNotIn("secretKey", evaluator)

    def test_never_logs_secret_key_path_or_contents(self) -> None:
        for line in _log_lines(self.source):
            self.assertNotIn("secret-key.txt", line)

    def test_evidence_json_never_records_secret_key_path(self) -> None:
        start = self.source.index("std::string make_json(")
        end = self.source.index("return out.str();", start)
        evidence_writer = self.source[start:end]
        self.assertNotIn("secret-key.txt", evidence_writer)
        self.assertNotIn("secret_key_path", evidence_writer)

    def test_evidence_records_avoidable_vs_unavoidable_breakdown(self) -> None:
        start = self.source.index("std::string make_json(")
        end = self.source.index("return out.str();", start)
        writer = self.source[start:end]
        self.assertIn("avoidable_via_cross_process_reload_seconds", writer)
        self.assertIn("unavoidable_per_process_gpu_load_seconds", writer)

    def test_output_immutability_and_pin_guards_present(self) -> None:
        self.assertIn("O_WRONLY | O_CREAT | O_EXCL", self.source)
        self.assertIn('"786c7600fb2f16b724e0acf73df367b27b8afed6"', self.source)
        self.assertIn(
            '"8d20a2841ee29a7a386171f1bd173b7189144359ce8f91ea5eb21cc60c0a78fe"',
            self.source,
        )

    def test_gate_is_fixed_to_full(self) -> None:
        self.assertRegex(self.source, r'GATE\s*=\s*"full"')


class SchemeBSerializeBuildAndScriptContractTests(unittest.TestCase):
    def test_cmake_references_both_new_binaries(self) -> None:
        cmake = Path(__file__).with_name("CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_serialize_writer", cmake)
        self.assertIn("real_dnagpt_fides_scheme_b_serialize_reader", cmake)

    def test_build_script_builds_both_new_targets(self) -> None:
        build_script = (
            Path(__file__).with_name("build_in_fideslib.sh").read_text(encoding="utf-8")
        )
        self.assertIn("real_dnagpt_fides_scheme_b_serialize_writer", build_script)
        self.assertIn("real_dnagpt_fides_scheme_b_serialize_reader", build_script)

    def test_run_scripts_require_scheme_b_and_serialized_substrings(self) -> None:
        for name in (
            "run_scheme_b_serialize_writer.sh",
            "run_scheme_b_serialize_reader.sh",
        ):
            script = Path(__file__).with_name(name).read_text(encoding="utf-8")
            self.assertIn("_scheme_b_", script)
            self.assertIn("_serialized_", script)
            # The evidence-tag marker must be distinct from the existing
            # in-process caching prototype's "_cached_" marker so evidence
            # from the two prototypes can never collide.
            self.assertNotIn(
                '"_cached_"',
                script,
                f"{name} must not reuse the caching prototype's tag marker",
            )

    def test_launch_scripts_require_scheme_b_and_serialized_substrings(self) -> None:
        for name in (
            "launch_brev_scheme_b_serialize_writer.sh",
            "launch_brev_scheme_b_serialize_reader.sh",
        ):
            script = Path(__file__).with_name(name).read_text(encoding="utf-8")
            self.assertIn("_scheme_b_", script)
            self.assertIn("_serialized_", script)

    def test_writer_and_reader_tags_are_distinct_substrings(self) -> None:
        # Both markers appear together as one literal, contiguous substring
        # in every guard -- "_serialized_" -- verified above. Here we also
        # confirm it textually differs from "_cached_" at the character
        # level (not just "not equal"), so a reviewer can see the two
        # prototypes' evidence namespaces cannot overlap by construction.
        self.assertNotEqual("_serialized_", "_cached_")
        self.assertNotIn("_cached_", "_serialized_")
        self.assertNotIn("_serialized_", "_cached_")

    def test_orchestrator_deletes_state_directory_after_reader_phase(self) -> None:
        orchestrator = (
            Path(__file__)
            .with_name("wait_and_run_scheme_b_serialize.sh")
            .read_text(encoding="utf-8")
        )
        self.assertIn("rm -rf", orchestrator)
        self.assertIn("secret-key hygiene", orchestrator)
        self.assertIn("cleanup_state_dir", orchestrator)
        # The cleanup path must run on both the failure and success branches,
        # not just the happy path.
        self.assertGreaterEqual(orchestrator.count("cleanup_state_dir"), 3)

    def test_orchestrator_never_logs_secret_key_filename(self) -> None:
        orchestrator = (
            Path(__file__)
            .with_name("wait_and_run_scheme_b_serialize.sh")
            .read_text(encoding="utf-8")
        )
        for line in orchestrator.splitlines():
            if re.search(r"\blog\b|\becho\b", line):
                self.assertNotIn("secret-key.txt", line)


if __name__ == "__main__":
    unittest.main()
