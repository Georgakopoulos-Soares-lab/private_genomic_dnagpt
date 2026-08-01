"""Static source contract for the Token-SIMD-parameter-matched cross-process
serialization pair used by the 2-GPU process-per-GPU sharding Stage 1
(Q/K/V split) micro-gate:
  real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13.cpp
  real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp

Per docs/hybrid/tasks.md's 2026-07-31 "2-GPU process-per-GPU sharding" entry,
the existing real_dnagpt_fides_scheme_b_serialize_writer/reader pair is
parameter-incompatible with the Token-SIMD B=8 T=103 layout (SLOTS=4096 vs.
32768, and a rotation-key set sized for the old T=2 gate, not the
TOKEN_BATCH-scaled one). This contract mirrors test_serialization_contract.py's
rigor for the new, Token-SIMD-parameter-matched fork pair, plus new checks
specific to the sharded reader's "--part" flag.

No FIDESlib/CUDA build is required to run this: it is a source-text contract.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

SRC = Path(__file__).with_name("src")

WRITER_SOURCE = SRC / "real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13.cpp"
READER_SOURCE = SRC / "real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp"

# Structural parents: the OLD (parameter-incompatible) serialization prototype
# this pair forks its writer-exits/reader-deserializes-then-LoadContext shape
# from. Neither is edited by this workstream.
OLD_WRITER_SOURCE = SRC / "real_dnagpt_fides_scheme_b_serialize_writer.cpp"
OLD_READER_SOURCE = SRC / "real_dnagpt_fides_scheme_b_serialize_reader.cpp"
OLD_WRITER_SHA256 = "ea720f1b6c49524f1b49853b67622307fd55c9fb751317fee4b30c6750f69403"
OLD_READER_SHA256 = "4dbb003fe8407046c852bf10f0c45b23b579cd1bbc7390099966999b6d6ab5d6"

# Token-SIMD parameter/schedule parent: the passing depth-13/digits-3/
# ring-65536 complete Token-SIMD block gate. Neither the writer nor the
# reader fork edits this file either.
DEPTH13_SOURCE = (
    SRC / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"
)
DEPTH13_SHA256 = "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e"


def _sha256(path: Path) -> str:
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
    """Neither the old (parameter-incompatible) serialization prototype nor
    the passing depth-13 Token-SIMD full-block source may be edited by this
    fork pair -- only forked from."""

    def test_old_writer_is_untouched(self) -> None:
        self.assertTrue(OLD_WRITER_SOURCE.exists())
        self.assertEqual(_sha256(OLD_WRITER_SOURCE), OLD_WRITER_SHA256)

    def test_old_reader_is_untouched(self) -> None:
        self.assertTrue(OLD_READER_SOURCE.exists())
        self.assertEqual(_sha256(OLD_READER_SOURCE), OLD_READER_SHA256)

    def test_depth13_token_simd_source_is_untouched(self) -> None:
        self.assertTrue(DEPTH13_SOURCE.exists())
        self.assertEqual(_sha256(DEPTH13_SOURCE), DEPTH13_SHA256)


class ShardWriterContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = WRITER_SOURCE.read_text(encoding="utf-8")

    def test_pins_both_parent_hashes(self) -> None:
        self.assertIn(OLD_WRITER_SHA256, self.source)
        self.assertIn(DEPTH13_SHA256, self.source)

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
            "that GPU step is exclusively the sharded reader's job",
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

    def test_uses_token_simd_batch_slots_and_depth(self) -> None:
        # SLOTS = COPIES * PACK_WIDTH * TOKEN_BATCH = 4 * 1024 * 8 = 32768,
        # not the old writer's SLOTS = COPIES * PACK_WIDTH = 4096.
        self.assertIn("constexpr std::size_t TOKEN_BATCH = 8;", self.source)
        self.assertIn("static_assert(SLOTS == 32768)", self.source)
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 13;", self.source)
        self.assertIn("constexpr std::uint32_t LARGE_DIGITS = 3;", self.source)
        self.assertIn("constexpr std::uint32_t RING_DIM = 65536;", self.source)
        self.assertIn("SetRingDim(RING_DIM)", self.source)

    def test_required_rotation_keys_is_token_batch_scaled(self) -> None:
        # Distinguishing marker vs. the old writer's
        # required_rotation_keys_for_full(), which has no TOKEN_BATCH scaling
        # anywhere in its body.
        self.assertIn("TOKEN_BATCH * baby", self.source)
        self.assertIn("TOKEN_BATCH * BSGS_N1 * giant", self.source)


class ShardReaderContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = READER_SOURCE.read_text(encoding="utf-8")

    def test_pins_both_parent_hashes(self) -> None:
        self.assertIn(OLD_READER_SHA256, self.source)
        self.assertIn(DEPTH13_SHA256, self.source)

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
                f"the sharded reader must NEVER regenerate keys or build a "
                f"fresh CPU context -- found forbidden call: {needle!r}",
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

    def test_output_immutability_and_pin_guards_present(self) -> None:
        self.assertIn("O_WRONLY | O_CREAT | O_EXCL", self.source)
        self.assertIn('"786c7600fb2f16b724e0acf73df367b27b8afed6"', self.source)

    def test_uses_token_simd_batch_slots_and_depth(self) -> None:
        self.assertIn("constexpr std::size_t TOKEN_BATCH = 8;", self.source)
        self.assertIn("static_assert(SLOTS == 32768)", self.source)
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 13;", self.source)

    def test_loads_all_three_qkv_weight_slices_and_oracle_arrays_unconditionally(
        self,
    ) -> None:
        load_fixture = self.source.split("Fixture load_fixture(", 1)[1].split(
            "\n}\n", 1
        )[0]
        self.assertNotIn("options.parts", load_fixture)
        self.assertNotIn("requested", load_fixture)
        for needle in (
            '"oracle__query.bin"',
            '"oracle__key.bin"',
            '"oracle__value.bin"',
        ):
            self.assertIn(needle, load_fixture)
        # weights__attn_qkv.bin is loaded once (3*D*D) and sliced into all
        # three parts unconditionally, mirroring the frozen depth-13 source.
        self.assertIn('"weights__attn_qkv.bin"', load_fixture)
        self.assertIn("for (std::size_t part = 0; part < 3; ++part)", load_fixture)


class PartFlagContractTests(unittest.TestCase):
    """Checks specific to the sharded reader's new --part flag, verified via
    source structure (not merely asserting good intent)."""

    def setUp(self) -> None:
        self.source = READER_SOURCE.read_text(encoding="utf-8")

    def test_rejects_unknown_part_names(self) -> None:
        self.assertIn("unknown --part value", self.source)
        self.assertIn("part_index(part)", self.source)
        # part_index() returns -1 for anything not in {query,key,value}, and
        # parse_options rejects that.
        self.assertIn("if (index < 0)", self.source)

    def test_requires_at_least_one_part(self) -> None:
        self.assertIn("raw_parts.empty()", self.source)
        self.assertIn("at least one --part is required", self.source)

    def test_part_names_are_exactly_query_key_value(self) -> None:
        self.assertIn(
            'const std::array<std::string, 3> PART_NAMES = {"query", "key", "value"};',
            self.source,
        )

    def test_evaluate_only_populates_requested_parts(self) -> None:
        evaluate = self.source.split("EvaluationResult evaluate(", 1)[1].split(
            "\n  private:", 1
        )[0]
        # The only loop that assigns into result.packed[...] must be gated
        # by iterating over `requested`, not over all of PART_NAMES/the
        # fixed {0,1,2} range.
        self.assertIn("for (const int part : requested)", evaluate)
        # matmul() must only ever be invoked with a part drawn from
        # `requested`, never unconditionally for all three parts (that would
        # be the frozen full-block source's unconditional Q+K+V behavior,
        # which this fork must NOT reproduce for a partial --part request).
        self.assertNotIn("matmul(baby, fixture_.qkv[0])", evaluate)
        self.assertNotIn("matmul(baby, fixture_.qkv[1])", evaluate)
        self.assertNotIn("matmul(baby, fixture_.qkv[2])", evaluate)

    def test_baby_rotations_computed_once_per_group_shared_across_parts(self) -> None:
        evaluate = self.source.split("EvaluationResult evaluate(", 1)[1].split(
            "\n  private:", 1
        )[0]
        # Exactly one baby_rotations() call per outer token-group loop
        # iteration (shared across every requested part's matmul in the
        # inner `for (const int part : requested)` loop) -- not once per
        # part.
        self.assertEqual(evaluate.count("baby_rotations(normalized)"), 1)

    def test_final_count_guard_rejects_computing_unrequested_parts(self) -> None:
        # main() throws if matrix_products != TOKEN_GROUPS * requested.size(),
        # which would fail if the reader computed more (or fewer) parts than
        # were actually requested.
        self.assertIn(
            "evaluation.matrix_products != TOKEN_GROUPS * requested.size()",
            self.source,
        )

    def test_metrics_reported_only_for_requested_parts(self) -> None:
        make_json = self.source.split("std::string make_json(", 1)[1].split(
            "return out.str();", 1
        )[0]
        self.assertIn(
            "for (std::size_t i = 0; i < options.parts.size(); ++i)", make_json
        )

    def test_part_flag_accepts_comma_separated_and_repeated_forms(self) -> None:
        self.assertIn("split_comma(next())", self.source)
        # --part is parsed inside the main arg loop, not restricted to a
        # single occurrence, so repeated "--part x --part y" also works.
        self.assertNotIn("only one --part", self.source)


class ShardBuildAndScriptContractTests(unittest.TestCase):
    def test_cmake_references_both_new_binaries(self) -> None:
        cmake = Path(__file__).with_name("CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn(
            "real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13", cmake
        )
        self.assertIn(
            "real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13", cmake
        )

    def test_build_script_builds_both_new_targets(self) -> None:
        build_script = (
            Path(__file__).with_name("build_in_fideslib.sh").read_text(encoding="utf-8")
        )
        self.assertIn(
            "real_dnagpt_fides_scheme_b_simd_shard_writer_t103_depth13", build_script
        )
        self.assertIn(
            "real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13", build_script
        )

    def test_run_scripts_require_scheme_b_and_qkvshard_substrings(self) -> None:
        for name in (
            "run_scheme_b_simd_shard_writer_t103_depth13.sh",
            "run_scheme_b_simd_shard_reader_t103_depth13.sh",
        ):
            script = Path(__file__).with_name(name).read_text(encoding="utf-8")
            self.assertIn("_scheme_b_", script)
            self.assertIn("_qkvshard_", script)
            # Evidence-tag marker must be distinct from both existing
            # prototypes' markers so evidence can never collide.
            self.assertNotIn('"_cached_"', script)
            self.assertNotIn('"_serialized_"', script)

    def test_launch_scripts_require_scheme_b_and_qkvshard_substrings(self) -> None:
        for name in (
            "launch_brev_scheme_b_simd_shard_writer_t103_depth13.sh",
            "launch_brev_scheme_b_simd_shard_reader_t103_depth13.sh",
        ):
            script = Path(__file__).with_name(name).read_text(encoding="utf-8")
            self.assertIn("_scheme_b_", script)
            self.assertIn("_qkvshard_", script)

    def test_qkvshard_tag_is_distinct_from_existing_markers(self) -> None:
        self.assertNotEqual("_qkvshard_", "_cached_")
        self.assertNotEqual("_qkvshard_", "_serialized_")
        self.assertNotIn("_cached_", "_qkvshard_")
        self.assertNotIn("_serialized_", "_qkvshard_")
        self.assertNotIn("_qkvshard_", "_cached_")
        self.assertNotIn("_qkvshard_", "_serialized_")


if __name__ == "__main__":
    unittest.main()
