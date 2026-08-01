"""Local contract for Stage-2 (MLP down-projection chunk split) 2-GPU
process-per-GPU sharding: the sharded MLP reader + the exact EvalAdd merge.

Per docs/hybrid/tasks.md's "2-GPU process-per-GPU sharding" entry, Stage 1
(Q/K/V split) deliberately deferred the *merge-correctness* question because
Q/K/V are three independent output ciphertexts never combined by any encrypted
op. Stage 2 is that deferred, harder case: the MLP down-projection is a genuine
partial-SUM accumulation --

    Ct mlp;                                    # frozen full-block source
    for (chunk = 0..COPIES-1)
        mlp = chunk==0 ? matmul(baby(hidden[chunk]), mlp_projection[chunk])
                       : EvalAdd(mlp, matmul(...));
    block_output = EvalAdd(residual1, mlp);

-- four D x D matrix products SUMMED (EvalAdd) into the same output ciphertext.
This contract proves, in float64 NumPy (no CUDA/FIDESlib built or launched),
that partitioning those four chunks 2-and-2 across two workers and re-summing
their partial products with EvalAdd reproduces the full un-sharded
down-projection output to ~1e-9 -- i.e. the Stage-2 merge is an exact math
no-op -- and that the two new C++ sources implement exactly that split with the
required fork/secret-key/serialization discipline.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np

from shard_layout import (
    MLP_CHUNKS,
    mlp_chunk_shard_assignment,
    mlp_chunk_tag,
    mlp_shard_of,
)
from simd_layout import bsgs_matmul_tokens, unpack_tokens

D = 768
B = 8
COPIES = 4

SRC = Path(__file__).with_name("src")
READER_SOURCE = (
    SRC / "real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13.cpp"
)
MERGE_SOURCE = SRC / "real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13.cpp"

# Parents, computed fresh below and pinned in the new sources.
FULLBLOCK_PARENT = (
    SRC
    / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp"
)
STAGE1_READER = SRC / "real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp"
DEPTH13_SCHEDULE = (
    SRC / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"
)

FULLBLOCK_PARENT_SHA256 = (
    "686c8b766436fde0fd4422f6a29f80361f8c8a2565768577857e9264b2976fd5"
)
STAGE1_READER_SHA256 = (
    "3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89"
)
DEPTH13_SCHEDULE_SHA256 = (
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e"
)


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


def _chunk_contributions(
    hidden_by_chunk: list[np.ndarray], weight_by_chunk: list[np.ndarray]
) -> list[np.ndarray]:
    """Model each MLP down-projection chunk's packed BSGS matmul, exactly as
    the C++ down-projection does (matmul(baby(hidden[chunk]),
    mlp_projection[chunk])); returns one packed slot-vector per chunk."""

    return [
        bsgs_matmul_tokens(hidden_by_chunk[chunk], weight_by_chunk[chunk], B)
        for chunk in range(COPIES)
    ]


class MlpChunkAssignmentTests(unittest.TestCase):
    def test_one_gpu_assignment_is_the_full_chunk_set(self) -> None:
        self.assertEqual(mlp_chunk_shard_assignment(1), {0: MLP_CHUNKS})

    def test_two_gpu_assignment_is_an_exact_2_and_2_partition(self) -> None:
        assignment = mlp_chunk_shard_assignment(2)
        self.assertEqual(assignment, {0: (0, 1), 1: (2, 3)})
        covered: list[int] = []
        for chunks in assignment.values():
            self.assertEqual(len(chunks), 2)
            covered.extend(chunks)
        self.assertEqual(sorted(covered), sorted(MLP_CHUNKS))
        self.assertEqual(len(covered), len(set(covered)))  # disjoint

    def test_shard_of_agrees_with_assignment(self) -> None:
        for chunk in MLP_CHUNKS:
            worker = mlp_shard_of(chunk, 2)
            self.assertIn(chunk, mlp_chunk_shard_assignment(2)[worker])

    def test_primary_owns_chunk_zero(self) -> None:
        self.assertEqual(mlp_shard_of(0, 2), 0)

    def test_undefined_gpu_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mlp_chunk_shard_assignment(3)
        with self.assertRaises(ValueError):
            mlp_shard_of(9, 2)

    def test_chunk_tags_match_c_plus_plus_convention(self) -> None:
        self.assertEqual(mlp_chunk_tag((0, 1)), "01")
        self.assertEqual(mlp_chunk_tag((2, 3)), "23")


class ShardedMlpMergeIsExactTests(unittest.TestCase):
    """The heart of the contract: 2-and-2 split + EvalAdd merge == the full
    un-sharded down-projection, to ~1e-9, on the real packed BSGS transform."""

    def setUp(self) -> None:
        rng = np.random.default_rng(20260801)
        active = B  # a full B=8 token group; the identity is per-group
        # Four post-GELU replicated hidden D-blocks (the C++ hidden[0..3]).
        self.hidden = [rng.standard_normal((active, D)) for _ in range(COPIES)]
        # Four D x D down-projection weight slices (mlp_projection[0..3]).
        self.weight = [rng.standard_normal((D, D)) for _ in range(COPIES)]
        self.active = active

    def test_two_and_two_split_reproduces_unsharded_down_projection(self) -> None:
        contributions = _chunk_contributions(self.hidden, self.weight)

        # Un-sharded accumulation, exactly as the frozen source does it:
        # mlp = ((c0 + c1) + c2) + c3, all in the packed (ciphertext) domain.
        unsharded = contributions[0].copy()
        for chunk in range(1, COPIES):
            unsharded = unsharded + contributions[chunk]

        # Sharded: worker 0 owns {0,1}, worker 1 owns {2,3}; each EvalAdds its
        # own two chunks into a PARTIAL sum, then the merge EvalAdds the two
        # partials. This is the exact operator graph of the C++ reader+merge.
        assignment = mlp_chunk_shard_assignment(2)
        partial_lo = contributions[assignment[0][0]] + contributions[assignment[0][1]]
        partial_hi = contributions[assignment[1][0]] + contributions[assignment[1][1]]
        merged = partial_lo + partial_hi  # <-- the Stage-2 EvalAdd merge

        max_abs = float(np.max(np.abs(merged - unsharded)))
        # Only floating-point reassociation ((a+b)+(c+d) vs ((a+b)+c)+d)
        # separates the two -- far below 1e-9.
        self.assertLess(max_abs, 1e-9, f"merge drift {max_abs:g}")

        # And after unpacking to token space, both equal the monolithic MLP
        # down-projection output (sum over all four chunks), proving the split
        # reproduces the *full* un-sharded output, not merely each other.
        got = unpack_tokens(merged, B, self.active)
        reference = unpack_tokens(unsharded, B, self.active)
        np.testing.assert_allclose(got, reference, rtol=0.0, atol=1e-9)

    def test_disjoint_halves_cover_every_chunk_exactly_once(self) -> None:
        # A wrong split (overlap or a dropped chunk) must NOT reconstruct the
        # full sum -- guards the assignment against silent miscoverage.
        contributions = _chunk_contributions(self.hidden, self.weight)
        full = sum(contributions[1:], contributions[0].copy())
        overlap_bad = (contributions[0] + contributions[1]) + (
            contributions[1] + contributions[2]
        )  # chunk 1 twice, 3 missing
        self.assertGreater(float(np.max(np.abs(overlap_bad - full))), 1e-6)

    def test_merge_is_associative_across_any_two_way_grouping(self) -> None:
        contributions = _chunk_contributions(self.hidden, self.weight)
        full = sum(contributions[1:], contributions[0].copy())
        for split in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            lo = contributions[split[0][0]] + contributions[split[0][1]]
            hi = contributions[split[1][0]] + contributions[split[1][1]]
            self.assertLess(float(np.max(np.abs((lo + hi) - full))), 1e-9)


class UpstreamParentsUntouchedTests(unittest.TestCase):
    def test_fullblock_parent_untouched(self) -> None:
        self.assertEqual(_sha256(FULLBLOCK_PARENT), FULLBLOCK_PARENT_SHA256)

    def test_stage1_reader_parent_untouched(self) -> None:
        self.assertEqual(_sha256(STAGE1_READER), STAGE1_READER_SHA256)

    def test_depth13_schedule_untouched(self) -> None:
        self.assertEqual(_sha256(DEPTH13_SCHEDULE), DEPTH13_SCHEDULE_SHA256)

    def test_new_sources_are_genuinely_separate_files(self) -> None:
        for new in (READER_SOURCE, MERGE_SOURCE):
            self.assertTrue(new.exists())
            self.assertNotEqual(_sha256(new), FULLBLOCK_PARENT_SHA256)
            self.assertNotEqual(_sha256(new), STAGE1_READER_SHA256)


class ShardMlpReaderContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = READER_SOURCE.read_text(encoding="utf-8")

    def test_pins_fullblock_and_stage1_reader_parent_hashes(self) -> None:
        self.assertIn(FULLBLOCK_PARENT_SHA256, self.source)
        self.assertIn(STAGE1_READER_SHA256, self.source)

    def test_never_regenerates_keys_or_builds_fresh_context(self) -> None:
        for needle in (
            "GenCryptoContext(",
            "->KeyGen()",
            "->EvalMultKeyGen(",
            "->EvalRotateKeyGen(",
        ):
            self.assertEqual(
                self.source.count(needle),
                0,
                f"reader must NEVER regenerate keys/context: {needle!r}",
            )

    def test_deserializes_context_public_and_secret_key(self) -> None:
        self.assertEqual(self.source.count("Serial::DeserializeFromFile("), 3)
        for needle in ('"crypto-context.txt"', '"public-key.txt"', '"secret-key.txt"'):
            self.assertIn(needle, self.source)

    def test_deserializes_mult_and_automorphism_keys_once(self) -> None:
        self.assertEqual(self.source.count("DeserializeEvalMultKey("), 1)
        self.assertEqual(self.source.count("DeserializeEvalAutomorphismKey("), 1)

    def test_deserialize_before_load_context(self) -> None:
        self.assertEqual(self.source.count("->LoadContext("), 1)
        last_deserialize = self.source.rindex("Serial::DeserializeFromFile(")
        self.assertLess(last_deserialize, self.source.index("->LoadContext("))

    def test_down_projection_only_accumulates_assigned_chunks(self) -> None:
        evaluate = self.source.split("EvaluationResult evaluate(", 1)[1].split(
            "\n  private:", 1
        )[0]
        # The partial down-projection loop iterates the assigned `chunks`, NOT
        # all COPIES chunks (that would be the frozen full-sum behavior).
        self.assertIn("for (const std::size_t chunk : chunks)", evaluate)
        self.assertIn("partial_mlp[group] = partial;", evaluate)
        # The frozen source's unconditional full-sum accumulation into `mlp`
        # over all four chunks must be gone.
        self.assertNotIn("mlp = chunk == 0", evaluate)

    def test_serializes_partials_and_primary_serializes_residual1(self) -> None:
        self.assertIn("Serial::SerializeToFile(path.string(), ct", self.source)
        self.assertIn("partial-mlp-chunks", self.source)
        self.assertIn("residual1-group", self.source)
        self.assertIn("if (options.primary())", self.source)

    def test_chunks_flag_requires_a_disjoint_contiguous_pair(self) -> None:
        self.assertIn("--chunks must be exactly the disjoint pair", self.source)
        self.assertIn("options.chunks[0] == 0 && options.chunks[1] == 1", self.source)
        self.assertIn("options.chunks[0] == 2 && options.chunks[1] == 3", self.source)

    def test_evaluator_has_no_private_key_access(self) -> None:
        evaluator = self.source.split("class EncryptedEvaluator", 1)[1].split(
            "std::string make_json(", 1
        )[0]
        self.assertNotIn("Decrypt(", evaluator)
        self.assertNotIn("secretKey", evaluator)

    def test_no_block_output_metric_or_final_gate_in_reader(self) -> None:
        # The reader must NOT measure a block output (the merge does).
        self.assertNotIn("measure_output(", self.source)
        self.assertIn('produces_full_block_output\\": false', self.source)

    def test_secret_key_never_logged(self) -> None:
        for line in _log_lines(self.source):
            self.assertNotIn("secret-key.txt", line)

    def test_output_immutability_and_pin_guards_present(self) -> None:
        self.assertIn("O_WRONLY | O_CREAT | O_EXCL", self.source)
        self.assertIn('"786c7600fb2f16b724e0acf73df367b27b8afed6"', self.source)

    def test_uses_token_simd_batch_slots_and_depth(self) -> None:
        self.assertIn("constexpr std::size_t TOKEN_BATCH = 8;", self.source)
        self.assertIn("static_assert(SLOTS == 32768)", self.source)
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 13;", self.source)


class ShardMlpMergeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = MERGE_SOURCE.read_text(encoding="utf-8")

    def test_pins_stage1_reader_and_schedule_parent_hashes(self) -> None:
        self.assertIn(STAGE1_READER_SHA256, self.source)
        self.assertIn(DEPTH13_SCHEDULE_SHA256, self.source)

    def test_never_regenerates_keys_or_builds_fresh_context(self) -> None:
        for needle in (
            "GenCryptoContext(",
            "->KeyGen()",
            "->EvalMultKeyGen(",
            "->EvalRotateKeyGen(",
        ):
            self.assertEqual(self.source.count(needle), 0, needle)

    def test_the_exact_evaladd_merge_line_is_present(self) -> None:
        # The one line that IS Stage 2: EvalAdd the two workers' partials.
        self.assertIn(
            "cc->EvalAdd(partials[0][group], partials[1][group])", self.source
        )
        # Then add residual1 to form the block output.
        self.assertIn("cc->EvalAdd(residual1[group], mlp)", self.source)

    def test_merger_does_no_dense_transform_or_boundary(self) -> None:
        # The merge must not run any matmul/attention/LayerNorm/GELU or client
        # boundary -- its only encrypted ops are the EvalAdds.
        for forbidden in (
            "matmul(",
            "baby_rotations(",
            "AccumulateSum(",
            "invsqrt_boundary",
            "gelu_boundary",
            "EvalMult(",
            "EvalRotate(",
        ):
            self.assertNotIn(forbidden, self.source, forbidden)

    def test_deserializes_partials_and_residual_ciphertexts(self) -> None:
        self.assertIn("deserialize_ciphertext(", self.source)
        self.assertIn("partial-mlp-chunks", self.source)
        self.assertIn("residual1-group", self.source)
        self.assertIn('PARTIAL_TAGS = {"01", "23"}', self.source)

    def test_measures_reconstructed_block_output_against_oracle(self) -> None:
        self.assertIn("measure_output(output, oracle_block_output)", self.source)
        self.assertIn('"oracle__block_output.bin"', self.source)

    def test_expects_exactly_token_group_merges(self) -> None:
        self.assertIn("merge_evaladds != TOKEN_GROUPS", self.source)
        self.assertIn("residual_evaladds != TOKEN_GROUPS", self.source)

    def test_secret_key_never_logged(self) -> None:
        for line in _log_lines(self.source):
            self.assertNotIn("secret-key.txt", line)

    def test_output_immutability_and_pin_guards_present(self) -> None:
        self.assertIn("O_WRONLY | O_CREAT | O_EXCL", self.source)
        self.assertIn('"786c7600fb2f16b724e0acf73df367b27b8afed6"', self.source)


class ShardMlpBuildWiringTests(unittest.TestCase):
    def test_cmake_references_both_new_targets(self) -> None:
        cmake = Path(__file__).with_name("CMakeLists.txt").read_text(encoding="utf-8")
        for target in (
            "real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13",
            "real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13",
        ):
            self.assertIn(f"add_executable(\n    {target}", cmake)
            self.assertIn(f"src/{target}.cpp", cmake)

    def test_build_script_builds_both_new_targets(self) -> None:
        build = (
            Path(__file__).with_name("build_in_fideslib.sh").read_text(encoding="utf-8")
        )
        for target in (
            "real_dnagpt_fides_scheme_b_simd_shard_mlp_reader_t103_depth13",
            "real_dnagpt_fides_scheme_b_simd_shard_mlp_merge_t103_depth13",
        ):
            self.assertIn(f"--target {target} ", build)

    def test_mlp_shard_tag_is_distinct_from_stage1_qkv_marker(self) -> None:
        # Stage-1's evidence marker is "_qkvshard_"; Stage 2 must not collide.
        self.assertNotIn("_qkvshard_", READER_SOURCE.read_text(encoding="utf-8"))
        self.assertNotIn("_qkvshard_", MERGE_SOURCE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
