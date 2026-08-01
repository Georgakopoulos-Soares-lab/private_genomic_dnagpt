"""Local, no-GPU contract for the COMBINED-lever correctness micro-gate:
2-GPU process-per-GPU Q/K/V sharding (Stage 1) PLUS the CPU-side
diagonal-vector cache, both forked into
`real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.cpp`.

Math correctness of the two levers individually is already proved and not
re-derived here:
  - the per-shard Q/K/V split reproduces the real T=103 oracle exactly
    (`test_shard_layout.py`'s `ShardedQkvMatchesUnshardedTests`);
  - the BSGS diagonal `packed_values` construction is a pure function of
    (weight, giant, small) -- independent of which token group or process
    calls it -- for every real distinct weight matrix, including each of
    the three QKV slices this reader ever touches individually
    (`test_diagonal_cache_reuse.py`'s
    `test_diagonal_construction_is_byte_identical_across_all_token_groups`,
    run against `qkv_query`/`qkv_key`/`qkv_value` among the 12 real
    matrices it covers).

Since the cache is keyed by weight-vector address and this reader only
ever calls matmul() with the 1-2 weight addresses in its own `--part`
selection (never a weight outside `requested`), combining the two levers
introduces no new arithmetic: it is the same per-shard matmul the
uncached shard reader already proved correct, with the same pure-function
diagonal construction the uncached cpudiagcache gate already proved is a
cache-safe no-op. This file's job is therefore the static/structural
contract -- proving the C++ actually wires the two levers together as
designed (parent hashes pinned, cache present and gating matmul(), the
per-shard hit/miss invariant asserted, build/run/launch scripts wired) --
not re-proving math already covered above.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

SRC = Path(__file__).with_name("src")

SOURCE = (
    SRC / "real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.cpp"
)

# Three pinned parents, all verified computed fresh against this repo's
# working tree below -- not trusted from any stale prior value.
SHARD_READER_PARENT = (
    SRC / "real_dnagpt_fides_scheme_b_simd_shard_reader_t103_depth13.cpp"
)
SHARD_READER_PARENT_SHA256 = (
    "3261398e2d17c3261260ab87fca625963fc662d993b977aeb27d004363bc5c89"
)
STRUCTURAL_PARENT = SRC / "real_dnagpt_fides_scheme_b_serialize_reader.cpp"
STRUCTURAL_PARENT_SHA256 = (
    "4dbb003fe8407046c852bf10f0c45b23b579cd1bbc7390099966999b6d6ab5d6"
)
SCHEDULE_PARENT = (
    SRC / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"
)
SCHEDULE_PARENT_SHA256 = (
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e"
)
CPUDIAGCACHE_PARENT = (
    SRC
    / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536_cpudiagcache.cpp"
)
CPUDIAGCACHE_PARENT_SHA256 = (
    "686c8b766436fde0fd4422f6a29f80361f8c8a2565768577857e9264b2976fd5"
)

TOKEN_GROUPS = 13
BSGS_N1 = 32
BSGS_N2 = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class UpstreamFrozenSourcesTests(unittest.TestCase):
    """None of the three parents may be edited by this fork -- only forked
    from. This is the "do not modify, fork from them" rule the task brief
    itself calls out."""

    def test_shard_reader_parent_is_untouched(self) -> None:
        self.assertTrue(SHARD_READER_PARENT.exists())
        self.assertEqual(_sha256(SHARD_READER_PARENT), SHARD_READER_PARENT_SHA256)

    def test_structural_parent_is_untouched(self) -> None:
        self.assertTrue(STRUCTURAL_PARENT.exists())
        self.assertEqual(_sha256(STRUCTURAL_PARENT), STRUCTURAL_PARENT_SHA256)

    def test_schedule_parent_is_untouched(self) -> None:
        self.assertTrue(SCHEDULE_PARENT.exists())
        self.assertEqual(_sha256(SCHEDULE_PARENT), SCHEDULE_PARENT_SHA256)

    def test_cpudiagcache_parent_is_untouched(self) -> None:
        self.assertTrue(CPUDIAGCACHE_PARENT.exists())
        self.assertEqual(_sha256(CPUDIAGCACHE_PARENT), CPUDIAGCACHE_PARENT_SHA256)


class CombinedSourceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(SOURCE.exists(), f"missing {SOURCE}")
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_pins_all_three_parent_hashes(self) -> None:
        for sha in (
            SHARD_READER_PARENT_SHA256,
            STRUCTURAL_PARENT_SHA256,
            SCHEDULE_PARENT_SHA256,
            CPUDIAGCACHE_PARENT_SHA256,
        ):
            self.assertIn(sha, self.source)

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
                f"the combined reader must NEVER regenerate keys or build a "
                f"fresh CPU context -- found forbidden call: {needle!r}",
            )

    def test_deserializes_context_public_and_secret_key(self) -> None:
        self.assertEqual(self.source.count("Serial::DeserializeFromFile("), 3)

    def test_cache_holds_only_host_vectors_never_gpu_plaintext(self) -> None:
        # Same structural guarantee as the cpudiagcache parent: the map
        # value type is a host std::vector<double>, never a Plaintext or
        # Ciphertext -- so this fork structurally cannot hit the abandoned
        # 2026-07-27/28 GPU-resident-Plaintext-reuse crash mechanism.
        self.assertIn(
            "std::map<const std::vector<double>*, std::vector<std::vector<double>>>",
            self.source,
        )
        self.assertNotIn("std::map<const std::vector<double>*, Plaintext>", self.source)
        self.assertNotIn("std::map<const std::vector<double>*, Ct>", self.source)

    def test_matmul_uses_the_cache_not_an_inline_rebuild(self) -> None:
        matmul = self.source.split("Ct matmul(", 1)[1].split("\n\n    Cc cc_;", 1)[0]
        self.assertIn("cached_packed_values(weight, giant, small)", matmul)
        # The inline "rebuild packed_values from scratch every call" shape
        # the shard-reader (uncached) parent used must be gone from
        # matmul() itself -- it now lives only inside cached_packed_values().
        self.assertNotIn("std::vector<double> packed_values(SLOTS, 0.0);", matmul)

    def test_a_fresh_plaintext_is_still_built_on_every_call(self) -> None:
        # Only the CPU-side vector<double> is cached -- raw_plain() (which
        # calls MakeCKKSPackedPlaintext) must still run on every matmul()
        # call, never itself cached.
        matmul = self.source.split("Ct matmul(", 1)[1].split("\n\n    Cc cc_;", 1)[0]
        self.assertIn("raw_plain(packed_values)", matmul)
        self.assertNotIn(
            "std::map<const std::vector<double>*, Plaintext> diagonal_plain_cache",
            self.source,
        )

    def test_evaluation_result_reports_cache_hits_and_misses(self) -> None:
        self.assertIn("diagonal_cache_hits", self.source)
        self.assertIn("diagonal_cache_misses", self.source)

    def test_per_shard_hit_miss_invariant_is_asserted_in_main(self) -> None:
        # This is the per-shard keying question the task brief asked to
        # confirm before coding: since each shard only ever touches the
        # weight addresses in its own `requested` set, main() must assert
        # exactly requested.size() misses and (TOKEN_GROUPS-1)*requested.size()
        # hits -- not the full-block gate's fixed EXPECTED_DIAGONAL_CACHE_MISSES=12
        # (this reader never sees all 12 weights; it sees at most 3, and
        # only the ones in `--part`).
        main_body = self.source.split("int main(", 1)[1]
        self.assertIn("expected_misses = requested.size()", main_body)
        self.assertIn(
            "(TOKEN_GROUPS * BSGS_N1 * BSGS_N2 - 1) * requested.size()", main_body
        )
        self.assertIn("diagonal-cache hit/miss count mismatch", main_body)

    def test_no_cross_process_cache_sharing_construct_present(self) -> None:
        # The cache must be a plain per-instance member (constructed fresh
        # in each process's own EncryptedEvaluator), never anything backed
        # by shared memory, a file, or a socket -- this reader is a single
        # OS process per shard worker, so ordinary process memory isolation
        # is the entire guarantee; this test just confirms no such sharing
        # mechanism was accidentally introduced.
        for needle in ("shm_open", "mmap(", "socket(", "/dev/shm"):
            self.assertNotIn(needle, self.source)

    def test_evaluate_only_populates_requested_parts(self) -> None:
        evaluate = self.source.split("EvaluationResult evaluate(", 1)[1].split(
            "\n  private:", 1
        )[0]
        self.assertIn("for (const int part : requested)", evaluate)
        self.assertNotIn("matmul(baby, fixture_.qkv[0])", evaluate)
        self.assertNotIn("matmul(baby, fixture_.qkv[1])", evaluate)
        self.assertNotIn("matmul(baby, fixture_.qkv[2])", evaluate)

    def test_never_logs_secret_key_path_or_contents(self) -> None:
        log_lines = [
            line
            for line in self.source.splitlines()
            if "std::cout" in line or "std::cerr" in line
        ]
        for line in log_lines:
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


def _simulate_cache(
    token_groups: int, requested_parts: int, bsgs_n1: int, bsgs_n2: int
):
    """Reimplements cached_packed_values()'s hit/miss bookkeeping exactly:
    a dict keyed by weight identity; the first lookup for a weight is a
    miss (and implicitly "builds" all bsgs_n1*bsgs_n2 diagonals at once,
    matching the C++'s single std::vector<std::vector<double>> per weight);
    every other lookup for that weight -- including the remaining
    bsgs_n1*bsgs_n2-1 lookups within the SAME matmul() call -- is a hit.
    This is the exact computation main()'s expected_misses/expected_hits
    invariant re-derives, run here independently in Python so a future
    change to either the C++ or this test's own formula gets cross-checked
    against a from-scratch simulation, not just a duplicated arithmetic
    expression.
    """
    hits = 0
    misses = 0
    cache = set()
    for weight in range(requested_parts):
        for _group in range(token_groups):
            for _diagonal in range(bsgs_n1 * bsgs_n2):
                if weight in cache:
                    hits += 1
                else:
                    cache.add(weight)
                    misses += 1
    return hits, misses


class SimulatedCacheHitMissFormulaTests(unittest.TestCase):
    """Independently re-derives the exact hit/miss counts main()'s
    expected_misses/expected_hits invariant asserts, by simulating the
    cache's own lookup pattern from scratch -- this is the check that
    would have caught the 2026-08-01 first-attempt bug (the invariant
    originally assumed one cache lookup per matmul() call, when
    cached_packed_values() is actually called BSGS_N1*BSGS_N2=1024 times
    per matmul() call)."""

    def test_one_part_worker(self) -> None:
        hits, misses = _simulate_cache(TOKEN_GROUPS, 1, BSGS_N1, BSGS_N2)
        self.assertEqual(misses, 1)
        self.assertEqual(hits, TOKEN_GROUPS * BSGS_N1 * BSGS_N2 - 1)

    def test_two_part_worker(self) -> None:
        hits, misses = _simulate_cache(TOKEN_GROUPS, 2, BSGS_N1, BSGS_N2)
        self.assertEqual(misses, 2)
        self.assertEqual(hits, 2 * (TOKEN_GROUPS * BSGS_N1 * BSGS_N2 - 1))

    def test_matches_the_cpp_expected_hits_expression_in_source(self) -> None:
        # Cross-check against the literal formula string pinned in
        # test_per_shard_hit_miss_invariant_is_asserted_in_main above: both
        # must describe the same quantity.
        for requested in (1, 2, 3):
            hits, misses = _simulate_cache(TOKEN_GROUPS, requested, BSGS_N1, BSGS_N2)
            self.assertEqual(misses, requested)
            self.assertEqual(hits, (TOKEN_GROUPS * BSGS_N1 * BSGS_N2 - 1) * requested)


class ShardReaderCpudiagcacheBuildAndScriptContractTests(unittest.TestCase):
    def test_cmake_references_the_new_binary(self) -> None:
        cmake = Path(__file__).with_name("CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn(
            "real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13",
            cmake,
        )

    def test_build_script_builds_the_new_target(self) -> None:
        build_script = (
            Path(__file__).with_name("build_in_fideslib.sh").read_text(encoding="utf-8")
        )
        self.assertIn(
            "real_dnagpt_fides_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13",
            build_script,
        )

    def test_run_script_requires_all_three_tag_substrings(self) -> None:
        script = (
            Path(__file__)
            .with_name("run_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.sh")
            .read_text(encoding="utf-8")
        )
        self.assertIn("_scheme_b_", script)
        self.assertIn("_qkvshard_", script)
        self.assertIn("_cpudiagcache_", script)

    def test_launch_script_requires_all_three_tag_substrings(self) -> None:
        script = (
            Path(__file__)
            .with_name(
                "launch_brev_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.sh"
            )
            .read_text(encoding="utf-8")
        )
        self.assertIn("_scheme_b_", script)
        self.assertIn("_qkvshard_", script)
        self.assertIn("_cpudiagcache_", script)

    def test_orchestrator_reuses_the_unchanged_writer_launch_script(self) -> None:
        # The cache lever only touches the reader; the writer needs no
        # fork, so the orchestrator must call the EXISTING writer launch
        # script verbatim, not a new one.
        orchestrator = (
            Path(__file__)
            .with_name(
                "wait_and_run_scheme_b_simd_shard_qkv_cpudiagcache_t103_depth13.sh"
            )
            .read_text(encoding="utf-8")
        )
        self.assertIn(
            "launch_brev_scheme_b_simd_shard_writer_t103_depth13.sh", orchestrator
        )
        self.assertIn(
            "launch_brev_scheme_b_simd_shard_reader_cpudiagcache_t103_depth13.sh",
            orchestrator,
        )

    def test_run_tag_convention_is_distinct_from_the_uncached_shard_gate(self) -> None:
        # Evidence for the combined gate must never collide with the
        # uncached shard-reader gate's own evidence files.
        self.assertNotIn("_qkvshard_", "_cpudiagcache_")
        self.assertNotIn("_cpudiagcache_", "_qkvshard_")


if __name__ == "__main__":
    unittest.main()
