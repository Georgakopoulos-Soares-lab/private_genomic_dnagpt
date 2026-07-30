"""Static and float64 contracts for the T=103 chunked/streaming-softmax
checkpoint (real_dnagpt_fides_scheme_b_general_attention_t103.cpp).

T=103 is the FULL task-representative GSR prompt length (docs/roadmap.md),
not a truncated prefix -- confirmed by tokenizing the exact fixture prompt
and observing exactly 103 tokens before writing this file. It exceeds the
T=32 file's spread-across-copies packing ceiling (T<=85,
HEADS_PER_COPY*SCORE_SLOT_STRIDE<=PACK_WIDTH-D), which is why this file
exists: it implements the two-phase chunked softmax scoped and numpy-proven
in docs/hybrid/tasks.md's 2026-07-29 "chunked/streaming causal-score
packing" entry, replacing SCORE_SLOT_STRIDE=T (T-dependent, hence bounded)
with a fixed CHUNK_WIDTH constant independent of T.

This test file proves, before any GPU work, that:

  - the frozen T=2, T=3, T=8, and T=32 sources are all untouched;
  - CHUNK_WIDTH is a fixed constant (not derived from T) and the packing
    static_assert uses HEADS_PER_COPY*CHUNK_WIDTH, not HEADS_PER_COPY*T;
  - chunk_row() partitions any row length into <=CHUNK_WIDTH chunks,
    bijectively covering every column, for row lengths up to and beyond
    T=103's own maximum (row_length=103) and well past the old T<=85
    ceiling;
  - Phase A (cross_boundary_reduce_chunk) is an IDENTITY transform (caches
    values, does not mutate them) reusing the unchanged cross_boundary_impl
    mechanics;
  - Phase B (cross_boundary_emit_weight) contains no Decrypt call --
    encrypt-only, computed purely from the client's own cache;
  - the two-phase design reproduces a plain reference softmax exactly
    against the real T=103 oracle's attention scores, for every non-trivial
    causal row;
  - the honest round-trip growth formula matches this file's own design
    exactly;
  - output-packing (carried forward from T=8/T=32) still works generally at
    T=103 (OUTPUT_GROUPS=ceil(103/4)=26);
  - CMake/build/run/launch/orchestrator scripts reference the new binary
    and require the correct tag markers.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).with_name("src")
SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention_t103.cpp"
T32_SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention_t32.cpp"
T8_SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention_t8.cpp"
T3_SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention.cpp"
ORIGINAL_SOURCE = HERE / "real_dnagpt_fides_scheme_b.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
T3_SHA256 = "c57e81d441b27400453c29e941d4f5e810df850a90cfc2d88ce68178ee4cd0f8"
T8_SHA256 = "9d4dde16a0803d29c7e97d517fac77a275057ad3258d99c3e260f65feca85545"
T32_SHA256 = "607e9c429bdea107149733e10f19765578175a5301b350556bd8604a1d384252"

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = SCRIPT_DIR / "run_scheme_b_general_attention_t103.sh"
LAUNCH_SCRIPT = SCRIPT_DIR / "launch_brev_scheme_b_general_attention_t103.sh"
WAIT_SCRIPT = SCRIPT_DIR / "wait_and_run_scheme_b_general_attention_t103.sh"
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"
FIXTURE_SHA_FILE = SCRIPT_DIR / "fixture_t103.sha256"

D = 768
T = 103
HEADS = 12
PACK_WIDTH = 1024
COPIES = 4
HEADS_PER_COPY = HEADS // COPIES
CHUNK_WIDTH = (PACK_WIDTH - D) // HEADS_PER_COPY  # 85, fixed regardless of T

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
)
FIXTURE_MANIFEST_SHA256 = (
    "d2c90ba15c62c648495b51070eee585a3570dc174eb31e14782aaf016b31f8f6"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_array(name: str) -> np.ndarray:
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    metadata = manifest["arrays"][name]
    path = FIXTURE / metadata["file"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError(f"fixture hash mismatch for {name}")
    return np.fromfile(path, dtype="<f8").reshape(metadata["shape"])


def chunk_row(row_length: int, chunk_width: int) -> list[tuple[int, int]]:
    """Plain-Python model of the C++ chunk_row(): partitions
    0..row_length-1 into (start, count) chunks of at most chunk_width."""
    if chunk_width <= 0:
        raise ValueError("chunk width must be positive")
    chunks = []
    start = 0
    while start < row_length:
        count = min(chunk_width, row_length - start)
        chunks.append((start, count))
        start += count
    return chunks


def _exact_softmax_select(raw_row: np.ndarray, select_index: int) -> float:
    shifted = raw_row - np.max(raw_row)
    numerator = np.exp(shifted[select_index])
    denominator = np.sum(np.exp(shifted))
    return float(numerator / denominator)


def _two_phase_weights(raw_row: np.ndarray, chunk_width: int) -> np.ndarray:
    """Plain-Python model of this file's Phase A (chunked reduction, cache
    reassembly) + Phase B (cached select) -- mirrors the C++ design
    exactly: phase A never needs the whole row in one ciphertext, phase B
    needs no new decrypt."""
    row_length = len(raw_row)
    chunks = chunk_row(row_length, chunk_width)
    cached = np.concatenate([raw_row[s : s + c] for (s, c) in chunks])
    assert np.array_equal(cached, raw_row), "chunk reassembly must be exact"
    row_max = cached.max()
    row_sum = np.sum(np.exp(cached - row_max))
    return np.exp(cached - row_max) / row_sum


def _general_attention_reference(
    scores: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, int, int]:
    heads, tokens, _ = scores.shape
    head_dim = value.shape[2]
    context = np.zeros((heads, tokens, head_dim))
    context[:, 0, :] = value[:, 0, :]
    round_trips = 0
    score_calls = 0
    for row in range(1, tokens):
        row_length = row + 1
        score_calls += heads * row_length
        accumulated = value[:, 0, :].copy()
        for col in range(1, row + 1):
            round_trips += 1
            for head in range(heads):
                weight = _exact_softmax_select(scores[head, row, :row_length], col)
                accumulated[head] += weight * (value[head, col] - value[head, 0])
        context[:, row, :] = accumulated
    return context, round_trips, score_calls


class GeneralAttentionT103SourceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_t2_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(ORIGINAL_SOURCE), ORIGINAL_SHA256)

    def test_t3_anchor_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(T3_SOURCE), T3_SHA256)

    def test_t8_anchor_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(T8_SOURCE), T8_SHA256)

    def test_t32_anchor_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(T32_SOURCE), T32_SHA256)

    def test_t_constant_is_103(self) -> None:
        self.assertIn("constexpr std::size_t T = 103;", self.source)

    def test_pinned_fixture_manifest_matches_t103_fixture(self) -> None:
        self.assertIn(f'"{FIXTURE_MANIFEST_SHA256}"', self.source)

    def test_chunk_width_is_fixed_constant_not_derived_from_t(self) -> None:
        # The actual fix: CHUNK_WIDTH must not reference T anywhere in its
        # definition (unlike the T=32 file's SCORE_SLOT_STRIDE=T).
        self.assertIn(
            "constexpr std::size_t CHUNK_WIDTH = (PACK_WIDTH - D) / HEADS_PER_COPY;",
            self.source,
        )
        self.assertNotIn("CHUNK_WIDTH = T", self.source)
        # SCORE_SLOT_STRIDE may still appear in prose comments explaining
        # what was removed (it does); it must not be DECLARED as a constant
        # in this file.
        self.assertNotIn("std::size_t SCORE_SLOT_STRIDE", self.source)
        self.assertEqual(CHUNK_WIDTH, 85)

    def test_static_assert_uses_chunk_width_not_t(self) -> None:
        self.assertIn(
            "static_assert(HEADS_PER_COPY * CHUNK_WIDTH <= PACK_WIDTH - D,",
            self.source,
        )
        # This must hold regardless of T -- confirm no static_assert in this
        # file multiplies HEADS_PER_COPY (or HEADS) by T.
        self.assertNotIn("HEADS_PER_COPY * T <=", self.source)
        self.assertNotIn("HEADS * T <=", self.source)

    def test_chunk_row_helper_present(self) -> None:
        self.assertIn(
            "std::vector<ChunkRange> chunk_row(std::size_t row_length,",
            self.source,
        )

    def test_phase_a_reduce_chunk_is_identity_transform(self) -> None:
        self.assertIn("cross_boundary_reduce_chunk(", self.source)
        self.assertIn("row_score_cache_[row]", self.source)
        # Identity: the transform lambda must not overwrite `values` --
        # confirmed by the explicit comment and absence of a `values =`
        # reassignment inside that method's body.
        method_start = self.source.index("Ct cross_boundary_reduce_chunk(")
        method_body = self.source[method_start : method_start + 1800]
        self.assertIn("values` intentionally left unchanged", method_body)
        self.assertNotIn("values = std::move", method_body)
        self.assertNotIn("values = transformed", method_body)

    def test_finalize_row_softmax_is_not_a_boundary_crossing(self) -> None:
        self.assertIn(
            "void finalize_row_softmax(std::size_t row, std::size_t heads) {",
            self.source,
        )
        method_start = self.source.index(
            "void finalize_row_softmax(std::size_t row, std::size_t heads) {"
        )
        method_body = self.source[method_start : method_start + 900]
        self.assertNotIn("Decrypt(", method_body)
        self.assertNotIn("Encrypt(", method_body)
        self.assertNotIn("++round_trips_", method_body)

    def test_phase_b_emit_weight_has_no_decrypt_call(self) -> None:
        self.assertIn("Ct cross_boundary_emit_weight(", self.source)
        method_start = self.source.index("Ct cross_boundary_emit_weight(")
        method_end = self.source.index("\n    }\n", method_start)
        method_body = self.source[method_start:method_end]
        self.assertNotIn("Decrypt(", method_body)
        self.assertIn("cc_->Encrypt(keys_.publicKey, refreshed)", method_body)
        self.assertIn("++round_trips_", method_body)

    def test_clear_row_cache_present(self) -> None:
        self.assertIn("void clear_row_cache(std::size_t row) {", self.source)

    def test_causal_row_loop_calls_reduce_then_finalize_then_emit(self) -> None:
        # Ordering matters: all chunks for a row must be reduced before
        # finalize, and finalize before any emit_weight call for that row.
        loop_start = self.source.index("for (std::size_t row = 1; row < T; ++row) {")
        loop_region = self.source[loop_start : loop_start + 4000]
        reduce_idx = loop_region.index("cross_boundary_reduce_chunk(")
        finalize_idx = loop_region.index("finalize_row_softmax(row, HEADS);")
        emit_idx = loop_region.index("cross_boundary_emit_weight(")
        self.assertLess(reduce_idx, finalize_idx)
        self.assertLess(finalize_idx, emit_idx)

    def test_output_packing_fix_carried_forward_unchanged(self) -> None:
        self.assertIn(
            "constexpr std::size_t OUTPUT_GROUPS = (T + COPIES - 1) / COPIES;",
            self.source,
        )
        self.assertIn("if (local_slot >= COPIES) {", self.source)
        self.assertEqual((T + COPIES - 1) // COPIES, 26)

    def test_measure_expects_flat_t_by_d_layout(self) -> None:
        self.assertIn(
            "if (got.size() != T * D || reference.size() != T * D) {", self.source
        )

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_general_attention_t103", cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_general_attention_t103", build_text)

    def test_run_script_requires_tag_marker(self) -> None:
        text = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_general_attention_t103_", text)
        self.assertIn("_scheme_b_", text)
        self.assertIn(FIXTURE_MANIFEST_SHA256, text)

    def test_launch_script_requires_tag_marker(self) -> None:
        text = LAUNCH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_general_attention_t103_", text)
        self.assertIn("_scheme_b_", text)

    def test_wait_script_requires_tag_marker(self) -> None:
        text = WAIT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_general_attention_t103_", text)
        self.assertIn("_scheme_b_", text)
        self.assertIn("t103", text)

    def test_fixture_sha256_file_exists_and_matches_manifest(self) -> None:
        text = FIXTURE_SHA_FILE.read_text(encoding="utf-8")
        self.assertIn(FIXTURE_MANIFEST_SHA256, text)
        self.assertIn("manifest.json", text)


class ChunkRowPartitionTests(unittest.TestCase):
    """Pure-Python proof of chunk_row()'s bijection property, independent of
    any fixture -- must hold before any GPU run."""

    def test_partition_is_bijective_over_every_tested_row_length(self) -> None:
        for row_length in (1, 2, 9, 32, 84, 85, 86, 102, 103, 200, 1000):
            for chunk_width in (1, 5, CHUNK_WIDTH, CHUNK_WIDTH + 1):
                chunks = chunk_row(row_length, chunk_width)
                seen = set()
                for start, count in chunks:
                    self.assertLessEqual(count, chunk_width)
                    self.assertGreater(count, 0)
                    for col in range(start, start + count):
                        self.assertNotIn(col, seen, f"col {col} covered twice")
                        seen.add(col)
                self.assertEqual(seen, set(range(row_length)))

    def test_t103_max_row_length_needs_exactly_two_chunks(self) -> None:
        # row=102 (the last, longest row at T=103) has row_length=103,
        # which is >CHUNK_WIDTH=85 -- this is the row that would have been
        # IMPOSSIBLE under the T=32 file's packing scheme (T<=85 ceiling).
        chunks = chunk_row(103, CHUNK_WIDTH)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], (0, 85))
        self.assertEqual(chunks[1], (85, 18))


class TwoPhaseSoftmaxDesignTests(unittest.TestCase):
    def test_two_phase_matches_reference_softmax_on_synthetic_rows(self) -> None:
        rng = np.random.default_rng(0)
        for row_length in (1, 2, 9, 32, 85, 86, 103, 200, 1000):
            raw_row = rng.normal(size=row_length)
            weights = _two_phase_weights(raw_row, CHUNK_WIDTH)
            for col in (0, row_length // 2, row_length - 1):
                expected = _exact_softmax_select(raw_row, col)
                self.assertAlmostEqual(weights[col], expected, places=13)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_two_phase_matches_reference_softmax_on_real_t103_oracle_scores(
        self,
    ) -> None:
        scores = _load_array("oracle.attention_scores")  # (HEADS, T, T)
        for row in range(1, T):
            row_length = row + 1
            for head in range(HEADS):
                raw_row = scores[head, row, :row_length]
                weights = _two_phase_weights(raw_row, CHUNK_WIDTH)
                for col in range(row_length):
                    expected = _exact_softmax_select(raw_row, col)
                    self.assertAlmostEqual(
                        weights[col],
                        expected,
                        places=12,
                        msg=f"row={row} head={head} col={col}",
                    )

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_general_attention_matches_oracle_at_t103(self) -> None:
        scores = _load_array("oracle.attention_scores")
        value = _load_array("oracle.value")
        context_heads_oracle = _load_array("oracle.attention_context_heads")
        context_merged_oracle = _load_array("oracle.attention_context_merged")
        attn_proj_w = _load_array("weights.attn_proj")
        attn_proj_oracle = _load_array("oracle.attention_projection")

        context_heads, round_trips, score_calls = _general_attention_reference(
            scores, value
        )
        np.testing.assert_allclose(
            context_heads, context_heads_oracle, rtol=0, atol=1e-9
        )

        expected_round_trips = T * (T - 1) // 2
        expected_score_calls = HEADS * (T * (T + 1) // 2 - 1)
        self.assertEqual(round_trips, expected_round_trips)
        self.assertEqual(score_calls, expected_score_calls)

        heads, tokens, head_dim = context_heads.shape
        context_merged = context_heads.transpose(1, 0, 2).reshape(
            tokens, heads * head_dim
        )
        np.testing.assert_allclose(
            context_merged, context_merged_oracle, rtol=0, atol=1e-9
        )
        attn_proj = context_merged @ attn_proj_w.T
        np.testing.assert_allclose(attn_proj, attn_proj_oracle, rtol=0, atol=1e-6)

    def test_honest_round_trip_growth_formula(self) -> None:
        # Phase A crossings per row = ceil(row_length/CHUNK_WIDTH); Phase B
        # crossings per row = row (unchanged from every prior file). Plus T
        # LN1 crossings for the attention gate total.
        total_attention_only = 0
        for row in range(1, T):
            row_length = row + 1
            total_attention_only += math.ceil(row_length / CHUNK_WIDTH) + row
        self.assertEqual(total_attention_only, 5373)
        self.assertEqual(total_attention_only + T, 5476)


if __name__ == "__main__":
    unittest.main()
