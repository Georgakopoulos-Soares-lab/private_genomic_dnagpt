"""Static and float64 contracts for the T=32 general causal-attention
checkpoint (real_dnagpt_fides_scheme_b_general_attention_t32.cpp).

This file fixes a SECOND, independent ceiling from the one the T=8 file
closed. The T=8 file fixed the output-packing bug (finish()/
output_block_plain() silently assumed T<=COPIES=4) and carries that fix
forward here unchanged. What's new in this file: the causal-score packing
ceiling flagged as still-open in docs/hybrid/tasks.md's 2026-07-29 entry --
the T=3/T=8 files' `one_hot_plain` isolates each head's raw causal score by
writing the SAME slot into all COPIES=4 physical copies (pure redundancy:
packed_scores never goes through sum_broadcast/cross-copy rotation and is
read directly), which forces all twelve heads' T-wide score reservations to
share ONE copy's D..PACK_WIDTH slack (256 slots):
HEADS*SCORE_SLOT_STRIDE<=PACK_WIDTH-D -> T<=21.

This file's fix: spread heads across copies instead of replicating.
HEADS_PER_COPY=HEADS/COPIES=3 heads live in each of the 4 copies (head h ->
copy h/3, local index h%3), each still getting a T-wide reservation within
that copy's own slack: HEADS_PER_COPY*T<=256 -> T<=85. The WRITE side of the
softmax-select boundary call (weight broadcast, multiplied against
value/value-delta ciphertexts, which ARE fully replicated across copies) is
UNCHANGED -- only the raw-score packing/read is made sparse.

This test file proves, before any real GPU run, that:

  - the frozen T=2, T=3, and T=8 sources are all untouched;
  - the new head->copy/local-slot assignment is a bijection covering all 12
    heads exactly once, with every local index < HEADS_PER_COPY;
  - the new packing ceiling (HEADS_PER_COPY*T<=PACK_WIDTH-D) actually raises
    the bound from T<=21 to T<=85, and T=32 sits comfortably inside it;
  - the sparse-per-copy read reproduces the OLD all-copies-redundant read's
    softmax weights bit-for-bit, both on synthetic data and against the
    real T=32 oracle's attention scores;
  - the (unchanged) write-side weight broadcast is unaffected by which
    packing scheme produced the weight;
  - the T=8 file's output-packing fix (OUTPUT_GROUPS) is carried forward
    unchanged, now producing ceil(32/4)=8 output groups;
  - CMake/build/run scripts reference the new binary and tag markers.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).with_name("src")
SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention_t32.cpp"
T8_SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention_t8.cpp"
T3_SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention.cpp"
ORIGINAL_SOURCE = HERE / "real_dnagpt_fides_scheme_b.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
T3_SHA256 = "c57e81d441b27400453c29e941d4f5e810df850a90cfc2d88ce68178ee4cd0f8"
T8_SHA256 = "9d4dde16a0803d29c7e97d517fac77a275057ad3258d99c3e260f65feca85545"

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = SCRIPT_DIR / "run_scheme_b_general_attention_t32.sh"
LAUNCH_SCRIPT = SCRIPT_DIR / "launch_brev_scheme_b_general_attention_t32.sh"
WAIT_SCRIPT = SCRIPT_DIR / "wait_and_run_scheme_b_general_attention_t32.sh"
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"
FIXTURE_SHA_FILE = SCRIPT_DIR / "fixture_t32.sha256"

D = 768
T = 32
HEADS = 12
PACK_WIDTH = 1024
COPIES = 4
HEADS_PER_COPY = HEADS // COPIES

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_block0_t32_d63353abdc1a_52d046d1fcf0"
)
FIXTURE_MANIFEST_SHA256 = (
    "6ae018b419a533dd9c77e6e0415bf7adf2b67c1f444013e68489a2fc84cca003"
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


def _exact_softmax_select(raw_row: np.ndarray, select_index: int) -> float:
    shifted = raw_row - np.max(raw_row)
    numerator = np.exp(shifted[select_index])
    denominator = np.sum(np.exp(shifted))
    return float(numerator / denominator)


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


def _old_redundant_packed(raw_scores: np.ndarray, score_slot_stride: int) -> np.ndarray:
    """Model of the T=3/T=8 files' one_hot_plain: head h's row replicated
    identically into ALL COPIES copies at slot h*stride+col."""
    heads, row_length = raw_scores.shape
    packed = np.zeros((COPIES, PACK_WIDTH))
    for h in range(heads):
        for col in range(row_length):
            for c in range(COPIES):
                packed[c, h * score_slot_stride + col] = raw_scores[h, col]
    return packed


def _new_sparse_packed(raw_scores: np.ndarray, score_slot_stride: int) -> np.ndarray:
    """Model of this file's one_hot_single_copy_plain: head h lives ONLY in
    copy h//HEADS_PER_COPY, at local slot (h%HEADS_PER_COPY)*stride+col."""
    heads, row_length = raw_scores.shape
    packed = np.zeros((COPIES, PACK_WIDTH))
    for h in range(heads):
        copy = h // HEADS_PER_COPY
        local = h % HEADS_PER_COPY
        for col in range(row_length):
            packed[copy, local * score_slot_stride + col] = raw_scores[h, col]
    return packed


def _old_read(
    packed: np.ndarray, row_length: int, select_index: int, score_slot_stride: int
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for h in range(HEADS):
        row = packed[0, h * score_slot_stride : h * score_slot_stride + row_length]
        weights[h] = _exact_softmax_select(row, select_index)
    return weights


def _new_read(
    packed: np.ndarray, row_length: int, select_index: int, score_slot_stride: int
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for h in range(HEADS):
        copy = h // HEADS_PER_COPY
        local = h % HEADS_PER_COPY
        start = local * score_slot_stride
        row = packed[copy, start : start + row_length]
        weights[h] = _exact_softmax_select(row, select_index)
    return weights


class GeneralAttentionT32SourceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_t2_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(ORIGINAL_SOURCE), ORIGINAL_SHA256)

    def test_t3_anchor_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(T3_SOURCE), T3_SHA256)

    def test_t8_anchor_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(T8_SOURCE), T8_SHA256)

    def test_t_constant_is_thirty_two(self) -> None:
        self.assertIn("constexpr std::size_t T = 32;", self.source)

    def test_pinned_fixture_manifest_matches_t32_fixture(self) -> None:
        self.assertIn(f'"{FIXTURE_MANIFEST_SHA256}"', self.source)

    def test_heads_per_copy_constant_present_and_correct(self) -> None:
        self.assertIn(
            "constexpr std::size_t HEADS_PER_COPY = HEADS / COPIES;", self.source
        )
        self.assertEqual(HEADS_PER_COPY, 3)

    def test_new_score_slot_static_assert_uses_heads_per_copy_not_heads(self) -> None:
        # The actual fix: the static_assert bound must be HEADS_PER_COPY, not
        # HEADS (the T=3/T=8 files' bound), or the packing ceiling is
        # unchanged.
        self.assertIn(
            "static_assert(HEADS_PER_COPY * SCORE_SLOT_STRIDE <= PACK_WIDTH - D,",
            self.source,
        )
        self.assertNotIn(
            "static_assert(HEADS * SCORE_SLOT_STRIDE <= PACK_WIDTH - D,", self.source
        )

    def test_packing_ceiling_raised_from_21_to_85(self) -> None:
        slack = PACK_WIDTH - D
        old_ceiling = slack // HEADS
        new_ceiling = slack // HEADS_PER_COPY
        self.assertEqual(old_ceiling, 21)
        self.assertEqual(new_ceiling, 85)
        self.assertLessEqual(T, new_ceiling)
        self.assertGreater(T, old_ceiling)  # proves this genuinely needed the fix

    def test_one_hot_single_copy_plain_present(self) -> None:
        self.assertIn(
            "Plaintext one_hot_single_copy_plain(std::size_t copy,",
            self.source,
        )
        # The old redundant-replication one-hot helper must be gone, not
        # merely renamed with the same body.
        self.assertNotIn(
            "Plaintext one_hot_plain(std::size_t slot_within_copy) {", self.source
        )

    def test_packed_scores_build_uses_copy_and_local_head(self) -> None:
        self.assertIn("const std::size_t copy = head / HEADS_PER_COPY;", self.source)
        self.assertIn(
            "const std::size_t local_head = head % HEADS_PER_COPY;", self.source
        )
        self.assertIn(
            "one_hot_single_copy_plain(\n                            copy, "
            "local_head * SCORE_SLOT_STRIDE + col)",
            self.source,
        )

    def test_softmax_select_signature_gains_heads_per_copy_param(self) -> None:
        self.assertIn(
            "std::size_t heads,\n"
            "                                     std::size_t heads_per_copy,\n"
            "                                     std::size_t read_stride,",
            self.source,
        )

    def test_softmax_select_call_site_passes_heads_per_copy(self) -> None:
        self.assertIn(
            "packed_scores, row_length, col, HEADS, HEADS_PER_COPY,\n"
            "                    SCORE_SLOT_STRIDE, HEAD_DIM, HEADS);",
            self.source,
        )

    def test_softmax_select_read_iterates_global_head_not_copy_then_head(self) -> None:
        # The read restructure: outer loop over head (not copy, then head as
        # in T=3/T=8), deriving (copy, local_head) per head.
        self.assertIn("for (std::size_t head = 0; head < heads; ++head) {", self.source)
        self.assertIn("const std::size_t copy = head / heads_per_copy;", self.source)
        self.assertIn(
            "const std::size_t local_head = head % heads_per_copy;", self.source
        )

    def test_softmax_select_write_side_still_replicated_across_all_copies(self) -> None:
        # The write side must remain unchanged: broadcast into every COPIES
        # copy (not just the head's assigned read copy).
        self.assertIn(
            "for (std::size_t out_copy = 0; out_copy < COPIES;\n"
            "                         ++out_copy) {",
            self.source,
        )
        self.assertIn(
            "const std::size_t write_start =\n"
            "                            out_copy * PACK_WIDTH + head * write_span;",
            self.source,
        )

    def test_causal_attention_loop_row_col_structure_matches_anchors(self) -> None:
        for source_text in (
            self.source,
            T8_SOURCE.read_text(encoding="utf-8"),
            T3_SOURCE.read_text(encoding="utf-8"),
        ):
            self.assertIn("for (std::size_t row = 1; row < T; ++row) {", source_text)
            self.assertIn("const std::size_t row_length = row + 1;", source_text)
            self.assertIn("for (std::size_t col = 1; col <= row; ++col) {", source_text)
            self.assertIn("client_.cross_boundary_softmax_select(", source_text)

    def test_output_packing_fix_carried_forward_unchanged(self) -> None:
        self.assertIn(
            "constexpr std::size_t OUTPUT_GROUPS = (T + COPIES - 1) / COPIES;",
            self.source,
        )
        self.assertIn("if (local_slot >= COPIES) {", self.source)
        self.assertEqual((T + COPIES - 1) // COPIES, 8)

    def test_measure_expects_flat_t_by_d_layout(self) -> None:
        self.assertIn(
            "if (got.size() != T * D || reference.size() != T * D) {", self.source
        )

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_general_attention_t32", cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_general_attention_t32", build_text)

    def test_run_script_requires_tag_marker(self) -> None:
        text = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_general_attention_t32_", text)
        self.assertIn("_scheme_b_", text)
        self.assertIn(FIXTURE_MANIFEST_SHA256, text)

    def test_launch_script_requires_tag_marker(self) -> None:
        text = LAUNCH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_general_attention_t32_", text)
        self.assertIn("_scheme_b_", text)

    def test_wait_script_requires_tag_marker(self) -> None:
        text = WAIT_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_general_attention_t32_", text)
        self.assertIn("_scheme_b_", text)
        self.assertIn("t32", text)

    def test_fixture_sha256_file_exists_and_matches_manifest(self) -> None:
        text = FIXTURE_SHA_FILE.read_text(encoding="utf-8")
        self.assertIn(FIXTURE_MANIFEST_SHA256, text)
        self.assertIn("manifest.json", text)


class GeneralAttentionT32HeadCopyAssignmentTests(unittest.TestCase):
    """Pure-Python proof of the new head->copy/local-slot assignment,
    independent of any fixture -- must hold before any GPU run."""

    def test_assignment_is_bijective_over_all_twelve_heads(self) -> None:
        seen: dict[tuple[int, int], int] = {}
        for head in range(HEADS):
            copy = head // HEADS_PER_COPY
            local = head % HEADS_PER_COPY
            self.assertLess(copy, COPIES)
            self.assertLess(local, HEADS_PER_COPY)
            self.assertNotIn((copy, local), seen)
            seen[(copy, local)] = head
        self.assertEqual(len(seen), HEADS)
        self.assertEqual(set(seen.values()), set(range(HEADS)))

    def test_every_copy_gets_exactly_heads_per_copy_heads(self) -> None:
        counts = [0] * COPIES
        for head in range(HEADS):
            counts[head // HEADS_PER_COPY] += 1
        self.assertEqual(counts, [HEADS_PER_COPY] * COPIES)

    def test_sparse_read_reproduces_redundant_read_on_synthetic_scores(self) -> None:
        rng = np.random.default_rng(0)
        for row_length, select_index in ((17, 5), (32, 31), (2, 1), (T, T - 1)):
            raw_scores = rng.normal(size=(HEADS, row_length))
            old_packed = _old_redundant_packed(raw_scores, T)
            new_packed = _new_sparse_packed(raw_scores, T)
            old_weights = _old_read(old_packed, row_length, select_index, T)
            new_weights = _new_read(new_packed, row_length, select_index, T)
            for head in range(HEADS):
                self.assertAlmostEqual(old_weights[head], new_weights[head], places=15)

    def test_write_side_broadcast_unaffected_by_packing_scheme(self) -> None:
        rng = np.random.default_rng(1)
        write_span = 64
        raw_scores = rng.normal(size=(HEADS, 9))
        old_packed = _old_redundant_packed(raw_scores, T)
        new_packed = _new_sparse_packed(raw_scores, T)
        old_weights = _old_read(old_packed, 9, 3, T)
        new_weights = _new_read(new_packed, 9, 3, T)
        for head in range(HEADS):
            old_broadcast = np.full((COPIES, write_span), old_weights[head])
            new_broadcast = np.full((COPIES, write_span), new_weights[head])
            np.testing.assert_array_equal(old_broadcast, new_broadcast)


class GeneralAttentionT32NumpyOracleTests(unittest.TestCase):
    """Mathematical-equivalence check against the real T=32 oracle: the
    causal-attention math itself (unchanged from T=3/T=8), plus the new
    sparse packing scheme applied to REAL per-head score data rather than
    synthetic random scores."""

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=32 fixture absent")
    def test_general_attention_matches_oracle_at_t32(self) -> None:
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
        self.assertEqual(round_trips, 496)
        self.assertEqual(score_calls, expected_score_calls)
        self.assertEqual(score_calls, 6324)

        heads, tokens, head_dim = context_heads.shape
        context_merged = context_heads.transpose(1, 0, 2).reshape(
            tokens, heads * head_dim
        )
        np.testing.assert_allclose(
            context_merged, context_merged_oracle, rtol=0, atol=1e-9
        )
        attn_proj = context_merged @ attn_proj_w.T
        np.testing.assert_allclose(attn_proj, attn_proj_oracle, rtol=0, atol=1e-6)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=32 fixture absent")
    def test_sparse_packing_reproduces_oracle_softmax_weights_per_row(self) -> None:
        # Applies the sparse-vs-redundant packing equivalence (already
        # proven on synthetic data above) to REAL per-head raw scores from
        # every non-trivial causal row in the actual T=32 oracle fixture --
        # not just one hand-picked row/column.
        scores = _load_array("oracle.attention_scores")  # (HEADS, T, T)
        for row in range(1, T):
            row_length = row + 1
            raw_scores = scores[:, row, :row_length]
            old_packed = _old_redundant_packed(raw_scores, T)
            new_packed = _new_sparse_packed(raw_scores, T)
            for col in range(1, row + 1):
                old_weights = _old_read(old_packed, row_length, col, T)
                new_weights = _new_read(new_packed, row_length, col, T)
                for head in range(HEADS):
                    self.assertAlmostEqual(
                        old_weights[head],
                        new_weights[head],
                        places=12,
                        msg=f"row={row} col={col} head={head}",
                    )


if __name__ == "__main__":
    unittest.main()
