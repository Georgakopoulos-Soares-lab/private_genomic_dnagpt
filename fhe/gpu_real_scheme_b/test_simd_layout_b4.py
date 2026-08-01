"""Local contracts for the additive Scheme B Token-SIMD design at B=4.

Mirrors test_simd_layout.py's B=8 contract suite exactly, but at batch_width=4
and (per docs/hybrid/roadmap.md item 4 / simd_current_state.txt section 7) the
smaller ring dimension 32768 that B=4 is intended to run at. simd_layout.py
is already generic over batch_width, so this file adds no new implementation
-- it proves the same properties hold at the other candidate packing width
before any C++ work, per docs/roadmap.md's "test one change at a time" and
this repo's go/no-go gate discipline (results/README.md, docs/hybrid/roadmap.md
"Required local correctness" / "Required projected leverage").

No CUDA/FIDESlib code is built or launched here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from simd_layout import (
    COPIES,
    D,
    HEADS,
    PACK_WIDTH,
    active_weight_shifts,
    bsgs_matmul_tokens,
    context_from_weight_tiles,
    dense_transform_count,
    expand_logical_plaintext,
    lane_shift,
    logical_rotate,
    pack_replicated_tokens,
    pack_score_tile,
    physical_slot,
    protocol_counts,
    rotate_left,
    score_tile,
    server_operation_counts,
    stable_causal_weights,
    unpack_score_tile,
    unpack_tokens,
    validate_batch_width,
)

T = 103
B = 4
RING_DIM_B4 = 32768
MAX_COMPLEX_SLOTS_B4 = RING_DIM_B4 // 2
LOGICAL_SLOTS_B4 = COPIES * PACK_WIDTH

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_array(name: str) -> np.ndarray:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["arrays"][name]
    path = FIXTURE / metadata["file"]
    if _sha256(path) != metadata["sha256"]:
        raise ValueError(f"fixture hash mismatch for {name}")
    return np.fromfile(path, dtype="<f8").reshape(metadata["shape"])


class TokenSimdB4CapacityTests(unittest.TestCase):
    def test_batch_capacity_is_exactly_four_at_ring_32768(self) -> None:
        """B=4 must exactly saturate ring=32768, mirroring B=8/ring=65536.

        docs/hybrid/roadmap.md item 4 states a smaller ring would require a
        separate B<=4 layout; this proves the exact match, not an
        approximation, at the honest smaller ring rather than reusing the
        existing ring=65536/32768-slot context (which would under-use the
        halved ring and not test the claim this comparison is meant to make).
        """

        validate_batch_width(4, max_complex_slots=MAX_COMPLEX_SLOTS_B4)
        with self.assertRaises(ValueError):
            validate_batch_width(5, max_complex_slots=MAX_COMPLEX_SLOTS_B4)
        self.assertEqual(MAX_COMPLEX_SLOTS_B4 // LOGICAL_SLOTS_B4, 4)

    def test_batch_capacity_also_fits_the_existing_ring_65536_context(self) -> None:
        """B=4 also fits (with slack) inside the already-provisioned B=8
        context, in case the go/no-go gate wants to isolate packing width
        from ring/depth as a second variable."""

        validate_batch_width(4)


class TokenSimdB4SlotLayoutTests(unittest.TestCase):
    def test_physical_slot_mapping_is_a_bijection(self) -> None:
        seen = set()
        for copy in range(COPIES):
            for feature in range(PACK_WIDTH):
                for token in range(B):
                    slot = physical_slot(copy, feature, token, B, copies=COPIES)
                    self.assertNotIn(slot, seen)
                    seen.add(slot)
                    self.assertEqual(slot % B, token)
        self.assertEqual(seen, set(range(COPIES * PACK_WIDTH * B)))

    def test_pack_unpack_full_and_partial_batches(self) -> None:
        rng = np.random.default_rng(20260731)
        for active in (1, 2, 3, 4):
            tokens = rng.normal(size=(active, D))
            packed = pack_replicated_tokens(
                tokens, B, max_complex_slots=MAX_COMPLEX_SLOTS_B4
            )
            got = unpack_tokens(packed, B, active)
            np.testing.assert_array_equal(got, tokens)
            slots = packed.reshape(COPIES, PACK_WIDTH, B)
            np.testing.assert_array_equal(slots[:, D:, :], 0.0)
            np.testing.assert_array_equal(slots[:, :, active:], 0.0)

    def test_scaled_rotations_preserve_every_token_lane(self) -> None:
        rng = np.random.default_rng(31)
        logical = rng.normal(size=(B, LOGICAL_SLOTS_B4))
        packed = logical.T.reshape(-1)
        for amount in (-2048, -1024, -33, -1, 0, 1, 31, 32, 992, 1024, 2048):
            got = logical_rotate(packed, amount, B).reshape(LOGICAL_SLOTS_B4, B).T
            expected = np.stack([rotate_left(row, amount) for row in logical])
            np.testing.assert_array_equal(got, expected)

    def test_expanded_plaintext_multiplies_each_token_independently(self) -> None:
        rng = np.random.default_rng(32)
        logical = rng.normal(size=(B, LOGICAL_SLOTS_B4))
        plaintext = rng.normal(size=LOGICAL_SLOTS_B4)
        packed = logical.T.reshape(-1)
        got = (
            (packed * expand_logical_plaintext(plaintext, B))
            .reshape(LOGICAL_SLOTS_B4, B)
            .T
        )
        np.testing.assert_array_equal(got, logical * plaintext)

    def test_lane_shift_repairs_raw_rotation_wrap(self) -> None:
        logical = np.arange(19 * B, dtype=np.float64).reshape(19, B)
        packed = logical.reshape(-1)
        for amount in range(-2 * B, 2 * B + 1):
            got = lane_shift(packed, amount, B).reshape(19, B)
            expected = np.roll(logical, -(amount % B), axis=1)
            np.testing.assert_array_equal(got, expected)

    def test_lane_and_logical_rotations_commute(self) -> None:
        rng = np.random.default_rng(33)
        packed = rng.normal(size=COPIES * PACK_WIDTH * B)
        for logical_amount in (-1024, -31, 1, 32, 1024):
            for token_amount in range(B):
                first = lane_shift(
                    logical_rotate(packed, logical_amount, B), token_amount, B
                )
                second = logical_rotate(
                    lane_shift(packed, token_amount, B), logical_amount, B
                )
                np.testing.assert_array_equal(first, second)


class TokenSimdB4DenseTransformTests(unittest.TestCase):
    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_real_q_projection_matches_four_independent_tokens(self) -> None:
        qkv = _load_array("weights.attn_qkv")
        ln1 = _load_array("oracle.ln1_output")
        query_oracle = _load_array("oracle.query")
        expected = query_oracle.transpose(1, 0, 2).reshape(T, D)
        query_weight = qkv[:D]

        for first, active in ((0, 4), (100, 3)):
            packed = bsgs_matmul_tokens(
                ln1[first : first + active],
                query_weight,
                B,
                max_complex_slots=MAX_COMPLEX_SLOTS_B4,
            )
            got = unpack_tokens(packed, B, active)
            np.testing.assert_allclose(
                got,
                expected[first : first + active],
                rtol=0.0,
                atol=1e-9,
            )

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_real_k_v_and_attention_projection_match_oracles(self) -> None:
        qkv = _load_array("weights.attn_qkv")
        ln1 = _load_array("oracle.ln1_output")
        context = _load_array("oracle.attention_context_merged")
        attention_weight = _load_array("weights.attn_proj")
        expected_attention = _load_array("oracle.attention_projection")
        expected_by_part = (
            _load_array("oracle.query").transpose(1, 0, 2).reshape(T, D),
            _load_array("oracle.key").transpose(1, 0, 2).reshape(T, D),
            _load_array("oracle.value").transpose(1, 0, 2).reshape(T, D),
        )

        for first, active in ((0, 4), (100, 3)):
            for part, expected in enumerate(expected_by_part):
                packed = bsgs_matmul_tokens(
                    ln1[first : first + active],
                    qkv[part * D : (part + 1) * D],
                    B,
                    max_complex_slots=MAX_COMPLEX_SLOTS_B4,
                )
                got = unpack_tokens(packed, B, active)
                np.testing.assert_allclose(
                    got,
                    expected[first : first + active],
                    rtol=0.0,
                    atol=1e-9,
                )

            packed_attention = bsgs_matmul_tokens(
                context[first : first + active],
                attention_weight,
                B,
                max_complex_slots=MAX_COMPLEX_SLOTS_B4,
            )
            got_attention = unpack_tokens(packed_attention, B, active)
            np.testing.assert_allclose(
                got_attention,
                expected_attention[first : first + active],
                rtol=0.0,
                atol=1e-9,
            )

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_real_mlp_fc_chunks_match_oracle(self) -> None:
        normalized2 = _load_array("oracle.ln2_output")
        fc_weight = _load_array("weights.mlp_fc")
        expected = _load_array("oracle.mlp_fc")
        for first, active in ((0, 4), (100, 3)):
            for chunk in range(COPIES):
                packed = bsgs_matmul_tokens(
                    normalized2[first : first + active],
                    fc_weight[chunk * D : (chunk + 1) * D],
                    B,
                    max_complex_slots=MAX_COMPLEX_SLOTS_B4,
                )
                got = unpack_tokens(packed, B, active)
                np.testing.assert_allclose(
                    got,
                    expected[
                        first : first + active,
                        chunk * D : (chunk + 1) * D,
                    ],
                    rtol=0.0,
                    atol=1e-9,
                )

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_real_mlp_projection_chunks_sum_to_oracle(self) -> None:
        activated = _load_array("oracle.gelu_tanh")
        projection_weight = _load_array("weights.mlp_proj")
        expected = _load_array("oracle.mlp_projection")
        for first, active in ((0, 4), (100, 3)):
            got = np.zeros((active, D), dtype=np.float64)
            for chunk in range(COPIES):
                packed = bsgs_matmul_tokens(
                    activated[
                        first : first + active,
                        chunk * D : (chunk + 1) * D,
                    ],
                    projection_weight[:, chunk * D : (chunk + 1) * D],
                    B,
                    max_complex_slots=MAX_COMPLEX_SLOTS_B4,
                )
                got += unpack_tokens(packed, B, active)
            np.testing.assert_allclose(
                got,
                expected[first : first + active],
                rtol=0.0,
                atol=1e-9,
            )

    def test_transform_count_reduction_is_honest_for_partial_batch(self) -> None:
        self.assertEqual(math.ceil(T / B), 26)
        self.assertEqual(dense_transform_count(T, 1), 14832)
        self.assertEqual(dense_transform_count(T, B), 3744)
        self.assertAlmostEqual(14832 / 3744, 3.9615384615384617)


class TokenSimdB4AttentionTileTests(unittest.TestCase):
    def test_score_tile_pack_round_trip(self) -> None:
        rng = np.random.default_rng(71)
        for query_count, key_count in ((4, 4), (3, 4), (3, 3), (1, 2)):
            scores = rng.normal(size=(query_count, key_count, HEADS))
            packed = pack_score_tile(scores, B)
            got = unpack_score_tile(packed, query_count, key_count, B)
            np.testing.assert_array_equal(got, scores)

    def test_score_tile_capacity_has_large_margin(self) -> None:
        self.assertEqual((HEADS // COPIES) * B, 12)
        self.assertLessEqual((HEADS // COPIES) * B, PACK_WIDTH - D)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_encrypted_layout_score_tiles_match_real_oracle(self) -> None:
        query = _load_array("oracle.query").transpose(1, 0, 2).reshape(T, D)
        key = _load_array("oracle.key").transpose(1, 0, 2).reshape(T, D)
        oracle = _load_array("oracle.attention_scores")
        cases = (
            (0, 4, 0, 4),
            (44, 4, 12, 4),
            (100, 3, 0, 4),
            (100, 3, 100, 3),
        )
        for query_start, query_count, key_start, key_count in cases:
            got = score_tile(
                query[query_start : query_start + query_count],
                key[key_start : key_start + key_count],
                B,
            )
            expected = oracle[
                :,
                query_start : query_start + query_count,
                key_start : key_start + key_count,
            ].transpose(1, 2, 0)
            np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
            packed = pack_score_tile(got, B)
            recovered = unpack_score_tile(packed, query_count, key_count, B)
            np.testing.assert_array_equal(recovered, got)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_tiled_softmax_weighted_values_match_real_context(self) -> None:
        scores = _load_array("oracle.attention_scores")
        values = _load_array("oracle.value")
        expected = _load_array("oracle.attention_context_merged")
        weights = stable_causal_weights(scores)
        got = context_from_weight_tiles(weights, values, B)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)

    def test_causal_lane_shifts_cover_every_pair_exactly_once(self) -> None:
        covered: list[tuple[int, int]] = []
        groups = math.ceil(T / B)
        for query_group in range(groups):
            query_start = query_group * B
            query_count = min(B, T - query_start)
            for key_group in range(query_group + 1):
                key_start = key_group * B
                key_count = min(B, T - key_start)
                for delta in active_weight_shifts(
                    query_start, query_count, key_start, key_count, B
                ):
                    for query_lane in range(query_count):
                        key_lane = (query_lane + delta) % B
                        if key_lane >= key_count:
                            continue
                        row = query_start + query_lane
                        col = key_start + key_lane
                        if col <= row:
                            covered.append((row, col))
        expected = [(row, col) for row in range(T) for col in range(row + 1)]
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(set(covered), set(expected))

    def test_protocol_count_reduction_is_explicit(self) -> None:
        counts = protocol_counts(T, B)
        self.assertEqual(counts.token_groups, 26)
        self.assertEqual(counts.layernorm1_crossings, 26)
        self.assertEqual(counts.layernorm2_crossings, 26)
        self.assertEqual(counts.gelu_crossings, 26)
        # These are the actual leverage numbers this task cares about: do
        # B=4's extra score tiles/weight encryptions eat into the dense
        # 3.96x reduction once attention crossings are counted honestly.
        self.assertGreater(counts.score_tile_decryptions, 91)
        self.assertGreater(counts.weight_tile_encryptions, 727)

    def test_projected_server_operation_schedule_is_explicit(self) -> None:
        counts = server_operation_counts(T, B)
        self.assertEqual(counts.token_groups, 26)
        self.assertEqual(counts.matrix_products, 312)


if __name__ == "__main__":
    unittest.main()
