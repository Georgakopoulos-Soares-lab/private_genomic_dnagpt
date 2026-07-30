"""Local contracts for the additive Scheme B Token-SIMD design.

No CUDA/FIDESlib code is built or launched here.  These tests establish the
slot-level algebra and real-fixture oracle equivalence required before an
additive C++ fork is allowed.
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
    LOGICAL_SLOTS,
    MAX_COMPLEX_SLOTS,
    PACK_WIDTH,
    active_weight_shifts,
    bsgs_matmul_tokens,
    context_from_weight_tiles,
    dense_transform_count,
    expand_logical_plaintext,
    frozen_server_operation_counts,
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
B = 8
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


class TokenSimdSlotLayoutTests(unittest.TestCase):
    def test_batch_capacity_is_exactly_eight_at_ring_65536(self) -> None:
        validate_batch_width(8)
        with self.assertRaises(ValueError):
            validate_batch_width(9)
        self.assertEqual(MAX_COMPLEX_SLOTS // LOGICAL_SLOTS, 8)

    def test_physical_slot_mapping_is_a_bijection(self) -> None:
        seen = set()
        for copy in range(COPIES):
            for feature in range(PACK_WIDTH):
                for token in range(B):
                    slot = physical_slot(copy, feature, token, B)
                    self.assertNotIn(slot, seen)
                    seen.add(slot)
                    self.assertEqual(slot % B, token)
        self.assertEqual(seen, set(range(COPIES * PACK_WIDTH * B)))

    def test_pack_unpack_full_and_partial_batches(self) -> None:
        rng = np.random.default_rng(20260729)
        for active in (1, 2, 4, 7, 8):
            tokens = rng.normal(size=(active, D))
            packed = pack_replicated_tokens(tokens, B)
            got = unpack_tokens(packed, B, active)
            np.testing.assert_array_equal(got, tokens)
            slots = packed.reshape(COPIES, PACK_WIDTH, B)
            np.testing.assert_array_equal(slots[:, D:, :], 0.0)
            np.testing.assert_array_equal(slots[:, :, active:], 0.0)

    def test_scaled_rotations_preserve_every_token_lane(self) -> None:
        rng = np.random.default_rng(3)
        logical = rng.normal(size=(B, LOGICAL_SLOTS))
        packed = logical.T.reshape(-1)
        for amount in (-2048, -1024, -33, -1, 0, 1, 31, 32, 992, 1024, 2048):
            got = logical_rotate(packed, amount, B).reshape(LOGICAL_SLOTS, B).T
            expected = np.stack([rotate_left(row, amount) for row in logical])
            np.testing.assert_array_equal(got, expected)

    def test_expanded_plaintext_multiplies_each_token_independently(self) -> None:
        rng = np.random.default_rng(4)
        logical = rng.normal(size=(B, LOGICAL_SLOTS))
        plaintext = rng.normal(size=LOGICAL_SLOTS)
        packed = logical.T.reshape(-1)
        got = (
            (packed * expand_logical_plaintext(plaintext, B))
            .reshape(LOGICAL_SLOTS, B)
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
        rng = np.random.default_rng(5)
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


class TokenSimdDenseTransformTests(unittest.TestCase):
    def test_literal_bsgs_matches_independent_toy_matmuls(self) -> None:
        rng = np.random.default_rng(6)
        dimension = 11
        pack_width = 16
        copies = 4
        batch_width = 4
        tokens = rng.normal(size=(batch_width, dimension))
        weight = rng.normal(size=(dimension, dimension))
        packed = bsgs_matmul_tokens(
            tokens,
            weight,
            batch_width,
            dimension=dimension,
            pack_width=pack_width,
            copies=copies,
            bsgs_n1=4,
            max_complex_slots=batch_width * pack_width * copies,
        )
        got = unpack_tokens(
            packed,
            batch_width,
            batch_width,
            dimension=dimension,
            pack_width=pack_width,
            copies=copies,
        )
        np.testing.assert_allclose(got, tokens @ weight.T, rtol=0.0, atol=1e-12)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_real_q_projection_matches_eight_independent_tokens(self) -> None:
        qkv = _load_array("weights.attn_qkv")
        ln1 = _load_array("oracle.ln1_output")
        query_oracle = _load_array("oracle.query")
        expected = query_oracle.transpose(1, 0, 2).reshape(T, D)
        query_weight = qkv[:D]

        for first, active in ((0, 8), (96, 7)):
            packed = bsgs_matmul_tokens(ln1[first : first + active], query_weight, B)
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

        for first, active in ((0, 8), (96, 7)):
            for part, expected in enumerate(expected_by_part):
                packed = bsgs_matmul_tokens(
                    ln1[first : first + active],
                    qkv[part * D : (part + 1) * D],
                    B,
                )
                got = unpack_tokens(packed, B, active)
                np.testing.assert_allclose(
                    got,
                    expected[first : first + active],
                    rtol=0.0,
                    atol=1e-9,
                )

            packed_attention = bsgs_matmul_tokens(
                context[first : first + active], attention_weight, B
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
        for first, active in ((0, 8), (96, 7)):
            for chunk in range(COPIES):
                packed = bsgs_matmul_tokens(
                    normalized2[first : first + active],
                    fc_weight[chunk * D : (chunk + 1) * D],
                    B,
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
        for first, active in ((0, 8), (96, 7)):
            got = np.zeros((active, D), dtype=np.float64)
            for chunk in range(COPIES):
                packed = bsgs_matmul_tokens(
                    activated[
                        first : first + active,
                        chunk * D : (chunk + 1) * D,
                    ],
                    projection_weight[:, chunk * D : (chunk + 1) * D],
                    B,
                )
                got += unpack_tokens(packed, B, active)
            np.testing.assert_allclose(
                got,
                expected[first : first + active],
                rtol=0.0,
                atol=1e-9,
            )

    def test_transform_count_reduction_is_honest_for_partial_batch(self) -> None:
        self.assertEqual(math.ceil(T / B), 13)
        self.assertEqual(dense_transform_count(T, 1), 14832)
        self.assertEqual(dense_transform_count(T, B), 1872)
        self.assertAlmostEqual(14832 / 1872, 7.923076923076923)


class TokenSimdAttentionTileTests(unittest.TestCase):
    def test_score_tile_pack_round_trip(self) -> None:
        rng = np.random.default_rng(7)
        for query_count, key_count in ((8, 8), (7, 8), (7, 7), (1, 3)):
            scores = rng.normal(size=(query_count, key_count, HEADS))
            packed = pack_score_tile(scores, B)
            got = unpack_score_tile(packed, query_count, key_count, B)
            np.testing.assert_array_equal(got, scores)

    def test_score_tile_capacity_has_large_margin(self) -> None:
        self.assertEqual((HEADS // COPIES) * B, 24)
        self.assertLessEqual((HEADS // COPIES) * B, PACK_WIDTH - D)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
    def test_encrypted_layout_score_tiles_match_real_oracle(self) -> None:
        query = _load_array("oracle.query").transpose(1, 0, 2).reshape(T, D)
        key = _load_array("oracle.key").transpose(1, 0, 2).reshape(T, D)
        oracle = _load_array("oracle.attention_scores")
        cases = (
            (0, 8, 0, 8),
            (88, 8, 24, 8),
            (96, 7, 0, 8),
            (96, 7, 96, 7),
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
        self.assertEqual(counts.token_groups, 13)
        self.assertEqual(counts.score_tile_decryptions, 91)
        self.assertEqual(counts.weight_tile_encryptions, 727)
        self.assertEqual(counts.layernorm1_crossings, 13)
        self.assertEqual(counts.layernorm2_crossings, 13)
        self.assertEqual(counts.gelu_crossings, 13)
        self.assertEqual(counts.attention_gate_crossings, 831)
        self.assertEqual(counts.full_block_crossings, 857)
        self.assertAlmostEqual(5682 / counts.full_block_crossings, 6.6301050175)

    def test_projected_server_operation_schedule_is_explicit(self) -> None:
        counts = server_operation_counts(T, B)
        self.assertEqual(counts.token_groups, 13)
        self.assertEqual(counts.matrix_products, 156)
        self.assertEqual(counts.active_score_weight_alignments, 727)
        self.assertEqual(counts.cached_key_lane_shifts, 90)
        self.assertEqual(counts.cached_value_lane_shifts, 90)
        self.assertEqual(counts.ciphertext_ciphertext_multiplications, 1506)
        self.assertEqual(counts.ciphertext_plaintext_multiplications, 177734)
        self.assertEqual(counts.explicit_rotations, 8173)
        self.assertEqual(counts.accumulate_sum_calls, 8776)

    def test_frozen_count_formula_matches_t32_evidence_and_t103_projection(
        self,
    ) -> None:
        t32 = frozen_server_operation_counts(32)
        self.assertEqual(t32.matrix_products, 384)
        self.assertEqual(t32.ciphertext_ciphertext_multiplications, 6948)
        self.assertEqual(t32.ciphertext_plaintext_multiplications, 412668)

        frozen = frozen_server_operation_counts(T)
        packed = server_operation_counts(T, B)
        self.assertEqual(frozen.matrix_products, 1236)
        self.assertEqual(frozen.ciphertext_ciphertext_multiplications, 69925)
        self.assertEqual(frozen.ciphertext_plaintext_multiplications, 1459989)
        self.assertAlmostEqual(
            frozen.matrix_products / packed.matrix_products,
            7.923076923076923,
        )
        self.assertAlmostEqual(
            frozen.ciphertext_ciphertext_multiplications
            / packed.ciphertext_ciphertext_multiplications,
            46.43094289508632,
        )
        self.assertAlmostEqual(
            frozen.ciphertext_plaintext_multiplications
            / packed.ciphertext_plaintext_multiplications,
            8.21446093600549,
        )


if __name__ == "__main__":
    unittest.main()
