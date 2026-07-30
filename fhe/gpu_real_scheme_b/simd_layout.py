"""Local, plaintext model of a future Scheme B Token-SIMD layout.

This is not an FHE implementation and does not produce performance evidence.
It models the exact slot permutations, masks, BSGS diagonal transform, tiled
attention scores, and weighted-value reconstruction that an additive C++
prototype must reproduce before a GPU gate is allowed.

The key layout is interleaved:

    physical(copy, feature, token) =
        (copy * pack_width + feature) * batch_width + token

Tokens are therefore the innermost slot dimension.  A logical rotation by k
slots in the frozen one-token layout becomes a physical rotation by
batch_width*k and cannot cross token lanes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


D = 768
HEADS = 12
HEAD_DIM = 64
PACK_WIDTH = 1024
COPIES = 4
LOGICAL_SLOTS = COPIES * PACK_WIDTH
BSGS_N1 = 32
BSGS_N2 = PACK_WIDTH // BSGS_N1
RING_DIM = 65536
MAX_COMPLEX_SLOTS = RING_DIM // 2


def validate_batch_width(
    batch_width: int,
    *,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
    max_complex_slots: int = MAX_COMPLEX_SLOTS,
) -> None:
    if batch_width <= 0:
        raise ValueError("batch_width must be positive")
    required = batch_width * pack_width * copies
    if required > max_complex_slots:
        raise ValueError(
            f"batch_width={batch_width} needs {required} slots, "
            f"exceeding {max_complex_slots}"
        )


def physical_slot(
    copy: int,
    feature: int,
    token: int,
    batch_width: int,
    *,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
) -> int:
    validate_batch_width(
        batch_width,
        pack_width=pack_width,
        copies=copies,
        max_complex_slots=batch_width * pack_width * copies,
    )
    if not 0 <= copy < copies:
        raise IndexError("copy outside layout")
    if not 0 <= feature < pack_width:
        raise IndexError("feature outside pack width")
    if not 0 <= token < batch_width:
        raise IndexError("token outside batch width")
    return (copy * pack_width + feature) * batch_width + token


def _token_matrix(tokens: np.ndarray, dimension: int) -> np.ndarray:
    matrix = np.asarray(tokens, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"tokens must have shape [active,{dimension}]")
    if not np.isfinite(matrix).all():
        raise ValueError("tokens contain a non-finite value")
    return matrix


def pack_replicated_tokens(
    tokens: np.ndarray,
    batch_width: int,
    *,
    dimension: int = D,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
    max_complex_slots: int = MAX_COMPLEX_SLOTS,
) -> np.ndarray:
    """Pack <=B token vectors, replicating each across all BSGS copies."""

    validate_batch_width(
        batch_width,
        pack_width=pack_width,
        copies=copies,
        max_complex_slots=max_complex_slots,
    )
    matrix = _token_matrix(tokens, dimension)
    if matrix.shape[0] > batch_width:
        raise ValueError("active token count exceeds batch width")
    slots = np.zeros((copies, pack_width, batch_width), dtype=np.float64)
    slots[:, :dimension, : matrix.shape[0]] = matrix.T[None, :, :]
    return slots.reshape(-1)


def unpack_tokens(
    packed: np.ndarray,
    batch_width: int,
    active_tokens: int,
    *,
    dimension: int = D,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
    require_replicated: bool = True,
) -> np.ndarray:
    if not 0 <= active_tokens <= batch_width:
        raise ValueError("active_tokens outside batch")
    values = np.asarray(packed, dtype=np.float64)
    expected = copies * pack_width * batch_width
    if values.shape != (expected,):
        raise ValueError(f"packed input must contain {expected} slots")
    slots = values.reshape(copies, pack_width, batch_width)
    if require_replicated:
        for copy in range(1, copies):
            np.testing.assert_allclose(slots[copy], slots[0], rtol=0.0, atol=0.0)
    return slots[0, :dimension, :active_tokens].T.copy()


def rotate_left(values: np.ndarray, amount: int) -> np.ndarray:
    """OpenFHE/FIDES logical left rotation used by existing contracts."""

    return np.roll(np.asarray(values, dtype=np.float64), -amount)


def logical_rotate(
    packed: np.ndarray, logical_amount: int, batch_width: int
) -> np.ndarray:
    """Apply one frozen-layout rotation without changing token identity."""

    return rotate_left(packed, batch_width * logical_amount)


def expand_logical_plaintext(
    logical_values: np.ndarray, batch_width: int
) -> np.ndarray:
    """Repeat a public logical coefficient for every packed token lane."""

    values = np.asarray(logical_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("logical plaintext must be one-dimensional")
    return np.repeat(values, batch_width)


def lane_shift(packed: np.ndarray, token_amount: int, batch_width: int) -> np.ndarray:
    """Cyclically shift tokens inside every logical slot.

    A raw rotation by ``token_amount`` crosses a logical-slot boundary for
    wrapped token lanes.  Two rotations plus complementary public masks repair
    that wrap:

      output[logical, t] = input[logical, (t + token_amount) mod B]

    This is the plaintext model of two EvalRotate calls, two ciphertext-
    plaintext masks, and one add.
    """

    values = np.asarray(packed, dtype=np.float64)
    if values.ndim != 1 or values.size % batch_width:
        raise ValueError("packed length must be divisible by batch width")
    amount = token_amount % batch_width
    if amount == 0:
        return values.copy()
    lane = np.arange(values.size) % batch_width
    no_wrap = lane < batch_width - amount
    from_next_logical = rotate_left(values, amount)
    from_same_logical = rotate_left(values, amount - batch_width)
    return np.where(no_wrap, from_next_logical, from_same_logical)


def _bsgs_diagonal(
    weight: np.ndarray,
    giant: int,
    small: int,
    *,
    dimension: int,
    pack_width: int,
    bsgs_n1: int,
) -> np.ndarray:
    row = np.arange(pack_width)
    rolled_row = (row + pack_width - bsgs_n1 * giant) % pack_width
    diagonal = bsgs_n1 * giant + small
    column = (rolled_row + diagonal) % pack_width
    result = np.zeros(pack_width, dtype=np.float64)
    valid = (rolled_row < dimension) & (column < dimension)
    result[valid] = weight[rolled_row[valid], column[valid]]
    return result


def bsgs_matmul_tokens(
    tokens: np.ndarray,
    weight: np.ndarray,
    batch_width: int,
    *,
    dimension: int = D,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
    bsgs_n1: int = BSGS_N1,
    max_complex_slots: int = MAX_COMPLEX_SLOTS,
) -> np.ndarray:
    """Literal NumPy replica of the scaled-rotation packed BSGS transform."""

    matrix = np.asarray(weight, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"weight must have shape {(dimension, dimension)}, got {matrix.shape}"
        )
    if pack_width % bsgs_n1:
        raise ValueError("pack_width must divide evenly by bsgs_n1")
    bsgs_n2 = pack_width // bsgs_n1
    packed = pack_replicated_tokens(
        tokens,
        batch_width,
        dimension=dimension,
        pack_width=pack_width,
        copies=copies,
        max_complex_slots=max_complex_slots,
    )
    baby = [logical_rotate(packed, small, batch_width) for small in range(bsgs_n1)]
    result = np.zeros_like(packed)
    for giant in range(bsgs_n2):
        inner = np.zeros_like(packed)
        for small in range(bsgs_n1):
            one_copy = _bsgs_diagonal(
                matrix,
                giant,
                small,
                dimension=dimension,
                pack_width=pack_width,
                bsgs_n1=bsgs_n1,
            )
            logical_plain = np.tile(one_copy, copies)
            inner += baby[small] * expand_logical_plaintext(logical_plain, batch_width)
        if giant:
            inner = logical_rotate(inner, bsgs_n1 * giant, batch_width)
        result += inner
    return result


def score_tile(
    query_tokens: np.ndarray,
    key_tokens: np.ndarray,
    batch_width: int,
    *,
    heads: int = HEADS,
    head_dim: int = HEAD_DIM,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
) -> np.ndarray:
    """Compute one BxB encrypted-score tile through lane alignments."""

    dimension = heads * head_dim
    query = _token_matrix(query_tokens, dimension)
    key = _token_matrix(key_tokens, dimension)
    if query.shape[0] > batch_width or key.shape[0] > batch_width:
        raise ValueError("score tile exceeds batch width")
    max_slots = batch_width * copies * pack_width
    query_packed = pack_replicated_tokens(
        query,
        batch_width,
        dimension=dimension,
        pack_width=pack_width,
        copies=copies,
        max_complex_slots=max_slots,
    )
    key_packed = pack_replicated_tokens(
        key,
        batch_width,
        dimension=dimension,
        pack_width=pack_width,
        copies=copies,
        max_complex_slots=max_slots,
    )
    scores = np.zeros((query.shape[0], key.shape[0], heads), dtype=np.float64)
    scale = 1.0 / math.sqrt(head_dim)
    for delta in range(batch_width):
        aligned_key = lane_shift(key_packed, delta, batch_width)
        products = unpack_tokens(
            query_packed * aligned_key,
            batch_width,
            query.shape[0],
            dimension=dimension,
            pack_width=pack_width,
            copies=copies,
        )
        reduced = products.reshape(query.shape[0], heads, head_dim).sum(axis=2)
        reduced *= scale
        for query_lane in range(query.shape[0]):
            key_lane = (query_lane + delta) % batch_width
            if key_lane < key.shape[0]:
                scores[query_lane, key_lane] = reduced[query_lane]
    return scores


def pack_score_tile(
    scores: np.ndarray,
    batch_width: int,
    *,
    heads: int = HEADS,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
) -> np.ndarray:
    """Pack a score tile into one ciphertext-shaped slot vector.

    Heads are spread across copies.  Each query occupies one token lane and
    stores all key lanes for a local head in adjacent logical score slots.
    """

    tile = np.asarray(scores, dtype=np.float64)
    if tile.ndim != 3 or tile.shape[2] != heads:
        raise ValueError("scores must have shape [queries,keys,heads]")
    query_count, key_count, _ = tile.shape
    if query_count > batch_width or key_count > batch_width:
        raise ValueError("score tile exceeds batch width")
    if heads % copies:
        raise ValueError("heads must divide evenly across copies")
    heads_per_copy = heads // copies
    if heads_per_copy * batch_width > pack_width:
        raise ValueError("score tile does not fit within one copy")
    packed = np.zeros((copies, pack_width, batch_width), dtype=np.float64)
    for query_lane in range(query_count):
        for key_lane in range(key_count):
            for head in range(heads):
                copy = head // heads_per_copy
                local_head = head % heads_per_copy
                feature = local_head * batch_width + key_lane
                packed[copy, feature, query_lane] = tile[query_lane, key_lane, head]
    return packed.reshape(-1)


def unpack_score_tile(
    packed: np.ndarray,
    query_count: int,
    key_count: int,
    batch_width: int,
    *,
    heads: int = HEADS,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
) -> np.ndarray:
    values = np.asarray(packed, dtype=np.float64)
    expected = copies * pack_width * batch_width
    if values.shape != (expected,):
        raise ValueError(f"packed score tile must contain {expected} slots")
    if heads % copies:
        raise ValueError("heads must divide evenly across copies")
    heads_per_copy = heads // copies
    slots = values.reshape(copies, pack_width, batch_width)
    scores = np.zeros((query_count, key_count, heads), dtype=np.float64)
    for query_lane in range(query_count):
        for key_lane in range(key_count):
            for head in range(heads):
                copy = head // heads_per_copy
                local_head = head % heads_per_copy
                feature = local_head * batch_width + key_lane
                scores[query_lane, key_lane, head] = slots[copy, feature, query_lane]
    return scores


def stable_causal_weights(scores: np.ndarray) -> np.ndarray:
    """Stable full-row softmax for [heads,T,T] raw attention scores."""

    raw = np.asarray(scores, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1] != raw.shape[2]:
        raise ValueError("scores must have shape [heads,T,T]")
    _, tokens, _ = raw.shape
    weights = np.zeros_like(raw)
    for row in range(tokens):
        causal = raw[:, row, : row + 1]
        maximum = causal.max(axis=1, keepdims=True)
        exponentials = np.exp(causal - maximum)
        weights[:, row, : row + 1] = exponentials / exponentials.sum(
            axis=1, keepdims=True
        )
    return weights


def active_weight_shifts(
    query_start: int,
    query_count: int,
    key_start: int,
    key_count: int,
    batch_width: int,
) -> tuple[int, ...]:
    """Return lane shifts that contain at least one active causal pair."""

    active: list[int] = []
    for delta in range(batch_width):
        found = False
        for query_lane in range(query_count):
            key_lane = (query_lane + delta) % batch_width
            if (
                key_lane < key_count
                and key_start + key_lane <= query_start + query_lane
            ):
                found = True
                break
        if found:
            active.append(delta)
    return tuple(active)


def context_from_weight_tiles(
    weights: np.ndarray,
    values: np.ndarray,
    batch_width: int,
    *,
    pack_width: int = PACK_WIDTH,
    copies: int = COPIES,
) -> np.ndarray:
    """Reconstruct encrypted attention context via packed weight/V tiles."""

    softmax = np.asarray(weights, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    if softmax.ndim != 3:
        raise ValueError("weights must have shape [heads,T,T]")
    heads, tokens, key_tokens = softmax.shape
    if tokens != key_tokens:
        raise ValueError("weights must be square in token dimensions")
    if value.shape[:2] != (heads, tokens):
        raise ValueError("values must have shape [heads,T,head_dim]")
    head_dim = value.shape[2]
    dimension = heads * head_dim
    groups = math.ceil(tokens / batch_width)
    output = np.zeros((tokens, dimension), dtype=np.float64)
    max_slots = batch_width * copies * pack_width

    for query_group in range(groups):
        query_start = query_group * batch_width
        query_count = min(batch_width, tokens - query_start)
        accumulated = np.zeros(copies * pack_width * batch_width, dtype=np.float64)
        for key_group in range(query_group + 1):
            key_start = key_group * batch_width
            key_count = min(batch_width, tokens - key_start)
            key_vectors = (
                value[:, key_start : key_start + key_count, :]
                .transpose(1, 0, 2)
                .reshape(key_count, dimension)
            )
            packed_value = pack_replicated_tokens(
                key_vectors,
                batch_width,
                dimension=dimension,
                pack_width=pack_width,
                copies=copies,
                max_complex_slots=max_slots,
            )
            for delta in active_weight_shifts(
                query_start,
                query_count,
                key_start,
                key_count,
                batch_width,
            ):
                weight_vectors = np.zeros((query_count, dimension), dtype=np.float64)
                for query_lane in range(query_count):
                    row = query_start + query_lane
                    key_lane = (query_lane + delta) % batch_width
                    if key_lane >= key_count:
                        continue
                    col = key_start + key_lane
                    if col > row:
                        continue
                    for head in range(heads):
                        first = head * head_dim
                        weight_vectors[query_lane, first : first + head_dim] = softmax[
                            head, row, col
                        ]
                packed_weight = pack_replicated_tokens(
                    weight_vectors,
                    batch_width,
                    dimension=dimension,
                    pack_width=pack_width,
                    copies=copies,
                    max_complex_slots=max_slots,
                )
                accumulated += packed_weight * lane_shift(
                    packed_value, delta, batch_width
                )
        output[query_start : query_start + query_count] = unpack_tokens(
            accumulated,
            batch_width,
            query_count,
            dimension=dimension,
            pack_width=pack_width,
            copies=copies,
        )
    return output


@dataclass(frozen=True)
class ProtocolCounts:
    token_groups: int
    score_tile_decryptions: int
    weight_tile_encryptions: int
    layernorm1_crossings: int
    layernorm2_crossings: int
    gelu_crossings: int

    @property
    def attention_gate_crossings(self) -> int:
        return (
            self.score_tile_decryptions
            + self.weight_tile_encryptions
            + self.layernorm1_crossings
        )

    @property
    def full_block_crossings(self) -> int:
        return (
            self.attention_gate_crossings
            + self.layernorm2_crossings
            + self.gelu_crossings
        )


def protocol_counts(tokens: int, batch_width: int) -> ProtocolCounts:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    groups = math.ceil(tokens / batch_width)
    score_tiles = 0
    weight_encryptions = 0
    for query_group in range(groups):
        query_start = query_group * batch_width
        query_count = min(batch_width, tokens - query_start)
        for key_group in range(query_group + 1):
            key_start = key_group * batch_width
            key_count = min(batch_width, tokens - key_start)
            score_tiles += 1
            weight_encryptions += len(
                active_weight_shifts(
                    query_start,
                    query_count,
                    key_start,
                    key_count,
                    batch_width,
                )
            )
    return ProtocolCounts(
        token_groups=groups,
        score_tile_decryptions=score_tiles,
        weight_tile_encryptions=weight_encryptions,
        layernorm1_crossings=groups,
        layernorm2_crossings=groups,
        gelu_crossings=groups,
    )


@dataclass(frozen=True)
class ServerOperationCounts:
    """Projected operations for one packed full block.

    ``explicit_rotations`` excludes rotations internal to AccumulateSum; those
    calls are reported separately.  Key/value lane permutations are cached per
    key group and reused across query groups.
    """

    token_groups: int
    matrix_products: int
    ciphertext_ciphertext_multiplications: int
    ciphertext_plaintext_multiplications: int
    explicit_rotations: int
    accumulate_sum_calls: int
    active_score_weight_alignments: int
    cached_key_lane_shifts: int
    cached_value_lane_shifts: int


@dataclass(frozen=True)
class FrozenServerOperationCounts:
    """Source-derived full-block counts for the one-token layout."""

    matrix_products: int
    ciphertext_ciphertext_multiplications: int
    ciphertext_plaintext_multiplications: int


def frozen_server_operation_counts(
    tokens: int, *, pack_width: int = PACK_WIDTH, heads: int = HEADS
) -> FrozenServerOperationCounts:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    matrix_products = 12 * tokens
    raw_score_calls = heads * (tokens * (tokens + 1) // 2 - 1)
    context_weight_products = tokens * (tokens - 1) // 2
    # One q*k per raw score, one weight*V per nontrivial causal column, and
    # four LayerNorm ct-ct products per token.
    ct_ct = raw_score_calls + context_weight_products + 4 * tokens
    # Every raw score uses head-mask, scale, and isolation plaintexts.
    # LN1/LN2 use six plaintexts/token, MLP copy selection uses eight, and
    # final output packing one: 15 additional plaintexts/token.
    ct_plain = matrix_products * pack_width + 3 * raw_score_calls + 15 * tokens
    return FrozenServerOperationCounts(
        matrix_products=matrix_products,
        ciphertext_ciphertext_multiplications=ct_ct,
        ciphertext_plaintext_multiplications=ct_plain,
    )


def _unique_cached_lane_shifts(tokens: int, batch_width: int) -> int:
    groups = math.ceil(tokens / batch_width)
    by_key_group: list[set[int]] = [set() for _ in range(groups)]
    for query_group in range(groups):
        query_start = query_group * batch_width
        query_count = min(batch_width, tokens - query_start)
        for key_group in range(query_group + 1):
            key_start = key_group * batch_width
            key_count = min(batch_width, tokens - key_start)
            by_key_group[key_group].update(
                delta
                for delta in active_weight_shifts(
                    query_start,
                    query_count,
                    key_start,
                    key_count,
                    batch_width,
                )
                if delta != 0
            )
    return sum(len(shifts) for shifts in by_key_group)


def server_operation_counts(
    tokens: int,
    batch_width: int,
    *,
    pack_width: int = PACK_WIDTH,
    bsgs_n1: int = BSGS_N1,
) -> ServerOperationCounts:
    """Count the first full-block Token-SIMD server schedule honestly."""

    counts = protocol_counts(tokens, batch_width)
    groups = counts.token_groups
    alignments = counts.weight_tile_encryptions
    matrix_products = 12 * groups
    giant_steps = pack_width // bsgs_n1

    # Per group: QKV share one baby set, attention projection one, MLP FC
    # chunks share one, and four independently packed MLP projection inputs
    # need four sets: seven baby sets total.
    dense_baby_rotations = groups * 7 * (bsgs_n1 - 1)
    dense_giant_rotations = matrix_products * (giant_steps - 1)

    cached_shifts = _unique_cached_lane_shifts(tokens, batch_width)
    # Each nonzero lane shift is two rotations and two public masks.  Key and
    # value shifts are separate ciphertexts but each is reused by all queries.
    lane_repair_rotations = 4 * cached_shifts
    lane_repair_plain_multiplications = 4 * cached_shifts

    # One encrypted q*k product and one encrypted weight*V product per active
    # tile alignment. LayerNorm has centered^2 and centered*inverse for both
    # LN1 and LN2: four ct-ct products per token group.
    ct_ct = 2 * alignments + 4 * groups

    dense_plain = matrix_products * pack_width
    # Per score alignment and head: one head mask before reduction and one
    # score-slot isolation mask after reduction.
    score_plain = 2 * HEADS * alignments
    # Four FC outputs are selected into copies and four activated chunks are
    # selected back out: eight masks per group.
    mlp_copy_plain = 8 * groups
    # Each LayerNorm has two mean scalings plus one learned gamma plaintext.
    layernorm_plain = 6 * groups
    ct_plain = (
        dense_plain
        + score_plain
        + lane_repair_plain_multiplications
        + mlp_copy_plain
        + layernorm_plain
    )

    # GELU's four selected chunks are restored across four copies. The frozen
    # implementation uses three explicit cross-copy rotations per chunk.
    mlp_copy_rotations = 12 * groups
    explicit_rotations = (
        dense_baby_rotations
        + dense_giant_rotations
        + lane_repair_rotations
        + mlp_copy_rotations
    )

    # Twelve per-head reductions for every score alignment, plus two mean
    # reductions in each of two LayerNorms per token group.
    accumulate_sum_calls = HEADS * alignments + 4 * groups

    return ServerOperationCounts(
        token_groups=groups,
        matrix_products=matrix_products,
        ciphertext_ciphertext_multiplications=ct_ct,
        ciphertext_plaintext_multiplications=ct_plain,
        explicit_rotations=explicit_rotations,
        accumulate_sum_calls=accumulate_sum_calls,
        active_score_weight_alignments=alignments,
        cached_key_lane_shifts=cached_shifts,
        cached_value_lane_shifts=cached_shifts,
    )


def dense_transform_count(tokens: int, batch_width: int, blocks: int = 12) -> int:
    if tokens <= 0 or blocks <= 0:
        raise ValueError("tokens and blocks must be positive")
    return blocks * 12 * math.ceil(tokens / batch_width)
