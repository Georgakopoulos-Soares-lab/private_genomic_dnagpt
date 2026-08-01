"""Stage-1 process-per-GPU sharding: Q/K/V split assignment.

Grounded in the real C++ structure, not a guess: in
src/real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp's
EncryptedEvaluator::evaluate (the per-group loop around
"query[group] = matmul(baby, fixture_.qkv[0])"), one shared
``baby_rotations(normalized)`` value feeds all three of the Q/K/V matmuls
for a group. There is no ciphertext-arithmetic dependency *between* Q, K,
and V -- they are three independent dense transforms of the same input --
so this is a zero-merge sharding candidate per docs/hybrid/tasks.md's
2026-07-31 "2-GPU process-per-GPU sharding"
entry: the two shards' outputs are only ever consumed together downstream
(attention), never combined via any encrypted operation.

This module is a plaintext/structural model only. It does not touch
FIDESlib/CUDA and produces no performance evidence.
"""

from __future__ import annotations

QKV_PARTS: tuple[str, ...] = ("query", "key", "value")


def qkv_shard_assignment(num_gpus: int) -> dict[int, tuple[str, ...]]:
    """Deterministic, exhaustive assignment of {query,key,value} to workers.

    2-GPU split puts query+key on one worker (they are consumed together,
    immediately, for the causal attention score tiles) and value alone on
    the other worker (it is only consumed later, after the client returns
    softmax weights) -- chosen so the two shards' *downstream* consumers
    are also naturally staggered, not just their own compute.
    """

    if num_gpus == 1:
        return {0: QKV_PARTS}
    if num_gpus == 2:
        return {0: ("query", "key"), 1: ("value",)}
    raise ValueError(f"no defined shard assignment for num_gpus={num_gpus}")


def shard_of(part: str, num_gpus: int) -> int:
    assignment = qkv_shard_assignment(num_gpus)
    for worker, parts in assignment.items():
        if part in parts:
            return worker
    raise ValueError(f"part {part!r} not assigned for num_gpus={num_gpus}")


# ---- Stage 2: MLP down-projection chunk split (the merge stage) ----
#
# Grounded in the real C++ structure: in the depth-13 full-block source's
# EncryptedEvaluator::evaluate, the MLP down-projection is
#     Ct mlp;
#     for (chunk = 0..COPIES-1)
#         mlp = chunk==0 ? matmul(baby(hidden[chunk]), mlp_projection[chunk])
#                        : EvalAdd(mlp, matmul(...));
#     block_output = EvalAdd(residual1, mlp);
# i.e. COPIES=4 independent D x D matrix products whose results are SUMMED
# (EvalAdd) into the same output ciphertext. Unlike Stage 1's Q/K/V (three
# independent output ciphertexts, never combined), this is a genuine
# partial-sum accumulation -- so sharding it REQUIRES an exact ciphertext
# EvalAdd merge of the workers' partial sums. That merge is the Stage-2
# novelty. Because EvalAdd is exact and associative, any partition of the four
# chunks reproduces the full sum; the 2-and-2 split assigns two chunks per
# worker.
MLP_CHUNKS: tuple[int, ...] = (0, 1, 2, 3)


def mlp_chunk_shard_assignment(num_gpus: int) -> dict[int, tuple[int, ...]]:
    """Deterministic, exhaustive 2-and-2 assignment of the four MLP
    down-projection chunks to workers.

    The 2-GPU split gives worker 0 the primary half {0,1} and worker 1 the
    half {2,3}. Worker 0 is "primary": it additionally serializes residual1,
    which the merge step needs to form
    block_output = residual1 + (partial_{0,1} + partial_{2,3}). The two
    partial sums are combined with a single exact EvalAdd per token group.
    """

    if num_gpus == 1:
        return {0: MLP_CHUNKS}
    if num_gpus == 2:
        return {0: (0, 1), 1: (2, 3)}
    raise ValueError(f"no defined MLP chunk assignment for num_gpus={num_gpus}")


def mlp_shard_of(chunk: int, num_gpus: int) -> int:
    if chunk not in MLP_CHUNKS:
        raise ValueError(f"unknown MLP chunk {chunk!r}")
    assignment = mlp_chunk_shard_assignment(num_gpus)
    for worker, chunks in assignment.items():
        if chunk in chunks:
            return worker
    raise ValueError(f"chunk {chunk!r} not assigned for num_gpus={num_gpus}")


def mlp_chunk_tag(chunks: tuple[int, ...]) -> str:
    """Concatenated chunk-index tag used to name a worker's serialized
    partials, matching the C++ reader's Options::chunk_tag()."""

    return "".join(str(chunk) for chunk in chunks)


class DuplicatedWorkerCost:
    """What each worker in the 2-GPU split must independently (re)compute.

    Both workers still need per-group LN1 + baby_rotations, because a
    worker only receives ciphertexts, not the ability to invoke the other
    worker's already-computed rotation state without another
    serialize/deserialize round trip -- and cross-process ciphertext
    reload was already measured net-negative for this project
    (fhe_fides_real_d768_t2_full_serialized_reader_scheme_b_a100_20260727:
    reader deserialize+LoadContext, 7.88s, cost more than a fresh
    single-process setup, 5.86s, at that context size). Recomputing
    baby_rotations locally on each worker avoids paying that reload cost a
    second time.

    The LN1 *client* round trip itself is NOT duplicated: LN1 is a
    client-side decrypt/exact-invsqrt/re-encrypt boundary performed once by
    the single trusted client, which can hand the same re-encrypted
    ciphertext to both GPU workers (plain data copy, no extra crypto
    operation) rather than have each worker request its own LN1 boundary.
    """

    ln1_client_crossings_shared_not_duplicated: bool = True
    baby_rotations_duplicated_per_worker: bool = True
    # Per fhe_fides_real_d768_t2_full_profiled_scheme_b_a100_20260727:
    # rotation/keyswitch cost is <0.1% of one matmul() call's wall time.
    # Duplicating baby_rotations across 2 workers therefore duplicates a
    # cost that is negligible relative to the matmul work it feeds.
    duplicated_cost_fraction_of_one_matmul_call: float = 0.001
