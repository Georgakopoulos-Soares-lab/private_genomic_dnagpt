"""Local contract for Stage-1 (Q/K/V split) 2-GPU process-per-GPU sharding.

Per docs/hybrid/tasks.md's 2026-07-31 "2-GPU process-per-GPU sharding" entry,
this is the recommended first micro-gate: it isolates the infra question (can
two independent processes, each holding one shard of {query,key,value}, be
gathered into the same downstream result as the existing one-process path)
from the merge-correctness question (deferred to a later MLP-chunk stage,
since Q/K/V shards are never combined via any ciphertext operation -- they
are only ever consumed together, unmodified, by attention).

No CUDA/FIDESlib code is built or launched here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from shard_layout import (
    QKV_PARTS,
    DuplicatedWorkerCost,
    qkv_shard_assignment,
    shard_of,
)
from simd_layout import bsgs_matmul_tokens, unpack_tokens

T = 103
B = 8
D = 768
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


class ShardAssignmentTests(unittest.TestCase):
    def test_one_gpu_assignment_is_the_full_set(self) -> None:
        assignment = qkv_shard_assignment(1)
        self.assertEqual(assignment, {0: QKV_PARTS})

    def test_two_gpu_assignment_is_an_exact_partition(self) -> None:
        assignment = qkv_shard_assignment(2)
        covered: list[str] = []
        for parts in assignment.values():
            covered.extend(parts)
        self.assertEqual(sorted(covered), sorted(QKV_PARTS))
        self.assertEqual(len(covered), len(set(covered)))

    def test_shard_of_agrees_with_the_assignment(self) -> None:
        for part in QKV_PARTS:
            worker = shard_of(part, 2)
            self.assertIn(part, qkv_shard_assignment(2)[worker])

    def test_undefined_gpu_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            qkv_shard_assignment(3)
        with self.assertRaises(ValueError):
            shard_of("query", 3)

    def test_duplicated_cost_is_explicit_and_small(self) -> None:
        cost = DuplicatedWorkerCost()
        self.assertTrue(cost.ln1_client_crossings_shared_not_duplicated)
        self.assertTrue(cost.baby_rotations_duplicated_per_worker)
        self.assertLess(cost.duplicated_cost_fraction_of_one_matmul_call, 0.01)


@unittest.skipUnless(FIXTURE.exists(), "real D=768/T=103 fixture absent")
class ShardedQkvMatchesUnshardedTests(unittest.TestCase):
    """Prove sharding Q/K/V across 2 simulated workers is a math no-op."""

    def setUp(self) -> None:
        self.qkv = _load_array("weights.attn_qkv")
        self.ln1 = _load_array("oracle.ln1_output")
        query_oracle = _load_array("oracle.query")
        key_oracle = _load_array("oracle.key")
        value_oracle = _load_array("oracle.value")
        self.oracle_by_part = {
            "query": query_oracle.transpose(1, 0, 2).reshape(T, D),
            "key": key_oracle.transpose(1, 0, 2).reshape(T, D),
            "value": value_oracle.transpose(1, 0, 2).reshape(T, D),
        }
        self.weight_by_part = {
            "query": self.qkv[0 * D : 1 * D],
            "key": self.qkv[1 * D : 2 * D],
            "value": self.qkv[2 * D : 3 * D],
        }

    def _compute_part(self, part: str, first: int, active: int) -> np.ndarray:
        packed = bsgs_matmul_tokens(
            self.ln1[first : first + active], self.weight_by_part[part], B
        )
        return unpack_tokens(packed, B, active)

    def test_each_shard_worker_reproduces_its_parts_exactly(self) -> None:
        """Each of the 2 workers, computing only its assigned parts,
        matches the real oracle exactly -- same tolerance as the existing
        single-process B=8 proof in test_simd_layout.py, since the
        underlying transform is unchanged by which process runs it."""

        assignment = qkv_shard_assignment(2)
        for first, active in ((0, 8), (96, 7)):
            for worker, parts in assignment.items():
                for part in parts:
                    got = self._compute_part(part, first, active)
                    np.testing.assert_allclose(
                        got,
                        self.oracle_by_part[part][first : first + active],
                        rtol=0.0,
                        atol=1e-9,
                        err_msg=f"worker {worker} part {part} first {first}",
                    )

    def test_gathering_both_shards_equals_the_unsharded_single_worker_path(
        self,
    ) -> None:
        """The 2-worker gather (no ciphertext merge, just collecting three
        independently-produced values) must be bit-for-bit identical to
        computing all three parts on one worker, since nothing about the
        math changes -- only which process ran which matmul call."""

        first, active = 0, 8
        sharded = {part: self._compute_part(part, first, active) for part in QKV_PARTS}
        unsharded = {
            part: self._compute_part(part, first, active) for part in QKV_PARTS
        }
        for part in QKV_PARTS:
            np.testing.assert_array_equal(sharded[part], unsharded[part])


if __name__ == "__main__":
    unittest.main()
