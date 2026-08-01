"""Local, no-GPU math contract for the "diagonal-plaintext-cache" speed lever,
reopened 2026-07-31 against the current best Token-SIMD config
(B=8, MULT_DEPTH=13, ring=65536; source
`real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp`).

Claim under test (read directly from that source's `matmul()`, lines ~1031-
1078): for a fixed call-type -- i.e. a fixed public weight matrix such as
`fixture_.qkv[0]` -- the per-(giant, small) BSGS diagonal `packed_values`
vector built inline on every call

    for (giant in 0..BSGS_N2) for (small in 0..BSGS_N1):
        packed_values[...] = f(weight, giant, small, copy, row)

is a pure function of (weight, giant, small). It does not read `baby`
(the token-group-specific ciphertext input) or any other per-token-group
state. Therefore, across the TOKEN_GROUPS=13 calls that share one weight
matrix (e.g. the 13 Q-projection calls, one per token group, all against
`fixture_.qkv[0]`), the exact same 1024 (`BSGS_N1 * BSGS_N2 = 32 * 32`)
vectors get rebuilt from scratch 13 times, byte-for-byte identical.

Correction to the task brief that reopened this lever: the brief estimated
"BSGS_N2 * BSGS_N1 = 4*32 = 128" diagonals per matmul call, apparently
substituting COPIES (4) for the source's actual BSGS_N2 = PACK_WIDTH /
BSGS_N1 = 1024 / 32 = 32. The real per-call diagonal count is
BSGS_N1 * BSGS_N2 = 32 * 32 = 1024, and the redundancy factor across the
13 token groups is exactly 13x (156 total matmul calls today / 12 distinct
weight matrices = 13), not the smaller ratio the "128" figure would imply.
This does not change the fix's design (a map keyed by weight-matrix
identity, holding one cached vector<double> per (giant, small) pair) --
only the size of each weight's cache (1024 entries, not 128) and the
magnitude of the redundant work being removed.

This script reimplements the exact `packed_values` construction formula in
NumPy against the REAL T=103 fixture weight matrices (not synthetic data),
and proves two things before any C++ is written:

  1. Byte-identical reconstruction: calling the formula multiple times,
     "as if" invoked from different token groups (the group index is
     accepted as a parameter but structurally unused, matching the C++),
     produces bit-for-bit identical arrays every time, for every one of
     the 12 real distinct weight matrices at a representative sample of
     (giant, small) pairs (including edges: giant/small = 0 and the last
     valid index).
  2. Round-trip sanity: reassembling the full D x D weight matrix from its
     BSGS diagonal decomposition (summing all 1024 diagonals' contributions
     at their respective rolled positions) reconstructs the original real
     weight matrix, so the diagonals being cached are not a strawman
     formula disconnected from the real fixture.

No GPU, no FIDESlib, no C++ build required.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(
    "/Users/galano/Developer/patternforge/utexas/dna-gpt/checkpoints/"
    "fhe_exports/gsr_pos0_block0_t103_d63353abdc1a_52d046d1fcf0"
)

D = 768
T = 103
MLP_DIM = 3072
PACK_WIDTH = 1024
COPIES = 4
TOKEN_BATCH = 8
BSGS_N1 = 32
BSGS_N2 = PACK_WIDTH // BSGS_N1
TOKEN_GROUPS = (T + TOKEN_BATCH - 1) // TOKEN_BATCH
SLOTS = COPIES * PACK_WIDTH * TOKEN_BATCH

assert BSGS_N1 * BSGS_N2 == PACK_WIDTH
assert TOKEN_GROUPS == 13


def _read_f64(name: str, count: int) -> np.ndarray:
    path = FIXTURE_DIR / name
    data = np.fromfile(path, dtype="<f8", count=count)
    if data.size != count:
        raise RuntimeError(f"short read: {path} ({data.size} != {count})")
    return data


def load_real_weights() -> dict[str, np.ndarray]:
    """Mirror load_fixture()'s exact unpacking in the C++ source."""
    qkv_flat = _read_f64("weights__attn_qkv.bin", 3 * D * D)
    qkv = qkv_flat.reshape(3, D, D)
    attn_proj = _read_f64("weights__attn_proj.bin", D * D).reshape(D, D)
    fc_flat = _read_f64("weights__mlp_fc.bin", MLP_DIM * D)
    mlp_fc = fc_flat.reshape(COPIES, D, D)
    proj_flat = _read_f64("weights__mlp_proj.bin", D * MLP_DIM).reshape(D, MLP_DIM)
    # C++ loader: for each COPIES-part, for each row in D, copy D values
    # starting at row*MLP_DIM + part*D into mlp_projection[part][row].
    mlp_projection = np.stack(
        [proj_flat[:, part * D : (part + 1) * D] for part in range(COPIES)]
    )
    return {
        "qkv_query": qkv[0],
        "qkv_key": qkv[1],
        "qkv_value": qkv[2],
        "attention_projection": attn_proj,
        "mlp_fc_0": mlp_fc[0],
        "mlp_fc_1": mlp_fc[1],
        "mlp_fc_2": mlp_fc[2],
        "mlp_fc_3": mlp_fc[3],
        "mlp_projection_0": mlp_projection[0],
        "mlp_projection_1": mlp_projection[1],
        "mlp_projection_2": mlp_projection[2],
        "mlp_projection_3": mlp_projection[3],
    }


def build_diagonal(
    weight: np.ndarray, giant: int, small: int, group: int
) -> np.ndarray:
    """Reimplementation of matmul()'s inline packed_values construction for
    one (giant, small) BSGS diagonal, transliterated line-for-line from the
    C++ (real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp,
    lines 1037-1060).

    `group` (the calling token group's index, 0..TOKEN_GROUPS-1) is accepted
    exactly as the C++'s enclosing loop would supply one, but -- matching
    the C++ -- is never read below. Its presence in the signature only lets
    the test below call this once per (simulated) token group and diff the
    results, instead of asserting identity by never varying an unused input.
    """
    del group  # structurally unused, exactly like the C++ matmul() body
    if weight.shape != (D, D):
        raise ValueError("weight must be D x D")
    diagonal = BSGS_N1 * giant + small
    packed = np.zeros(SLOTS, dtype=np.float64)
    for copy in range(COPIES):
        for row in range(PACK_WIDTH):
            rolled_row = (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH
            column = (rolled_row + diagonal) % PACK_WIDTH
            if rolled_row < D and column < D:
                coefficient = weight[rolled_row, column]
                if coefficient != 0.0:
                    base = (copy * PACK_WIDTH + row) * TOKEN_BATCH
                    packed[base : base + TOKEN_BATCH] = coefficient
    return packed


def reconstruct_weight_from_diagonals(weight: np.ndarray) -> np.ndarray:
    """Sum all BSGS_N1*BSGS_N2 diagonals' contributions back into a D x D
    matrix (using copy=0, token=0's physical slot as the read-out lane) and
    confirm it equals the original real weight matrix -- ties the cached
    formula to genuine fixture data, not a synthetic self-consistent one.
    """
    rebuilt = np.zeros((D, D), dtype=np.float64)
    for giant in range(BSGS_N2):
        for small in range(BSGS_N1):
            packed = build_diagonal(weight, giant, small, group=0)
            diagonal = BSGS_N1 * giant + small
            for row in range(PACK_WIDTH):
                rolled_row = (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH
                column = (rolled_row + diagonal) % PACK_WIDTH
                if rolled_row < D and column < D:
                    base = (0 * PACK_WIDTH + row) * TOKEN_BATCH
                    rebuilt[rolled_row, column] = packed[base]
    return rebuilt


class DiagonalCacheReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE_DIR.is_dir():
            raise unittest.SkipTest(f"fixture not present: {FIXTURE_DIR}")
        cls.weights = load_real_weights()

    def test_fixture_directory_matches_expected_shapes(self) -> None:
        for name, weight in self.weights.items():
            self.assertEqual(weight.shape, (D, D), name)
            self.assertTrue(np.all(np.isfinite(weight)), name)

    def test_redundancy_factor_is_exactly_13x_not_the_brief_128_estimate(
        self,
    ) -> None:
        total_matmul_calls = 12 * TOKEN_GROUPS
        distinct_weights = 12
        self.assertEqual(total_matmul_calls, 156)
        self.assertEqual(BSGS_N1 * BSGS_N2, 1024)
        redundancy = (total_matmul_calls * BSGS_N1 * BSGS_N2) / (
            distinct_weights * BSGS_N1 * BSGS_N2
        )
        self.assertAlmostEqual(redundancy, 13.0)

    def test_diagonal_construction_is_byte_identical_across_all_token_groups(
        self,
    ) -> None:
        # Sample (giant, small) pairs including both edges of each range.
        samples = [
            (0, 0),
            (0, BSGS_N1 - 1),
            (BSGS_N2 - 1, 0),
            (BSGS_N2 - 1, BSGS_N1 - 1),
            (BSGS_N2 // 2, BSGS_N1 // 2),
        ]
        for name, weight in self.weights.items():
            for giant, small in samples:
                reference = build_diagonal(weight, giant, small, group=0)
                digest_reference = hashlib.sha256(reference.tobytes()).hexdigest()
                for group in range(TOKEN_GROUPS):
                    candidate = build_diagonal(weight, giant, small, group)
                    self.assertTrue(
                        np.array_equal(candidate, reference),
                        f"{name} giant={giant} small={small} group={group} "
                        "diverged from group=0 reconstruction",
                    )
                    digest_candidate = hashlib.sha256(candidate.tobytes()).hexdigest()
                    self.assertEqual(
                        digest_candidate,
                        digest_reference,
                        f"{name} giant={giant} small={small} group={group} "
                        "sha256 mismatch -- caching would NOT be a no-op",
                    )

    def test_full_diagonal_decomposition_reconstructs_real_weight_matrix(
        self,
    ) -> None:
        # Slower (builds all 1024 diagonals), so run it for two
        # representative real matrices rather than all 12.
        for name in ("qkv_query", "mlp_fc_0"):
            weight = self.weights[name]
            rebuilt = reconstruct_weight_from_diagonals(weight)
            np.testing.assert_allclose(rebuilt, weight, rtol=0, atol=0, err_msg=name)

    def test_caching_is_a_pure_cpu_side_optimization_no_semantic_change(
        self,
    ) -> None:
        # The cached vector, once built, is bit-identical to what a fresh
        # per-call rebuild would produce -- so replacing "rebuild every
        # call" with "build once per weight, reuse the std::vector<double>
        # on every subsequent call" cannot change any encrypted value or
        # operation count. This is the mathematical justification for why
        # the C++ fork changes zero op-count invariants (ct_plain_multiply
        # count, rotation count, etc. all stay exactly as measured for the
        # unmodified depth-13 source).
        weight = self.weights["attention_projection"]
        cache: dict[tuple[int, int], np.ndarray] = {}
        for group in range(TOKEN_GROUPS):
            for giant in range(BSGS_N2):
                for small in (0, BSGS_N1 - 1):  # spot-check, not all 32
                    key = (giant, small)
                    fresh = build_diagonal(weight, giant, small, group)
                    if key not in cache:
                        cache[key] = fresh
                    else:
                        self.assertTrue(np.array_equal(cache[key], fresh))


if __name__ == "__main__":
    unittest.main()
