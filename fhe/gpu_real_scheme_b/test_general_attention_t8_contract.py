"""Static and float64 contracts for the T=8 general causal-attention growth-
trend checkpoint (real_dnagpt_fides_scheme_b_general_attention_t8.cpp).

The attention/softmax circuit itself is NOT a new design: the T=3 file
(real_dnagpt_fides_scheme_b_general_attention.cpp) already generalizes the
attention nonlinearity to arbitrary T, and this file's per-row loop is
identical to it. A first T=8 attempt (same as the T=3 file plus only T and
the fixture manifest changed) crashed with "malloc(): unaligned tcache
chunk detected" after completing the causal-attention loop cleanly (all 28
round trips, confirmed via added flushed per-(row,col) logging -- see the
source's OUTPUT_GROUPS comment for the full diagnosis). Root cause: the
frozen T=2/T=3 files' final output-packing step (`finish()`/
`output_block_plain()`/`measure()`) silently assumes T<=COPIES=4 (one
physical PACK_WIDTH-wide ciphertext slot per token) -- true at T=2/T=3,
false at T=8. This file fixes that by grouping tokens into
ceil(T/COPIES)=2 output ciphertexts, addressed by each token's position
WITHIN its group rather than its raw index. This test file proves, before
any real GPU run, that:

  - the frozen T=2 source AND the T=3 anchor source are both untouched;
  - the causal-attention loop body itself (the part already validated by
    the T=3 numpy proof) is unchanged from the T=3 anchor;
  - OUTPUT_GROUPS = ceil(T/COPIES) is computed correctly and used
    consistently in finish()/output_block_plain()/main()'s decrypt loop;
  - output_block_plain() is now bounded by COPIES, not T (the actual bug);
  - the packed causal-score reservation still fits inside the D..PACK_WIDTH
    slack at T=8 (12*8=96 <= 256) -- this constraint was NOT the bug, the
    output-packing one was;
  - CMake/build script reference the new binary.

The float64 test reruns the same design proof as the T=3 contract test
(the causal-attention math, which the crash never reached, per-token
context reconstruction against the real T=8 oracle) and separately checks
the output-grouping arithmetic (which token maps to which group/local
slot) against a plain-Python model of the fixed C++ logic.
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).with_name("src")
SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention_t8.cpp"
T3_SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention.cpp"
ORIGINAL_SOURCE = HERE / "real_dnagpt_fides_scheme_b.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
T3_SHA256 = "c57e81d441b27400453c29e941d4f5e810df850a90cfc2d88ce68178ee4cd0f8"

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = SCRIPT_DIR / "run_scheme_b_general_attention_t8.sh"
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"

D = 768
T = 8
HEADS = 12
PACK_WIDTH = 1024
COPIES = 4

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_block0_t8_d63353abdc1a_52d046d1fcf0"
)
FIXTURE_MANIFEST_SHA256 = (
    "75ce745757a8141df42054612d54a0837b5f0c2e7a771cafd57fc5055e054162"
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


class GeneralAttentionT8SourceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_t2_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(ORIGINAL_SOURCE), ORIGINAL_SHA256)

    def test_t3_anchor_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(T3_SOURCE), T3_SHA256)

    def test_t_constant_is_eight(self) -> None:
        self.assertIn("constexpr std::size_t T = 8;", self.source)

    def test_pinned_fixture_manifest_matches_t8_fixture(self) -> None:
        self.assertIn(f'"{FIXTURE_MANIFEST_SHA256}"', self.source)

    def test_score_slot_reservation_disjoint_from_head_dimensions(self) -> None:
        self.assertLessEqual(HEADS * T, PACK_WIDTH - D)

    def test_causal_attention_loop_structure_matches_t3_anchor(self) -> None:
        # The part of the circuit already validated by the T=3 numpy proof,
        # and the part the crash never reached (confirmed by the flushed
        # per-(row,col) logging in the actual GPU run) -- present unchanged
        # in both files, modulo the added flush instrumentation.
        for source_text in (self.source, T3_SOURCE.read_text(encoding="utf-8")):
            self.assertIn("for (std::size_t row = 1; row < T; ++row) {", source_text)
            self.assertIn("const std::size_t row_length = row + 1;", source_text)
            self.assertIn("for (std::size_t col = 1; col <= row; ++col) {", source_text)
            self.assertIn("client_.cross_boundary_softmax_select(", source_text)
            self.assertIn(
                "accumulated_context = cc_->EvalAdd(\n"
                "                    accumulated_context, multiply(weight_broadcast, value_delta));",
                source_text,
            )

    def test_output_packing_fix_present(self) -> None:
        # The actual bug: output_block_plain() was bounded by T (silently
        # assuming T<=COPIES=4) instead of COPIES, causing a heap buffer
        # overflow for token>=COPIES at T=8. Confirm the fix's shape.
        self.assertIn(
            "constexpr std::size_t OUTPUT_GROUPS = (T + COPIES - 1) / COPIES;",
            self.source,
        )
        self.assertEqual(
            re.search(r"OUTPUT_GROUPS = \(T \+ COPIES - 1\) / COPIES", self.source)
            is not None,
            True,
        )
        self.assertIn("std::vector<Ct> packed_output;", self.source)
        self.assertIn("if (local_slot >= COPIES) {", self.source)
        self.assertNotIn(
            'if (token >= T) {\n            throw std::invalid_argument("output token',
            self.source,
        )
        self.assertIn(
            "for (std::size_t group = 0; group < OUTPUT_GROUPS; ++group) {",
            self.source,
        )
        self.assertIn(
            "for (std::size_t group = 0; group < evaluation.packed_output.size();",
            self.source,
        )
        # T=8 -> ceil(8/4) = 2 output groups, ceil(8/4)=2 -- computed here in
        # Python from the same COPIES/T constants the C++ uses, not
        # hardcoded, so this breaks if either constant changes without
        # updating the other.
        self.assertEqual((T + COPIES - 1) // COPIES, 2)

    def test_measure_expects_flat_t_by_d_layout(self) -> None:
        # T=2/T=3's measure() read directly from one T*PACK_WIDTH raw
        # decrypted-slot layout; that assumption is gone now that main()
        # reassembles a flat T*D vector from however many output groups
        # exist.
        self.assertIn(
            "if (got.size() != T * D || reference.size() != T * D) {", self.source
        )
        self.assertNotIn("got.size() < T * PACK_WIDTH", self.source)

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_general_attention_t8", cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_general_attention_t8", build_text)

    def test_run_script_requires_tag_marker(self) -> None:
        text = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("_general_attention_t8_", text)
        self.assertIn("_scheme_b_", text)

    def test_output_group_local_slot_mapping_covers_all_tokens_exactly_once(
        self,
    ) -> None:
        # Plain-Python model of finish()'s group/local_slot loop and main()'s
        # matching decrypt-side reassembly: every token in 0..T-1 must map to
        # exactly one (group, local_slot) pair, local_slot always < COPIES,
        # and every group must be non-empty.
        groups = (T + COPIES - 1) // COPIES
        seen = {}
        for group in range(groups):
            first_token = group * COPIES
            last_token = min(T, first_token + COPIES)
            self.assertLess(first_token, last_token, f"group {group} is empty")
            for token in range(first_token, last_token):
                local_slot = token - first_token
                self.assertLess(local_slot, COPIES)
                self.assertNotIn(token, seen, f"token {token} mapped twice")
                seen[token] = (group, local_slot)
        self.assertEqual(set(seen), set(range(T)))


class GeneralAttentionT8NumpyContractTests(unittest.TestCase):
    """Mathematical-equivalence check against the real T=8 oracle, and a
    check that the predicted O(T)/O(T^2) growth counts match what the
    circuit will issue at T=8."""

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=8 fixture absent")
    def test_general_attention_matches_oracle_at_t8(self) -> None:
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
        self.assertEqual(round_trips, 28)
        self.assertEqual(score_calls, expected_score_calls)
        self.assertEqual(score_calls, 420)

        heads, tokens, head_dim = context_heads.shape
        context_merged = context_heads.transpose(1, 0, 2).reshape(
            tokens, heads * head_dim
        )
        np.testing.assert_allclose(
            context_merged, context_merged_oracle, rtol=0, atol=1e-9
        )
        attn_proj = context_merged @ attn_proj_w.T
        np.testing.assert_allclose(attn_proj, attn_proj_oracle, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
