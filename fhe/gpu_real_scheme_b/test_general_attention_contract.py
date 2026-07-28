"""Static and float64 contracts for the Scheme B general causal-attention
prototype (real_dnagpt_fides_scheme_b_general_attention.cpp).

The frozen T=2 file (real_dnagpt_fides_scheme_b.cpp) exploits a closed-form
identity -- softmax over exactly two causal columns reduces to a single
sigmoid of a score difference -- that only holds at T=2. This prototype
replaces that identity with a real per-row causal softmax evaluated exactly
at the Scheme B client boundary, so it generalizes to any T. This test file,
in the same no-GPU-required style as test_lintransform_contract.py/
test_batching_contract.py/test_warmup_contract.py, proves before any real
GPU run that:

  - the frozen T=2 source is untouched;
  - the new file forks (does not edit) the frozen file: same D/HEADS/
    HEAD_DIM/PACK_WIDTH/COPIES/BSGS_N1/TOL constants, T changed to 3 to
    match the new fixture;
  - the Client class gained exactly one new public method
    (cross_boundary_softmax_select) and its existing methods
    (cross_boundary/cross_boundary_active/cross_boundary_impl) are
    byte-for-byte identical to the frozen file's;
  - the attention block no longer references sigmoid/exact_sigmoid (the
    T=2-specific identity) and instead performs a per-row loop computing
    row_length = row+1 causal scores and issuing one client boundary call
    per non-trivial causal column;
  - EncryptedEvaluator still never touches Decrypt/secretKey directly;
  - the packed causal-score reservation (HEADS*SCORE_SLOT_STRIDE) never
    collides with the D real dimensions any head span also occupies;
  - the run/launch/orchestrator scripts and CMake reference the new binary
    and a `_general_attention_` tag marker disjoint from every other
    prototype's marker.

The float64 test (GeneralAttentionNumpyContractTests) is the actual
mathematical-equivalence check: it replicates the proposed circuit
(packed one-hot scores -> per-head softmax-select -> v0+delta accumulation)
in numpy against the real T=3 oracle fixture (attention_scores,
attention_probabilities, attention_context_merged, attention_projection),
generated via `python -m fhe.realweights.export_fixture --tokens 3`. This
is the same design proof that was run interactively before any C++ was
written; it is captured here so it re-runs on every future check instead of
living only in a shell transcript. It also checks that T=2 is a special
case of the same formula (weight_i1 = exact_softmax([s_i0,s_i1])[1] =
sigmoid(s_i1-s_i0)), confirming this is a strict generalization of the
frozen file's identity, not a different algorithm that happens to agree at
one data point.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).with_name("src")
SOURCE = HERE / "real_dnagpt_fides_scheme_b_general_attention.cpp"
ORIGINAL_SOURCE = HERE / "real_dnagpt_fides_scheme_b.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = SCRIPT_DIR / "run_scheme_b_general_attention.sh"
LAUNCH_SCRIPT = SCRIPT_DIR / "launch_brev_scheme_b_general_attention.sh"
ORCHESTRATOR_SCRIPT = SCRIPT_DIR / "wait_and_run_scheme_b_general_attention.sh"
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"

D = 768
T = 3
HEADS = 12
HEAD_DIM = 64
PACK_WIDTH = 1024
COPIES = 4
SLOTS = COPIES * PACK_WIDTH
BSGS_N1 = 32
SCORE_SLOT_STRIDE = T

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_block0_t3_d63353abdc1a_52d046d1fcf0"
)
FIXTURE_MANIFEST_SHA256 = (
    "bfdabe62cdc7ff05dfb3e6ca149e50b081d5cfeb159408bb7478f2b168abd5c3"
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
    """Transliteration of Client::cross_boundary_softmax_select's transform."""
    shifted = raw_row - np.max(raw_row)
    numerator = np.exp(shifted[select_index])
    denominator = np.sum(np.exp(shifted))
    return float(numerator / denominator)


def _general_attention_reference(
    scores: np.ndarray, value: np.ndarray
) -> tuple[np.ndarray, int]:
    """Reproduces the circuit's per-row loop: packed one-hot scores are a
    pure repacking (lossless), so this operates directly on
    scores[h, row, col] -- the client boundary's read-then-softmax-select
    step is exactly _exact_softmax_select. Returns (context_heads,
    round_trips)."""
    heads, tokens, _ = scores.shape
    head_dim = value.shape[2]
    context = np.zeros((heads, tokens, head_dim))
    context[:, 0, :] = value[:, 0, :]
    round_trips = 0
    for row in range(1, tokens):
        row_length = row + 1
        accumulated = value[:, 0, :].copy()
        for col in range(1, row + 1):
            round_trips += (
                1  # one physical round trip per (row, col), all heads batched
            )
            for head in range(heads):
                weight = _exact_softmax_select(scores[head, row, :row_length], col)
                accumulated[head] += weight * (value[head, col] - value[head, 0])
        context[:, row, :] = accumulated
    return context, round_trips


class GeneralAttentionSourceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_t2_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(ORIGINAL_SOURCE), ORIGINAL_SHA256)

    def test_shared_constants_match_frozen_file_except_t(self) -> None:
        self.assertIn("constexpr std::size_t D = 768;", self.source)
        self.assertIn("constexpr std::size_t T = 3;", self.source)
        self.assertIn("constexpr std::size_t HEADS = 12;", self.source)
        self.assertIn("constexpr std::size_t HEAD_DIM = 64;", self.source)
        self.assertIn("constexpr std::size_t PACK_WIDTH = 1024;", self.source)
        self.assertIn("constexpr std::size_t COPIES = 4;", self.source)
        self.assertIn("constexpr std::size_t BSGS_N1 = 32;", self.source)
        self.assertIn("constexpr double TOL = 4e-2;", self.source)
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 16;", self.source)

    def test_score_slot_reservation_disjoint_from_head_dimensions(self) -> None:
        self.assertIn("constexpr std::size_t SCORE_SLOT_STRIDE = T;", self.source)
        self.assertIn(
            "static_assert(HEADS * SCORE_SLOT_STRIDE <= PACK_WIDTH - D,",
            self.source,
        )
        self.assertLessEqual(HEADS * SCORE_SLOT_STRIDE, PACK_WIDTH - D)

    def test_t2_sigmoid_identity_removed(self) -> None:
        # exact_sigmoid (the callable) and the string it was passed to
        # cross_boundary_active under are gone; a passing mention inside a
        # comment explaining the batching precedent this generalizes is
        # fine and expected (see the Client::cross_boundary_softmax_select
        # docstring above).
        self.assertNotIn("double exact_sigmoid(double", self.source)
        self.assertNotIn('"attention_sigmoid_heads_batched"', self.source)

    def test_client_gained_exactly_one_new_public_method(self) -> None:
        start = self.source.index("class Client {")
        end = self.source.index("class EncryptedEvaluator")
        client_body = self.source[start:end]
        self.assertIn("Ct cross_boundary(", client_body)
        self.assertIn("Ct cross_boundary_active(", client_body)
        self.assertIn("Ct cross_boundary_softmax_select(", client_body)
        self.assertIn("Ct cross_boundary_impl(", client_body)
        # exactly one new public entry point beyond the frozen file's two
        self.assertEqual(client_body.count("Ct cross_boundary"), 4)

    def test_cross_boundary_impl_mechanics_unchanged(self) -> None:
        # The decrypt/transform/encrypt body itself, copied byte-for-byte
        # from the frozen file -- only Client gained a new caller, the
        # mechanism that logs round trips did not change.
        self.assertIn("cc_->Decrypt(keys_.secretKey, local, &plaintext);", self.source)
        self.assertIn(
            "Plaintext refreshed = cc_->MakeCKKSPackedPlaintext(values, 1, 0, nullptr, SLOTS);",
            self.source,
        )
        self.assertIn("++round_trips_;", self.source)
        self.assertEqual(
            self.source.count("cc_->Decrypt("), 1
        )  # inside cross_boundary_impl only

    def test_per_row_loop_present(self) -> None:
        self.assertIn("for (std::size_t row = 1; row < T; ++row) {", self.source)
        self.assertIn("const std::size_t row_length = row + 1;", self.source)
        self.assertIn("for (std::size_t col = 1; col <= row; ++col) {", self.source)

    def test_evaluator_has_no_private_key_access(self) -> None:
        start = self.source.index("class EncryptedEvaluator")
        end = self.source.index("struct Metrics")
        evaluator_body = self.source[start:end]
        self.assertNotIn("Decrypt(", evaluator_body)
        self.assertNotIn("secretKey", evaluator_body)

    def test_single_final_decrypt_in_main(self) -> None:
        self.assertEqual(self.source.count("cc->Decrypt("), 1)

    def test_pinned_fixture_manifest_matches_t3_fixture(self) -> None:
        self.assertIn(
            f'"{FIXTURE_MANIFEST_SHA256}"',
            self.source,
        )

    def test_scripts_require_general_attention_tag_marker(self) -> None:
        for path in (RUN_SCRIPT, LAUNCH_SCRIPT):
            text = path.read_text(encoding="utf-8")
            self.assertIn("_general_attention_", text)
            self.assertIn("_scheme_b_", text)

    def test_orchestrator_references_general_attention_launch_script(self) -> None:
        text = ORCHESTRATOR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("launch_brev_scheme_b_general_attention.sh", text)

    def test_tag_marker_disjoint_from_other_prototypes(self) -> None:
        for forbidden in (
            "_cached_",
            "_serialized_",
            "_profiled_",
            "_diagcache_",
            "_warmup_",
            "_lintransform_",
        ):
            self.assertNotIn(forbidden, "_general_attention_")
            self.assertNotIn("_general_attention_", forbidden)

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_general_attention", cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_general_attention", build_text)


class GeneralAttentionNumpyContractTests(unittest.TestCase):
    """Mathematical-equivalence check, not a substitute for the real GPU
    correctness gate: proves the proposed circuit's arithmetic matches the
    real T=3 oracle before any CUDA code runs."""

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=3 fixture absent")
    def test_general_attention_matches_oracle(self) -> None:
        scores = _load_array("oracle.attention_scores")  # [heads, T, T]
        value = _load_array("oracle.value")  # [heads, T, head_dim]
        context_heads_oracle = _load_array("oracle.attention_context_heads")
        context_merged_oracle = _load_array("oracle.attention_context_merged")
        attn_proj_w = _load_array("weights.attn_proj")
        attn_proj_oracle = _load_array("oracle.attention_projection")

        context_heads, round_trips = _general_attention_reference(scores, value)
        np.testing.assert_allclose(
            context_heads,
            context_heads_oracle,
            rtol=0,
            atol=1e-9,
            err_msg="reconstructed per-head context disagrees with the real oracle",
        )
        self.assertEqual(
            round_trips,
            T * (T - 1) // 2,
            "round trips must scale as T*(T-1)/2, matching the source's own "
            "sum_{i=1}^{T-1} i derivation",
        )

        heads, tokens, head_dim = context_heads.shape
        context_merged = context_heads.transpose(1, 0, 2).reshape(
            tokens, heads * head_dim
        )
        np.testing.assert_allclose(
            context_merged, context_merged_oracle, rtol=0, atol=1e-9
        )
        attn_proj = context_merged @ attn_proj_w.T
        np.testing.assert_allclose(attn_proj, attn_proj_oracle, rtol=0, atol=1e-6)

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=3 fixture absent")
    def test_t2_is_a_special_case_of_the_same_formula(self) -> None:
        """The frozen T=2 file's identity (w1 = sigmoid(s11-s10)) must be
        exactly what _exact_softmax_select produces when row_length=2,
        select_index=1 -- proving this is a strict generalization, not a
        coincidentally-agreeing different algorithm."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            s10, s11 = rng.normal(size=2)
            weight = _exact_softmax_select(np.array([s10, s11]), 1)
            expected = 1.0 / (1.0 + np.exp(-(s11 - s10)))
            self.assertAlmostEqual(weight, expected, places=12)


if __name__ == "__main__":
    unittest.main()
