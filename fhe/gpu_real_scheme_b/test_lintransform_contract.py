"""Static and float64 contracts for the Scheme B LinearTransform prototype
(real_dnagpt_fides_scheme_b_lintransform.cpp).

No FIDESlib/CUDA build is required to run this: it is a source-text plus
numpy contract, the same style as test_profiling_contract.py/
test_warmup_contract.py/test_batching_contract.py, proving (before any GPU
run) that:

  - the frozen single-shot/cached sources, and the profiled baseline this
    prototype is forked from, are all untouched;
  - exactly ONE of the six textual matmul() call sites (the QKV query
    projection) is converted to the new matmul_lintransform() path; the
    other five stay on the original manual matmul()/bsgs_inner_loop path;
  - matmul_lintransform() reuses the identical diagonal-value formula
    (rolled_row/column) as bsgs_inner_loop -- copy-pasted, not re-derived;
  - raw_plain() gained an optional `level` parameter, defaulting to 0, so
    every pre-existing call site (which passes no level argument) is
    unaffected;
  - matmul_lintransform() builds an independent native ciphertext clone
    before calling LinearTransform (which mutates in place), never the
    shared `baby.values[0]`/`normalized[token]` object the sibling
    key/value matmul calls still depend on;
  - an explicit level-match check is present (FIDESlib's own assert is
    compiled out under -DCMAKE_BUILD_TYPE=Release/NDEBUG -- see
    docs/hybrid/tasks.md's 2026-07-28 scoping Finding 5);
  - FIDESlib::CKKS::LinearTransform is called exactly once, with
    rowSize=BSGS_N1*BSGS_N2, bStep=BSGS_N1, stride=1, offset=0;
  - TOL, BSGS_N1/BSGS_N2, and the packing constants are unchanged;
  - EncryptedEvaluator still contains neither `Decrypt(` nor `secretKey`;
  - the run/launch/orchestrator scripts and CMake reference the new binary
    and require the `_lintransform_` tag marker, disjoint from every other
    prototype's marker.

The float64 test (SchemeBLinearTransformNumpyContractTests) is a
mathematical-equivalence check, not a substitute for the real GPU
correctness gate: it transliterates LinearTransform's documented outer
control-flow (src/CKKS/LinearTransform.cu's FUSED branch: gStep batched
giant-step partial sums, then a reverse-order rotate-and-add accumulation)
using this session's derivation of DotProductPtInternal's indexing from its
C++ source -- the low-level LTdotProductPtBatch CUDA kernel body was not
read. It checks three things agree: (1) a literal step-by-step replica of
the reverse-order loop exactly as written in LinearTransform.cu, (2) this
session's closed-form derivation that the whole primitive reduces to
`sum_m rotate(S_m, m*bStep*stride)`, and (3) the real oracle fixture's QKV
query output, computed from the real weight matrix and real LN1 output. If
(1) and (2) disagree, the derivation in docs/hybrid/tasks.md is wrong. If
either disagrees with (3), the diagonal construction itself is wrong.
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).with_name("src")
SOURCE = HERE / "real_dnagpt_fides_scheme_b_lintransform.cpp"
ORIGINAL_SOURCE = HERE / "real_dnagpt_fides_scheme_b.cpp"
CACHED_SOURCE = HERE / "real_dnagpt_fides_scheme_b_cached.cpp"
PROFILED_SOURCE = HERE / "real_dnagpt_fides_scheme_b_profiled.cpp"
ORIGINAL_SHA256 = "d88f1a0919a003a539e746aaf33e5d283c28810c2805a1af4b05dffac43e08df"
CACHED_SHA256 = "de79d6e7145b3ac9035b09cf7bc86354c755faab139c46f942bcc03f6ab72cc6"
PROFILED_SHA256 = "4ca9c66470c412140ed483479ae869963e8b097e2880b3ffd4d9287a81374d61"

SCRIPT_DIR = Path(__file__).parent
RUN_SCRIPT = SCRIPT_DIR / "run_scheme_b_lintransform.sh"
LAUNCH_SCRIPT = SCRIPT_DIR / "launch_brev_scheme_b_lintransform.sh"
ORCHESTRATOR_SCRIPT = SCRIPT_DIR / "wait_and_run_scheme_b_lintransform.sh"
CMAKE = SCRIPT_DIR / "CMakeLists.txt"
BUILD_SCRIPT = SCRIPT_DIR / "build_in_fideslib.sh"

D = 768
T = 2
PACK_WIDTH = 1024
COPIES = 4
SLOTS = COPIES * PACK_WIDTH
BSGS_N1 = 32
BSGS_N2 = PACK_WIDTH // BSGS_N1

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_block0_t2_d63353abdc1a_52d046d1fcf0"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_array(name: str) -> np.ndarray:
    import json

    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["arrays"][name]
    path = FIXTURE / metadata["file"]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError(f"fixture hash mismatch for {name}")
    return np.fromfile(path, dtype="<f8").reshape(metadata["shape"])


def _rotate(values: np.ndarray, index: int) -> np.ndarray:
    """OpenFHE/FIDES logical left rotation (matches test_batching_contract.py)."""
    return np.roll(np.asarray(values), -index)


def _diagonal(weight: np.ndarray, giant: int, small: int) -> np.ndarray:
    # Literal transliteration of bsgs_inner_loop's/matmul_lintransform's
    # packed_values loop (rolled_row/column), vectorized over `row` instead
    # of a Python for-loop, for one PACK_WIDTH-wide copy block.
    row = np.arange(PACK_WIDTH)
    rolled_row = (row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH
    diagonal_index = BSGS_N1 * giant + small
    column = (rolled_row + diagonal_index) % PACK_WIDTH
    out = np.zeros(PACK_WIDTH)
    valid = (rolled_row < D) & (column < D)
    out[valid] = weight[rolled_row[valid], column[valid]]
    return out


def _baby_rotations(x_padded: np.ndarray) -> list[np.ndarray]:
    return [_rotate(x_padded, small) for small in range(BSGS_N1)]


def _giant_step_partial_sums(
    weight: np.ndarray, baby: list[np.ndarray]
) -> list[np.ndarray]:
    # S_m = sum_i fastRotation[i] * Aptr[bStep*m + i], per this session's
    # reading of DotProductPtInternal's indexing (LinearTransform.cu).
    sums = []
    for giant in range(BSGS_N2):
        total = np.zeros(PACK_WIDTH)
        for small in range(BSGS_N1):
            total = total + baby[small] * _diagonal(weight, giant, small)
        sums.append(total)
    return sums


def _linear_transform_literal_replica(partial_sums: list[np.ndarray]) -> np.ndarray:
    # Transliterates LinearTransform.cu's FUSED-branch reverse loop exactly:
    #   for j = gStep-1 down to 0:
    #     if j != gStep-1: results[j] += results[j+1]
    #     if j > 0: rotate results[j] by bStep*stride
    #     elif offset != 0: rotate by offset   (offset=0 here, so: nothing)
    # stride=1 here, so the giant rotation amount is bStep*1 = BSGS_N1.
    results = [s.copy() for s in partial_sums]
    for j in range(BSGS_N2 - 1, -1, -1):
        if j != BSGS_N2 - 1:
            results[j] = results[j] + results[j + 1]
        if j > 0:
            results[j] = _rotate(results[j], BSGS_N1)
    return results[0]


def _closed_form(partial_sums: list[np.ndarray]) -> np.ndarray:
    # This session's derivation (docs/hybrid/tasks.md): the whole reverse
    # loop reduces to sum_m rotate(S_m, m*bStep*stride).
    total = np.zeros(PACK_WIDTH)
    for giant, s in enumerate(partial_sums):
        total = total + _rotate(s, BSGS_N1 * giant)
    return total


def _matmul_lintransform_reference(weight: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_padded = np.zeros(PACK_WIDTH)
    x_padded[:D] = x
    baby = _baby_rotations(x_padded)
    partial_sums = _giant_step_partial_sums(weight, baby)
    literal = _linear_transform_literal_replica(partial_sums)
    closed = _closed_form(partial_sums)
    np.testing.assert_allclose(
        literal,
        closed,
        rtol=0.0,
        atol=1e-9,
        err_msg=(
            "LinearTransform.cu's literal reverse-loop replica disagrees with "
            "this session's closed-form derivation -- the docs/hybrid/tasks.md "
            "derivation is wrong, independent of whether the diagonal values "
            "themselves are correct"
        ),
    )
    return literal[:D]


class SchemeBLinearTransformContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_frozen_single_shot_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(ORIGINAL_SOURCE), ORIGINAL_SHA256)

    def test_frozen_cached_source_is_untouched(self) -> None:
        self.assertEqual(_sha256(CACHED_SOURCE), CACHED_SHA256)

    def test_frozen_profiled_baseline_source_is_untouched(self) -> None:
        self.assertTrue(PROFILED_SOURCE.exists())
        self.assertEqual(_sha256(PROFILED_SOURCE), PROFILED_SHA256)

    def test_correctness_contract_unchanged(self) -> None:
        self.assertIn("constexpr double TOL = 4e-2;", self.source)
        self.assertIn("constexpr std::size_t BSGS_N1 = 32;", self.source)
        self.assertIn(
            "constexpr std::size_t BSGS_N2 = PACK_WIDTH / BSGS_N1;", self.source
        )
        self.assertIn("constexpr std::size_t D = 768;", self.source)
        self.assertIn("constexpr std::size_t T = 2;", self.source)
        self.assertIn("constexpr std::size_t PACK_WIDTH = 1024;", self.source)
        self.assertIn("constexpr std::size_t COPIES = 4;", self.source)
        self.assertIn("constexpr std::uint32_t MULT_DEPTH = 16;", self.source)
        self.assertRegex(self.source, r'GATE\s*=\s*"full"')

    def test_exactly_one_call_site_converted(self) -> None:
        # Five call sites stay on the original manual matmul(); exactly one
        # (QKV query) is converted to matmul_lintransform().
        self.assertEqual(self.source.count("return matmul("), 5)
        self.assertEqual(self.source.count("return matmul_lintransform("), 1)
        self.assertIn(
            "return matmul_lintransform(baby.values[0], fixture_.qkv[0]);",
            self.source,
        )

    def test_native_linear_transform_called_once_with_expected_shape(self) -> None:
        self.assertEqual(self.source.count("FIDESlib::CKKS::LinearTransform("), 1)
        self.assertIn(
            "*ctxt_gpu, static_cast<int>(BSGS_N1 * BSGS_N2),\n"
            "            static_cast<int>(BSGS_N1), pts_gpu, /*stride=*/1, "
            "/*offset=*/0);",
            self.source,
        )

    def test_native_header_included(self) -> None:
        self.assertIn("#include <CKKS/LinearTransform.cuh>", self.source)
        # LinearTransform.cuh alone only forward-declares native Ciphertext/
        # Plaintext -- calling ->c0/->getLevel() on them needs full
        # definitions too (a real GPU build caught this omission once).
        self.assertIn("#include <CKKS/Ciphertext.cuh>", self.source)
        self.assertIn("#include <CKKS/Plaintext.cuh>", self.source)

    def test_raw_plain_gained_optional_level_parameter_default_zero(self) -> None:
        self.assertIn(
            "Plaintext raw_plain(const std::vector<double>& values,\n"
            "                        std::uint32_t level = 0) {",
            self.source,
        )
        self.assertIn(
            "return cc_->MakeCKKSPackedPlaintext(values, 1, level, nullptr, SLOTS);",
            self.source,
        )

    def test_diagonal_formula_copy_pasted_from_bsgs_inner_loop(self) -> None:
        # The packed_values loop body (rolled_row/column) inside
        # matmul_lintransform must be textually identical to
        # bsgs_inner_loop's -- this is a reused, already-validated formula,
        # not a re-derivation.
        self.assertIn("(row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH", self.source)
        self.assertEqual(
            self.source.count("(row + PACK_WIDTH - BSGS_N1 * giant) % PACK_WIDTH"),
            2,
            "expected exactly two copies: bsgs_inner_loop and matmul_lintransform",
        )

    def test_level_match_check_present_and_explicit(self) -> None:
        # FIDESlib's own assert is compiled out under Release/NDEBUG (scoping
        # Finding 5) -- this prototype must not rely on it.
        self.assertIn("pt_gpu->c0.getLevel() != ctxt_gpu->getLevel()", self.source)
        self.assertIn("throw std::runtime_error(", self.source)
        self.assertIn("matmul_lintransform: diagonal plaintext level", self.source)

    def test_input_level_threaded_from_ciphertext_not_hardcoded(self) -> None:
        self.assertIn(
            "const std::uint32_t input_level =\n"
            "            static_cast<std::uint32_t>(input->GetLevel());",
            self.source,
        )
        self.assertIn(
            "pts_cpu.push_back(raw_plain(packed_values, input_level));", self.source
        )

    def test_independent_native_clone_not_in_place_on_shared_input(self) -> None:
        # LinearTransform mutates its ciphertext argument in place;
        # matmul_lintransform must never call it on `input` directly (which
        # is shared with the sibling key/value matmul calls and the manual
        # BSGS path via baby.values[0]/normalized[token]).
        self.assertIn(
            "Ct working = std::make_shared<CiphertextImpl<DCRTPoly>>(*input);",
            self.source,
        )
        start = self.source.index("Ct matmul_lintransform(")
        end = self.source.index("\n    }\n", start)
        body = self.source[start:end]
        self.assertNotIn("LinearTransform(*", body.replace("ctxt_gpu", ""))
        self.assertIn("*ctxt_gpu", body)

    def test_evaluator_has_no_private_key_access(self) -> None:
        start = self.source.index("class EncryptedEvaluator")
        end = self.source.index("struct Metrics")
        evaluator_body = self.source[start:end]
        self.assertNotIn("Decrypt(", evaluator_body)
        self.assertNotIn("secretKey", evaluator_body)

    def test_single_final_decrypt_in_main(self) -> None:
        self.assertEqual(self.source.count("cc->Decrypt("), 1)

    def test_scripts_require_lintransform_tag_marker(self) -> None:
        for path in (RUN_SCRIPT, LAUNCH_SCRIPT):
            text = path.read_text(encoding="utf-8")
            self.assertIn("_lintransform_", text)
            self.assertIn("_scheme_b_", text)

    def test_orchestrator_references_lintransform_launch_script(self) -> None:
        text = ORCHESTRATOR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("launch_brev_scheme_b_lintransform.sh", text)
        self.assertIn("SCHEME_B_LINTRANSFORM_RUN_TAG", text)

    def test_tag_marker_disjoint_from_other_prototypes(self) -> None:
        for forbidden in (
            "_cached_",
            "_serialized_",
            "_profiled_",
            "_diagcache_",
            "_warmup_",
        ):
            self.assertNotIn(forbidden, "_lintransform_")
            self.assertNotIn("_lintransform_", forbidden)

    def test_cmake_and_build_script_reference_new_binary(self) -> None:
        cmake_text = CMAKE.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_lintransform", cmake_text)
        build_text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("real_dnagpt_fides_scheme_b_lintransform", build_text)


class SchemeBLinearTransformNumpyContractTests(unittest.TestCase):
    """See module docstring: mathematical-equivalence check, not a GPU gate."""

    @unittest.skipUnless(FIXTURE.exists(), "real D=768/T=2 fixture absent")
    def test_linear_transform_reference_matches_oracle_query(self) -> None:
        attn_qkv = _load_array("weights.attn_qkv")  # [2304, 768]
        query_weight = attn_qkv[:D, :]
        ln1_output = _load_array("oracle.ln1_output")  # [T, D]
        oracle_query = _load_array("oracle.query")  # [heads, T, head_dim]
        heads, tokens, head_dim = oracle_query.shape
        expected = oracle_query.transpose(1, 0, 2).reshape(tokens, heads * head_dim)

        for token in range(T):
            got = _matmul_lintransform_reference(query_weight, ln1_output[token])
            direct = query_weight @ ln1_output[token]
            np.testing.assert_allclose(
                direct,
                expected[token],
                rtol=1e-9,
                atol=1e-9,
                err_msg="fixture's own weight@input does not match its oracle.query "
                "-- fixture inconsistency, not a LinearTransform bug",
            )
            np.testing.assert_allclose(
                got,
                expected[token],
                rtol=0.0,
                atol=1e-9,
                err_msg=(
                    "the LinearTransform-equivalent BSGS reconstruction "
                    f"disagrees with the real oracle for token {token}"
                ),
            )


if __name__ == "__main__":
    unittest.main()
