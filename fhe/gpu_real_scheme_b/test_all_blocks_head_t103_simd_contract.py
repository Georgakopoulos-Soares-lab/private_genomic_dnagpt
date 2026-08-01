"""Pre-GPU contracts for Scheme B's complete 12-block + GSR-head graph at
T=103 with Token-SIMD B=8 packing.

This file is intentionally CPU/NumPy-only. It is the T=103/Token-SIMD sibling
of ``test_all_blocks_head_t2_contract.py``: it proves the released fixture's
eleven inter-block boundaries plus one block-11-to-head boundary, independently
recomputes every block and the encrypted N-minus-A readout, proves the
Token-SIMD pack/unpack refresh cycle is lossless, and freezes the expected
Scheme B boundary arithmetic before any C++/GPU implementation is written.

Design note on why this differs structurally from the T=2 driver: at T=2 a
released block's entire hidden state fits one ciphertext, so the client
refresh between blocks is "decrypt one ciphertext, re-encrypt T=2 fresh
per-token ciphertexts". At T=103 with Token-SIMD B=8 packing, a block's
hidden state is already 13 ciphertexts (one per token group of <=8 tokens),
so the refresh is "decrypt all 13 group ciphertexts, re-encrypt 13 fresh
group ciphertexts" -- no per-token ciphertext ever exists mid-graph. This
file models that packed refresh with ``simd_layout.pack_replicated_tokens``/
``unpack_tokens``, the same primitives already proven against the real T=103
oracle by ``test_simd_layout.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Make both this file's own directory (for the bare ``simd_layout`` import,
# matching test_simd_layout.py's convention) and the repo root (for the
# ``fhe.*`` namespace-package imports, matching
# test_all_blocks_head_t2_contract.py's convention) importable regardless of
# how this file is invoked (direct script, ``python -m unittest <dotted>``,
# or a test runner with a different rootdir-insertion policy).
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _path in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fhe.multiblock.manual import classifier_head_reference, layer_norm, silu  # noqa: E402
from fhe.realweights.manual import block_reference  # noqa: E402

from simd_layout import (  # noqa: E402
    COPIES,
    PACK_WIDTH,
    pack_replicated_tokens,
    protocol_counts,
    unpack_tokens,
)

ROOT = _THIS_DIR
REPO = _REPO_ROOT

# ---------------------------------------------------------------------------
# Pinned identities
# ---------------------------------------------------------------------------

EXPORTER_SOURCE = REPO / "fhe" / "multiblock" / "export_fixture_t103.py"
EXPORTER_SOURCE_SHA256 = (
    "ab1f438d18e4dc50da4aa051703ded1bd1ec240686a928b92931b3f799e81fd3"
)
FIXTURE_CONTRACT = ROOT / "fixture_all_blocks_head_t103.sha256"
FIXTURE_CONTRACT_SHA256 = (
    "5256d3f59e215631f9ded7dba68f13e16c12c8fde927c53277bc1752171f7cca"
)
FIXTURE = (
    REPO
    / "checkpoints"
    / "fhe_exports"
    / "gsr_pos0_multiblock_t103_d63353abdc1a_52d046d1fcf0"
)
MANIFEST = FIXTURE / "manifest.json"
MANIFEST_SHA256 = "ab55533eaf8779b64a67c5cdd8ac67152419389e33f1c44addd4a092e0ff962d"

# Depth-13 per-block Token-SIMD source this driver composes 12x, and the
# T=2 two-block-refresh source this driver's inter-block refresh design is
# adapted from. Both are frozen/read-only anchors for this file; neither is
# edited here.
DEPTH13_SOURCE = (
    ROOT
    / "src"
    / "real_dnagpt_fides_scheme_b_simd_full_t103_depth13_digits3_ring65536.cpp"
)
DEPTH13_SOURCE_SHA256 = (
    "6e8cd08efad1567d2f9001f69d10e302e45f4e634f3bd37e2245382d215fbe5e"
)
TWO_BLOCK_REFRESH_SOURCE = (
    ROOT / "src" / "real_dnagpt_fides_scheme_b_two_block_refresh.cpp"
)
TWO_BLOCK_REFRESH_SOURCE_SHA256 = (
    "5cb82f81dc46efe2f4bfd7a1c9e9c1bf90cea3d629432b027b42e5df89093767"
)

D_MODEL = 768
T = 103
BLOCKS = 12
TOKEN_BATCH = 8
TOKEN_GROUPS = math.ceil(T / TOKEN_BATCH)
LAST_TOKEN = T - 1
LAST_TOKEN_GROUP = LAST_TOKEN // TOKEN_BATCH
LAST_TOKEN_LANE = LAST_TOKEN % TOKEN_BATCH

# Pinned from real-GPU evidence for the single Token-SIMD B=8 T=103 block
# this driver composes 12 times (identical schedule at depth 13 and depth
# 16 -- only MULT_DEPTH differs between those forks, not the protocol):
#   results/runs/fhe_fides_real_d768_t103_block0_simd_full_t103_depth13_digits3_ring65536_scheme_b_a100_20260731.json
#   protocol.round_trips == 857, protocol.logical_boundary_instances == 129162
PER_BLOCK_ROUND_TRIPS = 857
PER_BLOCK_LOGICAL_INSTANCES = 129162

INTER_BLOCK_REFRESHES = BLOCKS - 1  # 11
HEAD_INPUT_REFRESHES = 1
HEAD_ROUND_TRIPS = 3  # final_ln invsqrt, head_silu, head_ln invsqrt

EXPECTED_TOTAL_CROSSINGS = (
    BLOCKS * PER_BLOCK_ROUND_TRIPS
    + INTER_BLOCK_REFRESHES
    + HEAD_INPUT_REFRESHES
    + HEAD_ROUND_TRIPS
)
EXPECTED_TOTAL_LOGICAL_INSTANCES = (
    BLOCKS * PER_BLOCK_LOGICAL_INSTANCES
    + INTER_BLOCK_REFRESHES * T
    + HEAD_INPUT_REFRESHES * 1
    + HEAD_ROUND_TRIPS * 1
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def active_tokens(group: int, tokens: int = T, batch_width: int = TOKEN_BATCH) -> int:
    first = group * batch_width
    return min(batch_width, tokens - first)


class PinnedArithmeticTests(unittest.TestCase):
    """These do not touch the fixture; they freeze the boundary formulas."""

    def test_token_groups(self) -> None:
        self.assertEqual(TOKEN_GROUPS, 13)
        self.assertEqual(active_tokens(TOKEN_GROUPS - 1), 7)
        for group in range(TOKEN_GROUPS - 1):
            self.assertEqual(active_tokens(group), TOKEN_BATCH)

    def test_last_token_position(self) -> None:
        # The last (103rd) token, whose hidden state feeds the classifier
        # head, lives in the final token group at lane 6 (0-indexed), not
        # lane 7 -- the final group only has 7 active lanes (0..6).
        self.assertEqual(LAST_TOKEN_GROUP, 12)
        self.assertEqual(LAST_TOKEN_LANE, 6)
        self.assertEqual(active_tokens(LAST_TOKEN_GROUP), 7)
        self.assertLess(LAST_TOKEN_LANE, active_tokens(LAST_TOKEN_GROUP))

    def test_protocol_counts_reproduce_pinned_single_block_evidence(self) -> None:
        counts = protocol_counts(T, TOKEN_BATCH)
        self.assertEqual(counts.token_groups, TOKEN_GROUPS)
        self.assertEqual(counts.score_tile_decryptions, 91)
        self.assertEqual(counts.weight_tile_encryptions, 727)
        self.assertEqual(counts.layernorm1_crossings, TOKEN_GROUPS)
        self.assertEqual(counts.layernorm2_crossings, TOKEN_GROUPS)
        self.assertEqual(counts.gelu_crossings, TOKEN_GROUPS)
        self.assertEqual(counts.attention_gate_crossings, 91 + 727 + 13)
        self.assertEqual(counts.full_block_crossings, PER_BLOCK_ROUND_TRIPS)

    def test_total_declared_crossing_arithmetic(self) -> None:
        self.assertEqual(EXPECTED_TOTAL_CROSSINGS, 10299)
        self.assertEqual(EXPECTED_TOTAL_LOGICAL_INSTANCES, 1551081)
        # Of the 3 head round trips, 2 flow through the shared depth-13
        # Client's own round_trips()/logical_instances() counters (calling
        # its public invsqrt_boundary() for final_ln and head_ln); only the
        # SiLU step has no equivalent public Client method (depth13's Client
        # only exposes invsqrt_boundary/gelu_boundary/attention-tile
        # methods) and is tracked by this driver's own extra counter.
        client_tracked_head_round_trips = 2
        driver_tracked_head_round_trips = 1
        self.assertEqual(
            client_tracked_head_round_trips + driver_tracked_head_round_trips,
            HEAD_ROUND_TRIPS,
        )


@unittest.skipUnless(FIXTURE.exists(), "multiblock T=103 fixture absent")
class FixtureIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_pinned_source_and_contract_hashes(self) -> None:
        self.assertEqual(sha256(EXPORTER_SOURCE), EXPORTER_SOURCE_SHA256)
        self.assertEqual(sha256(FIXTURE_CONTRACT), FIXTURE_CONTRACT_SHA256)
        self.assertEqual(sha256(DEPTH13_SOURCE), DEPTH13_SOURCE_SHA256)
        self.assertEqual(
            sha256(TWO_BLOCK_REFRESH_SOURCE), TWO_BLOCK_REFRESH_SOURCE_SHA256
        )
        self.assertEqual(sha256(MANIFEST), MANIFEST_SHA256)

    def test_manifest_binds_complete_full_prompt_graph(self) -> None:
        self.assertEqual(self.manifest["schema"], "dnagpt.multiblock.gsr_general.v1")
        self.assertEqual(self.manifest["model"]["layers"], BLOCKS)
        self.assertEqual(self.manifest["model"]["embedding_dim"], D_MODEL)
        self.assertEqual(self.manifest["tokenization"]["exported_token_count_T"], T)
        self.assertEqual(self.manifest["tokenization"]["full_prompt_token_count"], T)
        self.assertTrue(self.manifest["tokenization"]["is_full_prompt"])
        self.assertTrue(self.manifest["oracle_gate"]["passed"])
        self.assertEqual(
            self.manifest["classification"]["encrypted_readout"],
            "last token -> final LN -> head linear -> SiLU -> head LN "
            "-> dot(head_readout_N - head_readout_A); sign selects N/A",
        )

    def test_every_consumed_file_matches_the_immutable_hash_contract(self) -> None:
        entries: dict[str, str] = {}
        for line in FIXTURE_CONTRACT.read_text(encoding="utf-8").splitlines():
            digest, filename = line.split(maxsplit=1)
            entries[filename] = digest
        self.assertEqual(len(entries), 392)
        for filename, digest in entries.items():
            self.assertEqual(sha256(FIXTURE / filename), digest, filename)


@unittest.skipUnless(FIXTURE.exists(), "multiblock T=103 fixture absent")
class AllBlocksHeadT103ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def load(self, name: str) -> np.ndarray:
        item = self.manifest["arrays"][name]
        path = FIXTURE / item["file"]
        self.assertEqual(sha256(path), item["sha256"], name)
        return np.fromfile(path, dtype="<f8").reshape(item["shape"])

    def block_weights(self, block: int) -> dict[str, np.ndarray]:
        return {
            field: self.load(f"weights.block{block}.{field}")
            for field in (
                "ln1",
                "attn_qkv",
                "attn_proj",
                "ln2",
                "mlp_fc",
                "mlp_proj",
            )
        }

    def test_every_inter_block_boundary_is_byte_exact(self) -> None:
        for block in range(BLOCKS - 1):
            output = self.load(f"oracle.block{block}.block_output")
            next_input = self.load(f"oracle.block{block + 1}.input_embeddings")
            np.testing.assert_array_equal(output, next_input)

    def test_all_twelve_blocks_recompute_through_the_refresh_lineage(self) -> None:
        hidden = self.load("input.embeddings")
        self.assertEqual(hidden.shape, (T, D_MODEL))
        for block in range(BLOCKS):
            expected_input = self.load(f"oracle.block{block}.input_embeddings")
            # atol=1e-9, not 1e-12: this compares two independently-computed
            # float64 forward passes (this test's block_reference() chain vs
            # the pinned oracle's), and the two accumulate rounding
            # differently over 12 chained blocks -- 1e-12 is tight enough to
            # occasionally fail on pure floating-point noise at these output
            # magnitudes (~1.7e-12 observed at block 5+), matching every
            # other real-fixture multi-step comparison in this codebase
            # (e.g. test_simd_layout.py's atol=1e-9), not a correctness bug.
            np.testing.assert_allclose(hidden, expected_input, rtol=0, atol=1e-9)
            actual = block_reference(
                hidden,
                self.block_weights(block),
                num_heads=12,
                eps=1e-5,
            )["block_output"]
            expected = self.load(f"oracle.block{block}.block_output")
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
            hidden = actual.copy()
        self._final_hidden = hidden

    def test_token_simd_pack_unpack_refresh_is_lossless_every_block(self) -> None:
        """Models the declared Scheme B inter-block refresh: decrypt all 13
        group ciphertexts, recover the T x D hidden state, re-encrypt 13
        fresh group ciphertexts. With no arithmetic involved, packing and
        unpacking a real block output must round-trip bit-exactly."""
        for block in range(BLOCKS):
            hidden = self.load(f"oracle.block{block}.block_output")
            self.assertEqual(hidden.shape, (T, D_MODEL))
            reconstructed = np.zeros_like(hidden)
            for group in range(TOKEN_GROUPS):
                first = group * TOKEN_BATCH
                active = active_tokens(group)
                tokens = hidden[first : first + active, :]
                packed = pack_replicated_tokens(tokens, TOKEN_BATCH, dimension=D_MODEL)
                self.assertEqual(packed.shape, (COPIES * PACK_WIDTH * TOKEN_BATCH,))
                unpacked = unpack_tokens(packed, TOKEN_BATCH, active, dimension=D_MODEL)
                reconstructed[first : first + active, :] = unpacked
            np.testing.assert_array_equal(reconstructed, hidden)

    def test_last_token_extraction_matches_full_hidden_state(self) -> None:
        block11_output = self.load("oracle.block11.block_output")
        last_token_vector = block11_output[LAST_TOKEN, :]
        first = LAST_TOKEN_GROUP * TOKEN_BATCH
        active = active_tokens(LAST_TOKEN_GROUP)
        group_tokens = block11_output[first : first + active, :]
        packed = pack_replicated_tokens(group_tokens, TOKEN_BATCH, dimension=D_MODEL)
        unpacked = unpack_tokens(packed, TOKEN_BATCH, active, dimension=D_MODEL)
        np.testing.assert_array_equal(unpacked[LAST_TOKEN_LANE], last_token_vector)

    def test_classifier_head_and_encrypted_margin_recompute(self) -> None:
        hidden = self.load("oracle.block11.block_output")
        weights = {
            "final_ln": self.load("weights.final_ln"),
            "head_linear": self.load("weights.head_linear"),
            "head_ln": self.load("weights.head_ln"),
            "head_readout": np.stack(
                [
                    self.load("weights.head_readout_n"),
                    self.load("weights.head_readout_a"),
                ]
            ),
        }
        head = classifier_head_reference(hidden, weights)

        final_ln, *_ = layer_norm(hidden, weights["final_ln"])
        np.testing.assert_allclose(
            final_ln, self.load("oracle.head.final_ln_output"), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            head["final_ln_output"],
            self.load("oracle.head.final_ln_output"),
            rtol=0,
            atol=1e-12,
        )

        head_linear = final_ln @ weights["head_linear"].T
        np.testing.assert_allclose(
            head_linear, self.load("oracle.head.head_linear"), rtol=0, atol=1e-12
        )

        head_silu = silu(head_linear)
        np.testing.assert_allclose(
            head_silu, self.load("oracle.head.head_silu"), rtol=0, atol=1e-12
        )

        head_ln, *_ = layer_norm(head_silu, weights["head_ln"])
        oracle_head_ln = self.load("oracle.head.head_ln_output")
        np.testing.assert_allclose(head_ln, oracle_head_ln, rtol=0, atol=1e-12)

        margin_row = self.load("weights.head_readout_margin_n_minus_a")
        margin = float(head_ln[LAST_TOKEN] @ margin_row)
        direct_dot_oracle = float(oracle_head_ln[LAST_TOKEN] @ margin_row)
        contract = self.manifest["classification"]["graph_gate"]
        self.assertAlmostEqual(margin, direct_dot_oracle, places=9)
        self.assertLess(abs(margin - contract["margin_n_minus_a"]), 1e-6)
        self.assertEqual(np.signbit(margin), np.signbit(contract["margin_n_minus_a"]))
        self.assertEqual(contract["binary_label"], "N")
        self.assertGreater(margin, 0.0)
        self.assertTrue(self.manifest["tokenization"]["is_full_prompt"])
        self.assertIn("real GSR task semantics", contract["evidence"])

    def test_head_dot_reads_the_correct_lane_after_pack_unpack(self) -> None:
        """The head operates on a single-token Token-SIMD group (the last
        token in lane 0 of a fresh single-active-token group). Prove that
        packing just the last token alone and unpacking it recovers the
        exact vector the plaintext head consumes."""
        hidden = self.load("oracle.block11.block_output")
        last_token_vector = hidden[LAST_TOKEN, :]
        packed = pack_replicated_tokens(
            last_token_vector.reshape(1, D_MODEL), TOKEN_BATCH, dimension=D_MODEL
        )
        unpacked = unpack_tokens(packed, TOKEN_BATCH, 1, dimension=D_MODEL)
        np.testing.assert_array_equal(unpacked[0], last_token_vector)


if __name__ == "__main__":
    unittest.main()
