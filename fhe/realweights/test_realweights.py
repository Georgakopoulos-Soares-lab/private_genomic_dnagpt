"""Fast contract tests that do not require downloaded checkpoints or datasets."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .calibrate import (
    Range,
    _asymmetric_domain,
    _positive_domain,
    _symmetric_domain,
    write_immutable_result,
)
from .contract import (
    BLOCK0_EXPORT_KEYS,
    EXPECTED_MISSING_KEYS,
    MOTIF,
    MOTIF_START,
    expected_checkpoint_keys,
    gsr_prompt,
    make_tokenizer,
    strip_central_motif,
)
from .manual import block_reference, relative_inf

from dna_gpt.model.gpt import Block


class RealWeightContractTests(unittest.TestCase):
    def test_released_checkpoint_key_contract_is_exact(self) -> None:
        keys = expected_checkpoint_keys()
        self.assertEqual(len(keys), 78)
        self.assertEqual(len(set(keys)), 78)
        self.assertFalse(any(key.endswith(".bias") for key in keys))
        self.assertEqual(
            tuple(BLOCK0_EXPORT_KEYS.values()),
            (
                "transformer.h.0.ln_1.weight",
                "transformer.h.0.attn.c_attn.weight",
                "transformer.h.0.attn.c_proj.weight",
                "transformer.h.0.ln_2.weight",
                "transformer.h.0.mlp.c_fc.weight",
                "transformer.h.0.mlp.c_proj.weight",
            ),
        )
        self.assertEqual(
            EXPECTED_MISSING_KEYS,
            (
                "number_embedding.0.weight",
                "number_embedding.2.weight",
                "number_embedding.3.weight",
                "num_regression.0.weight",
                "num_regression.2.weight",
                "num_regression.3.weight",
            ),
        )

    def test_gsr_transform_and_first_two_token_semantics(self) -> None:
        raw = "A" * MOTIF_START + MOTIF + "C" * 300
        stripped = strip_central_motif(raw)
        self.assertEqual(stripped, "A" * 300 + "C" * 300)
        prompt = gsr_prompt(raw)
        tokenizer = make_tokenizer()
        pieces = tokenizer.tokenize(prompt)
        ids = tokenizer.encode(prompt, to_tensor=False)
        self.assertEqual(pieces[:2], ["<R>", "AAAAAA"])
        self.assertEqual(
            ids[:2], [tokenizer.piece_to_id(piece) for piece in pieces[:2]]
        )

        with self.assertRaisesRegex(ValueError, "central motif"):
            strip_central_motif("A" * 606)

    def test_numpy_manual_block_matches_upstream_torch(self) -> None:
        torch.manual_seed(7)
        block = Block(dim=8, num_heads=2, dropout=0.0, bias=False).double().eval()
        x = torch.randn(1, 3, 8, dtype=torch.float64)
        with torch.inference_mode():
            expected = block(x)[0].numpy()
        weights = {
            "ln1": block.ln_1.weight.detach().numpy(),
            "attn_qkv": block.attn.c_attn.weight.detach().numpy(),
            "attn_proj": block.attn.c_proj.weight.detach().numpy(),
            "ln2": block.ln_2.weight.detach().numpy(),
            "mlp_fc": block.mlp.c_fc.weight.detach().numpy(),
            "mlp_proj": block.mlp.c_proj.weight.detach().numpy(),
        }
        actual = block_reference(x[0].numpy(), weights, num_heads=2)["block_output"]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertLess(relative_inf(expected, actual), 1e-12)

    def test_recommended_domains_expand_ranges_outward(self) -> None:
        value = Range(minimum=-1.01, maximum=2.01, count=2)
        self.assertEqual(_symmetric_domain(value, 1.25), [-2.6, 2.6])
        self.assertEqual(_asymmetric_domain(value, 1.25), [-1.3, 2.6])
        positive = Range(minimum=0.01, maximum=1.01, count=2)
        low, high = _positive_domain(positive, 1.25)
        self.assertLessEqual(low, positive.minimum)
        self.assertGreaterEqual(high, positive.maximum)

    def test_calibration_result_is_immutable(self) -> None:
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            first = write_immutable_result({"passed": True}, output_dir, "new_tag")
            self.assertTrue(first.is_file())
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                write_immutable_result({"passed": False}, output_dir, "new_tag")


if __name__ == "__main__":
    unittest.main()
