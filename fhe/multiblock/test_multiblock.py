"""Fast unit tests for the additive 12-block export contract."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from .export_fixture import block_weight_keys
from .manual import (
    BLOCK0_GPU_DOMAINS,
    block_calibration,
    classifier_head_reference,
    covers,
    outward_domain,
    t2_attention_schedule,
)


class MultiblockContractTests(unittest.TestCase):
    def test_all_12_blocks_have_exactly_six_bias_free_weights(self) -> None:
        keys = block_weight_keys()
        self.assertEqual(len(keys), 72)
        self.assertEqual(len(set(keys.values())), 72)
        self.assertFalse(any(key.endswith(".bias") for key in keys.values()))
        self.assertEqual(
            keys["block11.mlp_proj"],
            "transformer.h.11.mlp.c_proj.weight",
        )

    def test_t2_source0_shift_matches_softmax(self) -> None:
        scores = np.array(
            [
                [[0.2, 99.0], [-1.1, 0.7]],
                [[-0.3, 88.0], [2.0, -0.4]],
            ],
            dtype=np.float64,
        )
        delta, denominator = t2_attention_schedule(scores)
        scheduled_p1 = np.exp(delta) / denominator
        logits = scores[:, 1, :]
        stable = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        expected_p1 = (stable / stable.sum(axis=1, keepdims=True))[:, 1]
        np.testing.assert_allclose(scheduled_p1, expected_p1, rtol=1e-14, atol=1e-14)

    def test_classifier_head_matches_bias_free_torch(self) -> None:
        torch.manual_seed(23)
        hidden = torch.randn(2, 8, dtype=torch.float64)
        final_ln = torch.nn.LayerNorm(8, bias=False).double()
        linear = torch.nn.Linear(8, 8, bias=False).double()
        head_ln = torch.nn.LayerNorm(8, bias=False).double()
        readout = torch.nn.Linear(8, 11, bias=False).double()
        with torch.inference_mode():
            expected = readout(
                head_ln(torch.nn.functional.silu(linear(final_ln(hidden))))
            )
        actual = classifier_head_reference(
            hidden.numpy(),
            {
                "final_ln": final_ln.weight.detach().numpy(),
                "head_linear": linear.weight.detach().numpy(),
                "head_ln": head_ln.weight.detach().numpy(),
                "head_readout": readout.weight.detach().numpy(),
            },
        )
        np.testing.assert_allclose(
            actual["logits"], expected.numpy(), rtol=1e-12, atol=1e-12
        )

    def test_domain_coverage_fails_closed(self) -> None:
        observed = {"min": -6.1, "max": -0.1}
        self.assertFalse(
            covers(observed, BLOCK0_GPU_DOMAINS["attention_delta_s1_minus_s0"])
        )
        expanded = outward_domain(observed, 1.25)
        self.assertLessEqual(expanded[0], observed["min"])
        self.assertGreaterEqual(expanded[1], observed["max"])

    def test_block_calibration_flags_compiled_contract_failure(self) -> None:
        width = 4
        oracle = {
            "ln1_variance": np.array([[0.004], [0.008]]),
            "ln1_output": np.zeros((2, width)),
            "attention_scores": np.array([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.float64),
            "ln2_variance": np.array([[0.7], [0.9]]),
            "ln2_output": np.zeros((2, width)),
            "mlp_fc": np.zeros((2, 4 * width)),
            "gelu_tanh": np.zeros((2, 4 * width)),
            "block_output": np.zeros((2, width)),
        }
        result = block_calibration(oracle)
        self.assertFalse(result["block0_gpu_fixed_domain_passed"])
        self.assertIn(
            "attention_delta_s1_minus_s0",
            result["failed_block0_gpu_fixed_domains"],
        )
        self.assertIn(
            "attention_denominator",
            result["failed_block0_gpu_fixed_domains"],
        )


if __name__ == "__main__":
    unittest.main()
