import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fhe"))
sys.path.insert(0, str(ROOT / "DNAGPT"))

from dna_gpt.model.gpt import Block  # noqa: E402
from oracle import block_forward, gate, rel_inf, toy_weights  # noqa: E402


class FHEOracleTests(unittest.TestCase):
    def test_numpy_oracle_matches_upstream_dnagpt_block(self):
        dimension, tokens, heads = 8, 4, 2
        x, params = toy_weights(D=dimension, T=tokens, n_heads=heads, seed=0)
        expected = block_forward(x, params, n_heads=heads)

        block = Block(dimension, heads, dropout=0.0, bias=True).double().eval()
        tensors = {
            "ln1_w": block.ln_1.weight,
            "ln1_b": block.ln_1.bias,
            "ln2_w": block.ln_2.weight,
            "ln2_b": block.ln_2.bias,
            "attn_w": block.attn.c_attn.weight,
            "attn_b": block.attn.c_attn.bias,
            "proj_w": block.attn.c_proj.weight,
            "proj_b": block.attn.c_proj.bias,
            "fc_w": block.mlp.c_fc.weight,
            "fc_b": block.mlp.c_fc.bias,
            "fcp_w": block.mlp.c_proj.weight,
            "fcp_b": block.mlp.c_proj.bias,
        }
        with torch.no_grad():
            for name, tensor in tensors.items():
                tensor.copy_(torch.from_numpy(params[name]))
            got = block(torch.from_numpy(x).unsqueeze(0))[0].numpy()

        np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-11)

    def test_gate_checks_worst_token(self):
        oracle = np.array([[100.0, 0.0], [0.01, 0.0]])
        got = oracle.copy()
        got[1, 0] += 0.01
        result = gate(got, oracle, tol=4e-2)
        self.assertLess(result["global_rel_inf"], 4e-2)
        self.assertGreater(result["worst_token_rel_inf"], 4e-2)
        self.assertFalse(result["passed"])

    def test_rel_inf_rejects_nonfinite_and_zero_oracle(self):
        self.assertEqual(rel_inf([1.0], [0.0]), float("inf"))
        self.assertEqual(rel_inf([np.nan], [1.0]), float("inf"))


if __name__ == "__main__":
    unittest.main()
