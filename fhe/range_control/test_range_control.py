"""Fast tests for the fixed public 12-block approximation schedule."""

from __future__ import annotations

import unittest

import numpy as np

from .schedule import (
    DomainViolation,
    dual_rel_inf,
    evaluate_public_polynomial,
    gelu_tanh,
    public_scaled_inverse_sqrt,
    sigmoid_stable,
    t2_attention_probabilities,
)
from .simulate import DEFAULT_FIXTURE, Fixture, run_simulation


class ScheduleUnitTests(unittest.TestCase):
    def test_t2_sigmoid_identity_is_stable_at_extreme_scores(self) -> None:
        scores = np.array(
            [
                [[0.0, 999.0], [500.0, -500.0]],
                [[0.0, 999.0], [-500.0, 500.0]],
            ],
            dtype=np.float64,
        )
        delta = scores[:, 1, 1] - scores[:, 1, 0]
        exact = sigmoid_stable(delta)
        self.assertTrue(np.isfinite(exact).all())
        self.assertEqual(float(exact[0]), 0.0)
        self.assertEqual(float(exact[1]), 1.0)

    def test_t2_polynomial_matches_softmax_identity(self) -> None:
        scores = np.array(
            [
                [[0.0, 50.0], [-2.0, 1.0]],
                [[0.0, 50.0], [1.5, -0.5]],
            ],
            dtype=np.float64,
        )
        probabilities, report = t2_attention_probabilities(
            scores, [-4.0, 4.0], degree=47
        )
        delta = scores[:, 1, 1] - scores[:, 1, 0]
        np.testing.assert_allclose(
            probabilities[:, 1, 1],
            sigmoid_stable(delta),
            atol=2e-9,
            rtol=0.0,
        )
        self.assertTrue(report["unbounded_exp_denominator_eliminated"])

    def test_public_scaled_inverse_sqrt_accuracy(self) -> None:
        domain = [2.0, 600.0]
        value = np.geomspace(domain[0], domain[1], 4096)
        actual, report = public_scaled_inverse_sqrt(value, domain, degree=63)
        relative = np.max(np.abs(actual * np.sqrt(value) - 1.0))
        self.assertLess(relative, 3.5e-4)
        self.assertGreater(report["public_power_of_two_scale"], 0.0)

    def test_public_domain_violation_fails_closed_without_clipping(self) -> None:
        with self.assertRaises(DomainViolation):
            evaluate_public_polynomial(
                "gelu",
                np.array([-3.1, 0.0, 2.9]),
                [-3.0, 3.0],
                degree=15,
                name="test.gelu",
            )

    def test_tail_aware_gelu_covers_observed_wide_range(self) -> None:
        domain = [-47.92878927034578, 47.92878927034578]
        value = np.linspace(domain[0], domain[1], 20001)
        actual, report = evaluate_public_polynomial(
            "gelu", value, domain, degree=127, name="test.wide_gelu"
        )
        self.assertLess(float(np.max(np.abs(actual - gelu_tanh(value)))), 7e-3)
        self.assertEqual(report["observed"]["min"], domain[0])
        self.assertEqual(report["observed"]["max"], domain[1])

    def test_dual_gate_can_fail_one_token(self) -> None:
        reference = np.array([[100.0, 0.0], [1.0, 0.0]])
        actual = np.array([[100.0, 0.0], [1.05, 0.0]])
        gate = dual_rel_inf(reference, actual)
        self.assertLess(gate["global_rel_inf"], 0.04)
        self.assertGreater(gate["worst_token_rel_inf"], 0.04)
        self.assertFalse(gate["passed"])


@unittest.skipUnless(DEFAULT_FIXTURE.exists(), "generated all-12-block fixture absent")
class FixtureIntegrationTests(unittest.TestCase):
    def test_fixture_loader_rejects_no_contract_fields(self) -> None:
        fixture = Fixture(DEFAULT_FIXTURE)
        self.assertEqual(fixture.manifest["model"]["layers"], 12)
        self.assertEqual(fixture.array("input.embeddings").shape, (2, 768))

    def test_full_12_block_schedule_passes(self) -> None:
        result = run_simulation(DEFAULT_FIXTURE)
        self.assertTrue(result["passed"], result.get("failure"))
        self.assertEqual(result["domain_violation_count"], 0)
        self.assertEqual(len(result["blocks"]), 12)
        self.assertTrue(all(block["passed"] for block in result["blocks"]))
        self.assertTrue(result["head"]["passed"])
        self.assertTrue(
            result["head"]["readout"]["label_preserved"],
            result["head"]["readout"],
        )


if __name__ == "__main__":
    unittest.main()
