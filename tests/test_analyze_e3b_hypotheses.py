import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_e3b_repair import AnalysisGateError
from analyze_e3b_hypotheses import (
    _h2_consistent_wilson_sensitivity,
    _h5_statistics,
    _mover_difference,
    analyze_curve_hypotheses,
    analyze_h4,
    build_curve,
    build_kmin,
    holm_conditional,
    joint_person_bootstrap,
    run_analysis,
)


K_GRID = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64]
SEEDS = [42, 1337, 2024]


def synthetic_attempts() -> pd.DataFrame:
    rows = []
    positive = K_GRID[1:]
    for arm in ("control", "trained"):
        for person_index in range(25):
            for field_index, field in enumerate(("ssn", "email")):
                target_index = person_index * 2 + field_index
                # Spread thresholds over the grid; leave four control targets
                # right-censored to exercise the H4 likelihood.
                threshold_position = min(12, 2 + target_index // 5)
                threshold = positive[threshold_position]
                if arm == "control" and target_index >= 46:
                    threshold = None
                if arm == "trained" and threshold is not None:
                    threshold = positive[max(0, threshold_position - 1)]
                h_bits = 28.0 + 1.5 * target_index
                for seed in SEEDS:
                    for k in K_GRID:
                        hit = bool(k > 0 and threshold is not None and k >= threshold)
                        rows.append(
                            {
                                "target_membership": arm,
                                "person_id": f"{arm}-{person_index:02d}",
                                "field": field,
                                "seed": seed,
                                "capacity_k": k,
                                "exact_match": hit,
                                "target_H_bits": h_bits,
                            }
                        )
    return pd.DataFrame(rows)


class E3bHypothesisAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attempts = synthetic_attempts()

    def test_bootstrap_uses_independent_blocks_and_one_joint_draw_across_k(self):
        result = joint_person_bootstrap(
            self.attempts,
            k_grid=K_GRID,
            n_bootstrap=80,
            bootstrap_seed=19,
        )
        rng = np.random.default_rng(19)
        expected_control = rng.integers(0, 25, size=(80, 25))
        expected_trained = rng.integers(0, 25, size=(80, 25))
        np.testing.assert_array_equal(result["indices"]["control"], expected_control)
        np.testing.assert_array_equal(result["indices"]["trained"], expected_trained)
        self.assertFalse(np.array_equal(expected_control, expected_trained))

        # A target that crosses once must remain ordered within every replicate;
        # independent per-k draws would violate this in the synthetic fixture.
        self.assertTrue(
            np.all(np.diff(result["replicates"]["control"][:, 1:], axis=1) >= 0)
        )

    def test_formal_entrypoint_hard_gates_bootstrap_contract(self):
        nowhere = Path("/does/not/matter")
        common = {
            "attempts_dir": nowhere,
            "manifests_dir": nowhere,
            "target_manifest": nowhere,
            "output_dir": nowhere,
            "expected_formal_commit": "0" * 40,
        }
        with self.assertRaisesRegex(AnalysisGateError, "exactly 10000"):
            run_analysis(**common, n_bootstrap=9999)
        with self.assertRaisesRegex(AnalysisGateError, "bootstrap seed 20240601"):
            run_analysis(**common, bootstrap_seed=7)

    def test_h2_has_no_global_p_and_holm_reserves_one(self):
        bootstrap = joint_person_bootstrap(
            self.attempts,
            k_grid=K_GRID,
            n_bootstrap=120,
            bootstrap_seed=23,
        )
        rows, _ = build_curve(self.attempts, bootstrap, K_GRID)
        hypotheses = analyze_curve_hypotheses(rows, bootstrap, K_GRID)
        self.assertIsNone(hypotheses["H2"]["p_raw"])
        self.assertEqual(hypotheses["H2"]["holm_reserved_p"], 1.0)
        literal = hypotheses["H2"]["literal_one_percent"]
        self.assertIn("joint_any_direction_capacities", literal)
        self.assertIn("joint_positive_signal_capacities", literal)
        sensitivity = hypotheses["H2"]["consistent_interval_sensitivity"]
        self.assertEqual(
            sensitivity["method"],
            "all k use one Wilson convention; tau uses Newcombe/MOVER from the same intervals",
        )
        self.assertEqual(
            [row["effective_n"] for row in sensitivity["conventions"]],
            [100 / 3, 300 / 7],
        )

        # Supply a minimal H4 result to exercise the fixed four-slot family.
        hypotheses["H4"] = {"p_raw_two_sided": 0.03}
        adjusted = holm_conditional(hypotheses)
        self.assertEqual(adjusted["results"]["H2"]["p_raw_or_reserved"], 1.0)
        self.assertTrue(adjusted["results"]["H2"]["h2_reserved_not_tested"])
        self.assertNotIn("H5", adjusted["results"])

    def test_h2_uniform_wilson_worked_example_and_no_positive_joint_point(self):
        def row(k, c_rate, d_rate, tau_low, tau_high):
            return {
                "k": k,
                "control": {"estimate": c_rate},
                "trained": {"estimate": d_rate},
                "tau_reported_95_ci": [tau_low, tau_high],
            }

        # The 2/150 k=4 value is the documented E3a worked example. Formal
        # E3b has 1/150 and is deliberately computed from its own raw rows.
        rows = [row(4, 2 / 150, 3 / 150, -0.01, 0.04), row(6, 28 / 150, 48 / 150, 0.02, 0.24)]
        primary = [
            {
                "tolerance": tolerance,
                "joint_positive_signal_capacities": [6] if tolerance >= 0.30 else [],
            }
            for tolerance in (0.01, 0.05, 0.09, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 0.90, 1.00)
        ]
        result = _h2_consistent_wilson_sensitivity(rows, primary)
        target_only, repeated = result["conventions"]
        self.assertAlmostEqual(target_only["per_k"][0]["control_wilson_95_ci"][1], 0.12598078, places=7)
        self.assertAlmostEqual(repeated["per_k"][0]["control_wilson_95_ci"][1], 0.10518258, places=7)
        for convention in result["conventions"]:
            self.assertTrue(
                all(not item["joint_positive_signal_capacities"] for item in convention["mapping"])
            )
        self.assertTrue(result["k6_primary_positive_joint_disappears_in_both"])
        self.assertTrue(result["all_conventions_have_no_positive_joint_point"])

    def test_holm_numbers_h5_ties_and_boundary_mover(self):
        hypotheses = {
            "H1": {"p_raw_one_sided": 0.01},
            "H2": {},
            "H3": {"p_raw_two_sided": 0.03},
            "H4": {"p_raw_two_sided": 0.04},
        }
        adjusted = holm_conditional(hypotheses)["results"]
        self.assertAlmostEqual(adjusted["H1"]["p_holm"], 0.04)
        self.assertAlmostEqual(adjusted["H3"]["p_holm"], 0.09)
        self.assertAlmostEqual(adjusted["H4"]["p_holm"], 0.09)
        self.assertEqual(adjusted["H2"]["p_holm"], 1.0)

        h5 = _h5_statistics(
            np.asarray([1.0, 2.0, 4.0, 8.0]),
            np.asarray([0.0, 0.2, 0.2, 0.0]),
            np.asarray(
                [
                    [0.0, 0.3, 0.3, 0.0],  # tied maximum: envelope keeps 2 and 4
                    [0.0, 0.2, 0.4, 0.0],
                    [0.0, 0.4, 0.2, 0.0],
                    [0.0, 0.3, 0.3, 0.0],
                ]
            ),
        )
        self.assertEqual(h5["tied_maximum_replicates"], 2)
        self.assertEqual(h5["argmax_envelope_95_ci"].tolist(), [2.0, 4.0])

        flat = _h5_statistics(
            np.asarray([1.0, 2.0, 4.0, 8.0]),
            np.zeros(4),
            np.zeros((20, 4)),
        )
        self.assertEqual(flat["argmax_envelope_95_ci"].tolist(), [1.0, 8.0])
        self.assertFalse(flat["interior_peak_localized"])
        self.assertEqual(flat["verdict"], "exploratory_peak_not_localized")

        boundary = _mover_difference(0.0, 0.0, 100 / 3, 100 / 3)
        self.assertAlmostEqual(boundary[0], -0.1033350450, places=9)
        self.assertAlmostEqual(boundary[1], 0.1033350450, places=9)

    def test_h4_keeps_right_censored_targets_and_refits_every_draw(self):
        bootstrap = joint_person_bootstrap(
            self.attempts,
            k_grid=K_GRID,
            n_bootstrap=30,
            bootstrap_seed=29,
        )
        kmin = build_kmin(self.attempts, K_GRID[1:])
        result = analyze_h4(
            kmin,
            bootstrap["indices"]["control"],
            repo_root=Path(__file__).resolve().parents[1],
        )
        self.assertEqual(result["n_targets"], 50)
        self.assertEqual(result["right_censored_targets"], 4)
        self.assertEqual(result["bootstrap_replicates"], 30)
        self.assertEqual(result["bootstrap_failures"], [])
        self.assertTrue(np.isfinite(result["gamma"]))
        self.assertTrue(np.isfinite(result["lognormal_distribution_sensitivity"]["gamma"]))
        self.assertEqual(
            result["lognormal_distribution_sensitivity"]["bootstrap_replicates"], 30
        )
        self.assertEqual(
            result["solver"]["sha256"],
            "ed231efdd9a03d4fb630fd93316fb09477a8b08b2d06d3ff978ada91831ec8ba",
        )


if __name__ == "__main__":
    unittest.main()
