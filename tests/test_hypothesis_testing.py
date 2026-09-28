"""
Unit tests for evaluation/hypothesis_testing.py:
- Holm-Bonferroni correction
- Cluster-bootstrap confidence interval calculation
- H2 stratified utility evaluation
- H3 matched-control paired testing
- H4 Spearman rank correlation
- Mahalanobis OOD calculation
"""

import unittest
import numpy as np
from sklearn.metrics import accuracy_score

from evaluation.hypothesis_testing import (
    holm_bonferroni,
    cluster_bootstrap_ci,
    evaluate_h2_stratified_utility,
    evaluate_h3_matched_control_removal,
    evaluate_h4_kinematic_urgency,
    compute_mahalanobis_ood
)


class TestHypothesisTesting(unittest.TestCase):

    def test_holm_bonferroni_correction(self):
        p_vals = [0.01, 0.04, 0.03, 0.20]
        adj = holm_bonferroni(p_vals, alpha=0.05)
        # Smallest p=0.01 multiplied by 4 = 0.04 <= 0.05 (significant)
        self.assertEqual(len(adj), 4)
        self.assertTrue(adj[0][1])  # 0.01 should be significant
        self.assertFalse(adj[3][1]) # 0.20 should not be significant

    def test_cluster_bootstrap_ci(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1])
        cluster_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

        ci = cluster_bootstrap_ci(
            accuracy_score, y_true, y_pred, cluster_ids, num_bootstrap=50, seed=42
        )
        self.assertIn("point_estimate", ci)
        self.assertIn("ci_lower", ci)
        self.assertIn("ci_upper", ci)
        self.assertLessEqual(ci["ci_lower"], ci["point_estimate"])
        self.assertGreaterEqual(ci["ci_upper"], ci["point_estimate"])

    def test_h3_matched_control_paired_testing(self):
        # Scenario: top neighbor removal drops confidence consistently more than matched control
        deltas_top = np.array([0.45, 0.50, 0.38, 0.42, 0.55, 0.48])
        deltas_ctrl = np.array([0.10, 0.15, 0.12, 0.08, 0.20, 0.14])

        res = evaluate_h3_matched_control_removal(deltas_top, deltas_ctrl)
        self.assertTrue(res["is_significant"])
        self.assertGreater(res["median_difference"], 0.2)
        self.assertLess(res["p_value"], 0.05)

    def test_h4_kinematic_urgency_monotonicity(self):
        scale_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        # Closing: probability decreases as speed increases
        probs = np.array([
            [0.90, 0.80, 0.70, 0.55, 0.40, 0.20],
            [0.85, 0.75, 0.65, 0.50, 0.35, 0.15],
            [0.88, 0.78, 0.68, 0.52, 0.38, 0.18],
        ])
        is_closing = np.array([True, True, True])

        res = evaluate_h4_kinematic_urgency(scale_grid, probs, is_closing)
        self.assertTrue(res["is_supported"])
        self.assertEqual(res["expected_negative_share"], 100.0)

    def test_mahalanobis_ood(self):
        train = np.random.randn(100, 4)
        perturbed = train + 0.1 * np.random.randn(100, 4)
        ood = compute_mahalanobis_ood(train, perturbed)
        self.assertIn("mean_mahalanobis_distance", ood)
        self.assertIn("feature_range_violation_pct", ood)
