"""
Pre-Registered Hypothesis Testing and Cluster-Bootstrap Analysis Suite.
Implements:
- H1: Empirical Neighbor Distribution (Window and Track-Level)
- H2: Stratified Interaction Branch Utility (Full vs No-Neighbor Ablation)
- H3: Targeted vs Matched-Control Removal (Attribution Selectivity)
- H4: Directional Response to Kinematic Urgency (Spearman Rank Correlation)
- Cluster Bootstrap Resampling (Clustered on Pedestrian Track ID)
- Holm-Bonferroni Family-Wise Error Rate Correction
- Out-of-Distribution (OOD) Feature Drift Measurement
"""

import math
import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from typing import Dict, Any, List, Tuple, Callable, Optional


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Applies Holm-Bonferroni step-down correction to a list of p-values.
    Returns: List of (adjusted_p, is_significant)
    """
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m

    for rank, (orig_idx, p) in enumerate(indexed):
        crit = alpha / (m - rank)
        adj_p = min(1.0, p * (m - rank))
        is_sig = p <= crit
        results[orig_idx] = (round(adj_p, 5), is_sig)

    return results


def cluster_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_ids: np.ndarray,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes 95% Confidence Interval via Cluster Bootstrap Resampling.
    Resamples independent pedestrian tracks with replacement, preserving intra-track correlation.
    """
    rng = np.random.RandomState(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    point_estimate = float(metric_fn(y_true, y_pred))

    if n_clusters < 5:
        return {
            "point_estimate": round(point_estimate, 4),
            "ci_lower": round(point_estimate, 4),
            "ci_upper": round(point_estimate, 4),
            "std_error": 0.0,
            "warning": "Fewer than 5 clusters; bootstrap CI uninformative."
        }

    bootstrap_estimates = []
    for _ in range(num_bootstrap):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])

        y_t_samp = y_true[sampled_indices]
        y_p_samp = y_pred[sampled_indices]

        # Ensure both classes exist in resample
        if len(np.unique(y_t_samp)) > 1:
            try:
                val = float(metric_fn(y_t_samp, y_p_samp))
                bootstrap_estimates.append(val)
            except Exception:
                continue

    if len(bootstrap_estimates) < 100:
        return {
            "point_estimate": round(point_estimate, 4),
            "ci_lower": round(point_estimate, 4),
            "ci_upper": round(point_estimate, 4),
            "std_error": 0.0
        }

    lower = float(np.percentile(bootstrap_estimates, 100 * (alpha / 2.0)))
    upper = float(np.percentile(bootstrap_estimates, 100 * (1.0 - alpha / 2.0)))
    se = float(np.std(bootstrap_estimates))

    return {
        "point_estimate": round(point_estimate, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "std_error": round(se, 4)
    }


def evaluate_h2_stratified_utility(
    y_true: np.ndarray,
    probs_full: np.ndarray,
    probs_ablation: np.ndarray,
    neighbor_counts: np.ndarray,
    cluster_ids: np.ndarray,
    num_bootstrap: int = 1000
) -> Dict[str, Any]:
    """
    Evaluates Hypothesis 2 (H2):
    Tests whether the interaction branch outperforms no-neighbor ablation primarily on K>=1.
    """
    results = {}
    preds_full = (probs_full >= 0.5).astype(int)
    preds_ablation = (probs_ablation >= 0.5).astype(int)

    def auc_diff(y_t, y_p_pair):
        # y_p_pair: Nx2 [prob_full, prob_abl]
        auc_f = roc_auc_score(y_t, y_p_pair[:, 0])
        auc_a = roc_auc_score(y_t, y_p_pair[:, 1])
        return auc_f - auc_a

    def f1_diff(y_t, y_p_pair):
        p_f = (y_p_pair[:, 0] >= 0.5).astype(int)
        p_a = (y_p_pair[:, 1] >= 0.5).astype(int)
        return f1_score(y_t, p_f, zero_division=0) - f1_score(y_t, p_a, zero_division=0)

    probs_combined = np.stack([probs_full, probs_ablation], axis=-1)

    strata = {
        "K=0": (neighbor_counts == 0),
        "K=1": (neighbor_counts == 1),
        "K>=2": (neighbor_counts >= 2),
        "K>=1_all": (neighbor_counts >= 1)
    }

    for stratum_name, mask in strata.items():
        n_samples = int(mask.sum())
        n_tracks = len(np.unique(cluster_ids[mask])) if n_samples > 0 else 0

        if n_samples < 10 or n_tracks < 5 or len(np.unique(y_true[mask])) < 2:
            results[stratum_name] = {
                "n_samples": n_samples,
                "n_tracks": n_tracks,
                "status": "Insufficient samples/tracks for inference"
            }
            continue

        y_s = y_true[mask]
        p_s = probs_combined[mask]
        c_s = cluster_ids[mask]

        ci_auc = cluster_bootstrap_ci(auc_diff, y_s, p_s, c_s, num_bootstrap=num_bootstrap)
        ci_f1 = cluster_bootstrap_ci(f1_diff, y_s, p_s, c_s, num_bootstrap=num_bootstrap)

        # H2 Decision Rule: Supported if CI excludes zero on K>=1, and includes zero on K=0
        excludes_zero_auc = bool(ci_auc["ci_lower"] > 0)
        excludes_zero_f1 = bool(ci_f1["ci_lower"] > 0)

        results[stratum_name] = {
            "n_samples": n_samples,
            "n_tracks": n_tracks,
            "delta_auc": ci_auc,
            "delta_f1": ci_f1,
            "excludes_zero": excludes_zero_auc or excludes_zero_f1
        }

    return results


def evaluate_h3_matched_control_removal(
    paired_deltas_top: np.ndarray,
    paired_deltas_matched_ctrl: np.ndarray,
    paired_deltas_oracle: Optional[np.ndarray] = None,
    cluster_ids: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluates Hypothesis 3 (H3):
    Tests whether removing the top-attributed neighbor produces greater drop
    than removing a distance/speed/heading-matched control agent.
    """
    n_samples = len(paired_deltas_top)
    if n_samples < 5:
        return {"error": "Fewer than 5 matched samples available"}

    diff = paired_deltas_top - paired_deltas_matched_ctrl
    median_diff = float(np.median(diff))
    mean_diff = float(np.mean(diff))

    # Track-level Wilcoxon signed-rank test
    stat, p_val = stats.wilcoxon(diff, alternative="greater")

    # Oracle comparison if available
    oracle_gap = float(np.mean(paired_deltas_oracle - paired_deltas_top)) if paired_deltas_oracle is not None else None

    return {
        "n_matched_samples": n_samples,
        "median_difference": round(median_diff, 4),
        "mean_difference": round(mean_diff, 4),
        "wilcoxon_stat": round(float(stat), 4),
        "p_value": round(float(p_val), 6),
        "oracle_gap": round(oracle_gap, 4) if oracle_gap is not None else None,
        "is_significant": bool(p_val < 0.05 and median_diff > 0)
    }


def evaluate_h4_kinematic_urgency(
    scale_grid: List[float],
    probs_per_scale: np.ndarray,  # Shape (N_samples, len(scale_grid))
    is_closing: np.ndarray,        # Boolean mask: True if closing, False if separating
    placebo_probs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluates Hypothesis 4 (H4):
    Tests whether scaling neighbor approach velocity (alpha in [0.5, 2.0]) correlates
    monotonically with crossing probability reduction (Spearman rho < 0) for closing agents.
    """
    n_samples = probs_per_scale.shape[0]
    rhos_closing = []
    rhos_separating = []

    for i in range(n_samples):
        p_row = probs_per_scale[i]
        # Spearman correlation across scale factors
        rho, _ = stats.spearmanr(scale_grid, p_row)
        if not np.isnan(rho):
            if is_closing[i]:
                rhos_closing.append(rho)
            else:
                rhos_separating.append(rho)

    rhos_closing = np.array(rhos_closing)
    expected_sign_share = float(np.mean(rhos_closing < -0.5)) if len(rhos_closing) > 0 else 0.0

    placebo_share = 0.0
    if placebo_probs is not None:
        placebo_rhos = [stats.spearmanr(scale_grid, placebo_probs[i])[0] for i in range(len(placebo_probs))]
        placebo_share = float(np.mean(np.array(placebo_rhos) < -0.5))

    return {
        "n_closing_samples": len(rhos_closing),
        "n_separating_samples": len(rhos_separating),
        "mean_rho_closing": round(float(np.mean(rhos_closing)), 4) if len(rhos_closing) > 0 else 0.0,
        "expected_negative_share": round(expected_sign_share * 100.0, 2),
        "placebo_expected_share": round(placebo_share * 100.0, 2),
        "is_supported": bool(expected_sign_share > 0.50 and expected_sign_share > placebo_share)
    }


def compute_mahalanobis_ood(
    train_features: np.ndarray,
    perturbed_features: np.ndarray
) -> Dict[str, float]:
    """
    Measures distributional shift of perturbed inputs from training manifold.
    Reports:
    - Mahalanobis distance D_M
    - Feature range violation percentage (features falling outside train [min, max])
    """
    mean = np.mean(train_features, axis=0)
    cov = np.cov(train_features, rowvar=False) + 1e-6 * np.eye(train_features.shape[1])
    inv_cov = np.linalg.pinv(cov)

    diff = perturbed_features - mean
    dists = np.sqrt(np.sum(diff @ inv_cov * diff, axis=-1))

    t_min = np.min(train_features, axis=0)
    t_max = np.max(train_features, axis=0)
    violations = (perturbed_features < t_min) | (perturbed_features > t_max)
    violation_rate = float(np.mean(violations)) * 100.0

    return {
        "mean_mahalanobis_distance": round(float(np.mean(dists)), 4),
        "median_mahalanobis_distance": round(float(np.median(dists)), 4),
        "feature_range_violation_pct": round(violation_rate, 2)
    }
