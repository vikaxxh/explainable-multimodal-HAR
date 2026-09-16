"""
Quantitative XAI Evaluation Metrics.
Phase 26 of implementation plan:
Calculates:
- Deletion Metric (Area Under Deletion Curve)
- Insertion Metric (Area Under Insertion Curve)
- Monotonicity and Faithfulness Ratios
"""

import numpy as np
from typing import List, Dict, Any


def compute_auc(curve: List[float]) -> float:
    """Computes Area Under Curve normalized to [0, 1]."""
    if len(curve) <= 1:
        return float(curve[0]) if curve else 0.0
    x = np.linspace(0, 1, len(curve))
    return float(np.trapz(curve, x))


def compute_faithfulness_scores(
    deletion_curve: List[float],
    insertion_curve: List[float]
) -> Dict[str, float]:
    """
    Evaluates faithfulness quality.
    A faithful XAI method should have:
    - Low Deletion AUC (rapid drop in confidence when salient evidence removed)
    - High Insertion AUC (rapid increase in confidence when salient evidence added)
    """
    del_auc = compute_auc(deletion_curve)
    ins_auc = compute_auc(insertion_curve)

    # Relative Faithfulness Index: higher is better
    rfi = ins_auc - del_auc

    return {
        "deletion_auc": round(del_auc, 4),
        "insertion_auc": round(ins_auc, 4),
        "faithfulness_index": round(rfi, 4),
        "is_faithful": bool(ins_auc > del_auc)
    }
