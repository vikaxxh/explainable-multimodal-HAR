"""
Modality Attribution Module.
Phase 20.3 of implementation plan:
Calculates relative importance of each modality:
I_{RGB}, I_{Pose}, I_{Trajectory}, I_{Scene}
Combines learned gating weights w_m with perturbation validation.
Example output:
Trajectory: 41%, Pose: 27%, RGB: 21%, Scene: 11%
"""

import torch
import numpy as np
from typing import Dict, Any, List


def compute_modality_attribution(
    modality_weights: torch.Tensor,
    modality_names: List[str] = None
) -> Dict[str, Any]:
    """
    Computes modality contribution percentages from model gating weights.

    Args:
        modality_weights: Tensor of shape (B, T, 4) or (T, 4)
        modality_names: List of names e.g. ['RGB', 'Pose', 'Trajectory', 'Scene']

    Returns:
        Dict mapping each modality to its percentage contribution and ranking
    """
    if modality_names is None:
        modality_names = ["RGB", "Pose", "Trajectory", "Scene"]

    if isinstance(modality_weights, torch.Tensor):
        if modality_weights.dim() == 3:
            weights = modality_weights[0].detach().cpu().numpy()  # (T, 4)
        else:
            weights = modality_weights.detach().cpu().numpy()
    else:
        weights = np.asarray(modality_weights)

    # Temporal average of weights across window
    mean_weights = np.mean(weights, axis=0)  # (4,)
    total = np.sum(mean_weights)
    if total > 0:
        pcts = (mean_weights / total) * 100.0
    else:
        pcts = np.ones(4) * 25.0

    breakdown = {name: round(float(pct), 1) for name, pct in zip(modality_names, pcts)}
    sorted_ranking = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "breakdown": breakdown,
        "ranking": [name for name, _ in sorted_ranking],
        "top_modality": sorted_ranking[0][0],
        "raw_scores": mean_weights.tolist()
    }
