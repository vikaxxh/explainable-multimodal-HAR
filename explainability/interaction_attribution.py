"""
Interaction Edge Attribution Module.
Phase 21 of implementation plan:
Calculates interaction relational importance I_{i, j} for interacting agent pairs:
e.g. Pedestrian P1 <-> E-scooter S1.
Identifies which surrounding agent influenced the primary agent's behavior decision.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional


def compute_interaction_attribution(
    interaction_weights: torch.Tensor,
    neighbor_types: Optional[List[str]] = None,
    neighbor_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Computes importance scores for each neighbor agent interacting with the primary agent.

    Args:
        interaction_weights: Tensor of shape (B, T, N) or (T, N)
        neighbor_types: Optional list of string labels (e.g. ['escooter', 'bicycle'])
        neighbor_ids: Optional list of agent IDs

    Returns:
        Dict with agent rankings, percentage attribution, and primary interacting agent
    """
    if isinstance(interaction_weights, torch.Tensor):
        if interaction_weights.dim() == 3:
            weights = interaction_weights[0].detach().cpu().numpy()  # (T, N)
        else:
            weights = interaction_weights.detach().cpu().numpy()
    else:
        weights = np.asarray(interaction_weights)

    T, N = weights.shape
    # Average attention across the temporal window
    mean_attn = np.mean(weights, axis=0)  # (N,)
    total = np.sum(mean_attn)
    if total > 0:
        norm_scores = (mean_attn / total) * 100.0
    else:
        norm_scores = np.zeros(N)

    results = []
    for k in range(N):
        name = neighbor_types[k] if neighbor_types and k < len(neighbor_types) else f"Agent_{k+1}"
        agent_id = neighbor_ids[k] if neighbor_ids and k < len(neighbor_ids) else k + 1
        results.append({
            "agent_index": k,
            "agent_id": agent_id,
            "agent_type": name,
            "importance_score": round(float(mean_attn[k]), 4),
            "percentage": round(float(norm_scores[k]), 1)
        })

    results.sort(key=lambda item: item["importance_score"], reverse=True)
    top_agent = results[0] if results else None

    summary_text = (
        f"The {top_agent['agent_type']} (ID: {top_agent['agent_id']}) interaction "
        f"contributed strongly ({top_agent['percentage']}%) to the predicted behavior."
        if top_agent and top_agent["importance_score"] > 0 else "No significant interaction detected."
    )

    return {
        "ranked_interactions": results,
        "primary_interacting_agent": top_agent,
        "summary": summary_text
    }
