"""
Spatial Attribution Module.
Stage 7 of thesis proposal:
Determines which spatial region and which spatial agent bounding box
influenced the behavioral prediction:
- Primary Pedestrian region attribution
- Interacting Micromobility (e.g. e-scooter, bicycle) region attribution
- Background scene context attribution
Uses spatial perturbation and patch saliency scoring.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional


class SpatialAttributionEvaluator:
    """
    Computes spatial region attribution across interacting agents and background.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def compute_agent_spatial_attribution(
        self,
        batch: Dict[str, torch.Tensor],
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Quantifies spatial importance of Primary Agent vs Interacting Neighbor Agents vs Background.

        Args:
            batch: Model batch dictionary
            sample_idx: Sample index within batch

        Returns:
            Dict containing percentage spatial attribution and ranking
        """
        # Baseline prediction
        orig_out = self.model(batch)
        orig_probs = F.softmax(orig_out["ped_logits"][sample_idx], dim=-1)
        target_class = int(torch.argmax(orig_probs))
        base_conf = float(orig_probs[target_class].item())

        # 1. Primary Agent Perturbation (Zero out primary agent trajectory & pose)
        pert_prim = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pert_prim["trajectory"][sample_idx].zero_()
        pert_prim["pose"][sample_idx].zero_()
        prim_out = self.model(pert_prim)
        prim_probs = F.softmax(prim_out["ped_logits"][sample_idx], dim=-1)
        prim_drop = max(0.0, base_conf - float(prim_probs[target_class].item()))

        # 2. Interacting Neighbor Agent Perturbation (Zero out neighbor trajectories & mask)
        pert_neigh = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pert_neigh["neighbor_agents"][sample_idx].zero_()
        pert_neigh["neighbor_mask"][sample_idx].zero_()
        neigh_out = self.model(pert_neigh)
        neigh_probs = F.softmax(neigh_out["ped_logits"][sample_idx], dim=-1)
        neigh_drop = max(0.0, base_conf - float(neigh_probs[target_class].item()))

        # 3. Background Context Perturbation (Zero out scene context)
        pert_scene = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pert_scene["scene"][sample_idx].zero_()
        scene_out = self.model(pert_scene)
        scene_probs = F.softmax(scene_out["ped_logits"][sample_idx], dim=-1)
        scene_drop = max(0.0, base_conf - float(scene_probs[target_class].item()))

        total_impact = prim_drop + neigh_drop + scene_drop + 1e-6
        pct_prim = (prim_drop / total_impact) * 100.0
        pct_neigh = (neigh_drop / total_impact) * 100.0
        pct_scene = (scene_drop / total_impact) * 100.0

        ranking = [
            {"entity": "Primary Pedestrian", "attribution_drop": round(prim_drop, 4), "percentage": round(pct_prim, 1)},
            {"entity": "Interacting Micromobility Agent", "attribution_drop": round(neigh_drop, 4), "percentage": round(pct_neigh, 1)},
            {"entity": "Background Scene Context", "attribution_drop": round(scene_drop, 4), "percentage": round(pct_scene, 1)}
        ]
        ranking.sort(key=lambda x: x["attribution_drop"], reverse=True)

        return {
            "target_class": target_class,
            "baseline_confidence": round(base_conf * 100.0, 2),
            "spatial_attribution": ranking,
            "top_spatial_entity": ranking[0]["entity"],
            "summary": f"Spatial decision evidence was dominated by the {ranking[0]['entity']} ({ranking[0]['percentage']}%)."
        }
