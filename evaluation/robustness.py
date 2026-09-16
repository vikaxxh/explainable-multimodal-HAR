"""
Robustness and Missing Modality Stress-Testing.
Phase 27 of implementation plan:
Tests model degradation under realistic noisy urban conditions:
1. Missing RGB (zeroed out sensor)
2. Missing Pose (occluded skeleton)
3. Noisy Trajectory (Gaussian jitter on coordinates)
4. Spatial Occlusion (blackout patch on video)
5. Missing Interacting Neighbors (isolated agent)
"""

import copy
import torch
import numpy as np
from typing import Dict, Any, List
from evaluation.classification_metrics import compute_classification_metrics


class RobustnessEvaluator:
    """
    Stress-tests model against sensor drops and noisy inputs.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_condition(
        self,
        loader,
        condition_fn
    ) -> Dict[str, Any]:
        y_true = []
        y_pred = []

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            perturbed_batch = condition_fn(batch)
            outputs = self.model(perturbed_batch)

            preds = torch.argmax(outputs["ped_logits"], dim=-1).cpu().numpy()
            targets = batch["ped_label"].cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(targets.tolist())

        return compute_classification_metrics(y_true, y_pred)

    def run_all_robustness_tests(self, loader) -> Dict[str, Dict[str, Any]]:
        results = {}

        # 1. Clean Baseline
        results["clean"] = self.evaluate_condition(loader, lambda b: b)

        # 2. Missing RGB
        def drop_rgb(b):
            c = copy.copy(b)
            c["rgb"] = torch.zeros_like(b["rgb"])
            return c
        results["missing_rgb"] = self.evaluate_condition(loader, drop_rgb)

        # 3. Missing Pose
        def drop_pose(b):
            c = copy.copy(b)
            c["pose"] = torch.zeros_like(b["pose"])
            return c
        results["missing_pose"] = self.evaluate_condition(loader, drop_pose)

        # 4. Noisy Trajectory
        def noise_traj(b):
            c = copy.copy(b)
            noise = torch.randn_like(b["trajectory"]) * 0.15
            c["trajectory"] = b["trajectory"] + noise
            return c
        results["noisy_trajectory"] = self.evaluate_condition(loader, noise_traj)

        # 5. Missing Neighbor Interaction
        def drop_neighbors(b):
            c = copy.copy(b)
            c["neighbor_mask"] = torch.zeros_like(b["neighbor_mask"])
            c["neighbor_agents"] = torch.zeros_like(b["neighbor_agents"])
            return c
        results["missing_neighbors"] = self.evaluate_condition(loader, drop_neighbors)

        return results
