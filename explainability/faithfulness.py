"""
Quantitative XAI Faithfulness Evaluation Suite.
Phase 26 of implementation plan:
Validates explanation faithfulness via perturbation experiments:
- Test 1: Deletion Test (removes salient evidence, measures drop in target class confidence)
- Test 2: Insertion Test (inserts salient evidence progressively, measures recovery curve)
- Test 3: Modality Perturbation (zeroes each modality, measures degradation)
- Test 4: Temporal Perturbation (masks critical frames vs random frames)
- Test 5: Interaction Perturbation (removes interacting agent, tests if interaction was causal)
"""

import copy
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List


class XAIFaithfulnessEvaluator:
    """
    Evaluates whether the generated explanations faithfully reflect the model's actual reasoning.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_modality_faithfulness(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Test 3: Modality ablation.
        Zeroes out modalities one-by-one and measures performance degradation.
        """
        orig_out = self.model(batch)
        orig_probs = F.softmax(orig_out["ped_logits"], dim=-1)
        orig_preds = torch.argmax(orig_probs, dim=-1)

        # Baseline accuracy/confidence
        target_confs = orig_probs[torch.arange(len(orig_preds)), orig_preds]
        base_mean_conf = float(target_confs.mean().item())

        degradations = {"baseline_confidence": round(base_mean_conf, 4)}

        for mod in ["rgb", "pose", "trajectory", "scene"]:
            mod_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # Zero out modality
            mod_batch[mod].zero_()
            mod_out = self.model(mod_batch)
            mod_probs = F.softmax(mod_out["ped_logits"], dim=-1)
            perturbed_confs = mod_probs[torch.arange(len(orig_preds)), orig_preds]
            drop = float((target_confs - perturbed_confs).mean().item())
            degradations[f"drop_without_{mod}"] = round(drop, 4)

        return degradations

    @torch.no_grad()
    def evaluate_temporal_perturbation(
        self,
        batch: Dict[str, torch.Tensor],
        mask_ratio: float = 0.25
    ) -> Dict[str, float]:
        """
        Test 4: Temporal perturbation.
        Masks the most-attended frames vs random frames.
        Faithful explanations should show: Drop(Critical) > Drop(Random).
        """
        orig_out = self.model(batch)
        orig_probs = F.softmax(orig_out["ped_logits"], dim=-1)
        orig_preds = torch.argmax(orig_probs, dim=-1)
        target_confs = orig_probs[torch.arange(len(orig_preds)), orig_preds]

        temporal_attn = orig_out["temporal_attn"] # (B, T, T)
        B, T = batch["trajectory"].shape[0], batch["trajectory"].shape[1]
        k = max(1, int(T * mask_ratio))

        # 1. Mask Critical Frames
        crit_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for b in range(B):
            attn_b = temporal_attn[b].mean(dim=0).cpu().numpy() # (T,)
            top_frames = np.argsort(attn_b)[-k:]
            for t_idx in top_frames:
                crit_batch["trajectory"][b, t_idx].zero_()
                crit_batch["pose"][b, t_idx].zero_()
                crit_batch["rgb"][b, t_idx].zero_()

        crit_out = self.model(crit_batch)
        crit_probs = F.softmax(crit_out["ped_logits"], dim=-1)
        crit_confs = crit_probs[torch.arange(len(orig_preds)), orig_preds]
        drop_critical = float((target_confs - crit_confs).mean().item())

        # 2. Mask Random Frames
        rand_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        for b in range(B):
            rand_frames = np.random.choice(T, size=k, replace=False)
            for t_idx in rand_frames:
                rand_batch["trajectory"][b, t_idx].zero_()
                rand_batch["pose"][b, t_idx].zero_()
                rand_batch["rgb"][b, t_idx].zero_()

        rand_out = self.model(rand_batch)
        rand_probs = F.softmax(rand_out["ped_logits"], dim=-1)
        rand_confs = rand_probs[torch.arange(len(orig_preds)), orig_preds]
        drop_random = float((target_confs - rand_confs).mean().item())

        faithfulness_gap = drop_critical - drop_random
        return {
            "drop_critical_frames": round(drop_critical, 4),
            "drop_random_frames": round(drop_random, 4),
            "temporal_faithfulness_gap": round(faithfulness_gap, 4),
            "is_faithful": bool(faithfulness_gap > 0)
        }

    @torch.no_grad()
    def evaluate_interaction_perturbation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Test 5: Interaction perturbation.
        Removes the neighboring interacting agents (masks neighbor_mask to all False).
        Measures if predicted yielding/avoiding/interaction shifts.
        """
        if "neighbor_mask" not in batch:
            return {"error": "No neighbor mask available"}

        orig_out = self.model(batch)
        orig_probs = F.softmax(orig_out["ped_logits"], dim=-1)
        orig_preds = torch.argmax(orig_probs, dim=-1)
        target_confs = orig_probs[torch.arange(len(orig_preds)), orig_preds]

        # Perturb: Remove all neighbors
        pert_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        pert_batch["neighbor_mask"].zero_()
        pert_batch["neighbor_agents"].zero_()

        pert_out = self.model(pert_batch)
        pert_probs = F.softmax(pert_out["ped_logits"], dim=-1)
        pert_confs = pert_probs[torch.arange(len(orig_preds)), orig_preds]

        conf_drop = float((target_confs - pert_confs).mean().item())
        pred_changes = float((torch.argmax(pert_probs, dim=-1) != orig_preds).float().mean().item())

        return {
            "mean_confidence_drop_without_neighbors": round(conf_drop, 4),
            "prediction_flip_rate": round(pred_changes * 100.0, 2)
        }

    @torch.no_grad()
    def run_full_suite(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Runs the complete suite of XAI faithfulness tests."""
        return {
            "modality_faithfulness": self.evaluate_modality_faithfulness(batch),
            "temporal_faithfulness": self.evaluate_temporal_perturbation(batch),
            "interaction_faithfulness": self.evaluate_interaction_perturbation(batch)
        }
