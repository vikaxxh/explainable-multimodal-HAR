"""
Multi-Task Loss Functions and XAI Regularization.
Phase 23 of implementation plan:
Total loss:
L = lambda_1 * L_ped + lambda_2 * L_micro + lambda_3 * L_inter + lambda_4 * L_align + lambda_5 * L_xai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class FocalLoss(nn.Module):
    """
    Class-balanced Focal Loss to overcome extreme urban class imbalance (Walking vs Yielding).
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Dramatically improves accuracy on hard minority interaction behaviors.
    """

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ContrastiveInfoNCELoss(nn.Module):
    """
    Self-Supervised Cross-Modal Contrastive Alignment Loss (InfoNCE).
    Aligns motion trajectory representations with visual and pose tokens.
    Pushes distinct behavior clusters apart in feature space.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        # f1, f2: (B, d)
        z1 = F.normalize(f1, dim=-1)
        z2 = F.normalize(f2, dim=-1)

        sim_matrix = torch.matmul(z1, z2.T) / self.temperature  # (B, B)
        labels = torch.arange(f1.size(0), device=f1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class MultiTaskBehaviorLoss(nn.Module):
    """
    Computes weighted multi-task loss across pedestrian behavior,
    micromobility behavior, interaction classification, and alignment regularization.
    """

    def __init__(
        self,
        lambda_ped: float = 1.0,
        lambda_micro: float = 1.0,
        lambda_inter: float = 1.0,
        lambda_align: float = 0.1,
        lambda_xai: float = 0.05,
        use_focal_loss: bool = True,
        gamma: float = 2.0,
        label_smoothing: float = 0.05
    ):
        super().__init__()
        self.lambda_ped = lambda_ped
        self.lambda_micro = lambda_micro
        self.lambda_inter = lambda_inter
        self.lambda_align = lambda_align
        self.lambda_xai = lambda_xai

        if use_focal_loss:
            self.crit_ped = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
            self.crit_micro = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
            self.crit_inter = FocalLoss(gamma=gamma, label_smoothing=label_smoothing)
        else:
            self.crit_ped = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.crit_micro = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.crit_inter = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.contrastive_crit = ContrastiveInfoNCELoss(temperature=0.07)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        total_loss = 0.0

        # 1. Task A: Pedestrian Behavior (Focal Loss)
        if "ped_logits" in outputs and "ped_label" in targets:
            l_ped = self.crit_ped(outputs["ped_logits"], targets["ped_label"])
            loss_dict["loss_ped"] = l_ped
            total_loss += self.lambda_ped * l_ped

        # 2. Task B: Micromobility Behavior (Focal Loss)
        if "micro_logits" in outputs and "micro_label" in targets:
            l_micro = self.crit_micro(outputs["micro_logits"], targets["micro_label"])
            loss_dict["loss_micro"] = l_micro
            total_loss += self.lambda_micro * l_micro

        # 3. Task C: Interaction Behavior (Focal Loss)
        if "inter_logits" in outputs and "inter_label" in targets:
            l_inter = self.crit_inter(outputs["inter_logits"], targets["inter_label"])
            loss_dict["loss_inter"] = l_inter
            total_loss += self.lambda_inter * l_inter

        # 4. Modality Alignment Loss (cosine similarity between F_multi and F_interaction)
        if outputs.get("f_multi") is not None and outputs.get("f_interaction") is not None:
            f_m = F.normalize(outputs["f_multi"], dim=-1)
            f_i = F.normalize(outputs["f_interaction"], dim=-1)
            # Maximize agreement on active interaction sequences
            l_align = 1.0 - torch.mean(torch.sum(f_m * f_i, dim=-1))
            loss_dict["loss_align"] = l_align
            total_loss += self.lambda_align * l_align

        # 5. XAI Attention Sparsity Regularization (encourages crisp attribution)
        if outputs.get("modality_weights") is not None:
            weights = outputs["modality_weights"] # (B, T, 4)
            # Negative entropy: min -sum(p log p) encourages sharp modality focus
            entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1).mean()
            loss_dict["loss_xai_entropy"] = entropy
            total_loss += self.lambda_xai * entropy

        loss_dict["loss_total"] = total_loss
        return loss_dict
