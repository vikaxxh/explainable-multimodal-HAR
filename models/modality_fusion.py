"""
Modality Gating and Adaptive Multimodal Fusion.
Phase 14 of implementation plan:
Different behaviors require different modalities:
- Turning: Pose + Trajectory may dominate
- Yielding: Trajectory + Interaction may dominate
- Crossing: Trajectory + Scene context may dominate

Learns:
w_m = softmax(g(F_m))
F_{multi} = sum_m (w_m * F_m)
Exports w_m directly for Modality Attribution in XAI (Phase 20.3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List


class ModalityGating(nn.Module):
    """
    Adaptive modality gating network.
    Dynamically computes importance weights w_m across modalities at each timestep.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_modalities: int = 4,
        reduction_dim: int = 64
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        # Gating network: projects each modality embedding to a scalar score per time step
        self.gating_net = nn.Sequential(
            nn.Linear(feature_dim, reduction_dim),
            nn.Tanh(),
            nn.Linear(reduction_dim, 1)
        )

        # Context-aware gating: allows cross-talk before computing gating weights
        self.cross_gate = nn.Linear(feature_dim * num_modalities, num_modalities)

    def forward(
        self,
        modality_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_dict: Dictionary containing keys ['rgb', 'pose', 'traj', 'scene'],
                           each with tensor of shape (B, T, feature_dim)
        Returns:
            f_multi: (B, T, feature_dim) fused multimodal representation
            gating_weights: (B, T, num_modalities) normalized attention weights across modalities
        """
        keys = ["rgb", "pose", "traj", "scene"]
        # Stack modalities: (B, T, M, d)
        stacked = torch.stack([modality_dict[k] for k in keys], dim=2)
        B, T, M, D = stacked.shape

        # Concatenated context for gating
        concat_feats = stacked.view(B, T, M * D)
        logits = self.cross_gate(concat_feats)  # (B, T, M)
        weights = F.softmax(logits, dim=-1)     # (B, T, M)

        # Weighted sum across modalities: sum_m (w_m * F_m)
        # weights unsqueezed: (B, T, M, 1)
        f_multi = torch.sum(stacked * weights.unsqueeze(-1), dim=2)  # (B, T, D)

        return f_multi, weights
