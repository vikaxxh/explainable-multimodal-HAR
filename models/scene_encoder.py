"""
Scene and Context Feature Encoder.
Phase 7 & 12 of implementation plan:
Encodes semantic scene context distributions S in R^{B x T x K} (K=10 classes)
into context embeddings F^{scene} in R^{B x T x d} (d=256).
"""

import torch
import torch.nn as nn
from typing import Optional


class SceneEncoder(nn.Module):
    """
    Context Encoder projecting semantic context distributions to common dimension d=256.
    Allows distinguishing actions like 'walking on sidewalk' from 'entering roadway'.
    """

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, scene_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_seq: Tensor of shape (B, T, num_classes)
        Returns:
            f_scene: Tensor of shape (B, T, feature_dim)
        """
        return self.net(scene_seq)
