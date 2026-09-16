"""
Micromobility Behavior Classification Head.
Phase 19 of implementation plan:
Task B: Recognizes 9 micromobility behaviors (e-scooters, bicycles):
[Moving, Stopping, Starting, Turning, Accelerating, Decelerating, Yielding, Avoiding, Overtaking]
"""

import torch
import torch.nn as nn
from datasets.taxonomy import MICROMOBILITY_CLASSES
from models.behavior_query_decoder import BehaviorQueryDecoder


class MicromobilityBehaviorHead(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,
        num_classes: int = len(MICROMOBILITY_CLASSES),
        dropout: float = 0.1,
        use_query_decoder: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_query_decoder = use_query_decoder
        self.query_decoder = BehaviorQueryDecoder(feature_dim=feature_dim, dropout=dropout) if use_query_decoder else None

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )

    def forward(self, z: torch.Tensor, pool: str = "query") -> torch.Tensor:
        """
        Args:
            z: Tensor of shape (B, T, d)
            pool: 'query', 'mean', 'last', or 'max' pooling over time
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        if pool == "query" and self.query_decoder is not None:
            pooled, _ = self.query_decoder(z)
        elif pool == "mean":
            pooled = torch.mean(z, dim=1)
        elif pool == "last":
            pooled = z[:, -1, :]
        elif pool == "max":
            pooled, _ = torch.max(z, dim=1)
        else:
            pooled = torch.mean(z, dim=1)

        return self.classifier(pooled)
