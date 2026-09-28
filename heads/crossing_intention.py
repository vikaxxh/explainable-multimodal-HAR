"""
Binary Crossing Intention and Action State Prediction Heads.
Pure human-annotated targets:
- Crossing Intention: [0: Not Crossing, 1: Crossing]
- Action State: [0: Standing, 1: Walking]
"""

import torch
import torch.nn as nn
from models.behavior_query_decoder import BehaviorQueryDecoder


class CrossingIntentionHead(nn.Module):
    """
    Dedicated binary crossing intention head evaluated against ground-truth human annotations.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        dropout: float = 0.1,
        use_query_decoder: bool = True
    ):
        super().__init__()
        self.use_query_decoder = use_query_decoder
        self.query_decoder = BehaviorQueryDecoder(feature_dim=feature_dim, dropout=dropout) if use_query_decoder else None

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2)
        )

    def forward(self, z: torch.Tensor, pool: str = "query") -> torch.Tensor:
        """
        Args:
            z: Latent spatio-temporal representations (B, T, d)
            pool: 'query', 'mean', or 'last'
        Returns:
            logits: (B, 2) binary crossing logits [not_crossing, crossing]
        """
        if pool == "query" and self.query_decoder is not None:
            pooled, _ = self.query_decoder(z)
        elif pool == "last":
            pooled = z[:, -1, :]
        else:
            pooled = torch.mean(z, dim=1)

        return self.classifier(pooled)


class ActionStateHead(nn.Module):
    """
    Action state recognition head (Standing vs. Walking) from human annotations.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 2)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(z, dim=1)
        return self.classifier(pooled)
