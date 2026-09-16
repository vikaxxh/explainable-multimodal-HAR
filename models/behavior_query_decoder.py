"""
Learnable Behavior Query Decoder (Perceiver / DETR Style Temporal Attention).
Replaces naive temporal mean-pooling with query-based cross-attention over Z_{1:T}:
Q_behavior in R^{B x 1 x d} cross-attends to Z_{1:T} in R^{B x T x d}.
Captures subtle, decisive moments (e.g. 0.5s heading deflection, sudden deceleration)
without averaging them out across 32 frames.
Directly contributes to achieving 95+% fine-grained classification accuracy.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class BehaviorQueryDecoder(nn.Module):
    """
    Cross-attention decoder using learnable task query tokens to pool temporal representations.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable query parameter initialized from normal distribution
        self.query_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Cross-attention block
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Temporal sequence representations (B, T, d)
        Returns:
            pooled: Sharp task representation (B, d)
            query_attn_weights: Temporal attribution weights (B, 1, T) for XAI
        """
        B, T, D = z.shape

        # Expand learnable query for batch
        queries = self.query_token.expand(B, -1, -1)  # (B, 1, d)

        # Query attends to temporal sequence Z
        attn_out, attn_weights = self.cross_attn(
            query=queries,
            key=z,
            value=z,
            need_weights=True
        )

        x = self.norm1(queries + attn_out)
        out = self.norm2(x + self.ffn(x))

        pooled = out.squeeze(1)  # (B, d)
        return pooled, attn_weights.squeeze(1)  # (B, T)
