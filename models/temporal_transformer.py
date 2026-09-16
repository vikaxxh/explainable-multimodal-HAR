"""
Spatio-Temporal Sequence Transformer.
Phase 18 of implementation plan:
Models temporal transitions across sequence window T (e.g. 16, 32, 64 frames):
F_{1:T} -> Transformer -> Z_{1:T}
Learns progressive behavior shifts (e.g. Walking -> Approaching -> Slowing -> Yielding).
Exports temporal attention matrix for Phase 20.1 Temporal Attribution (I_t).
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        return x + self.pe[:, :x.size(1), :]


class TemporalTransformer(nn.Module):
    """
    Multi-layer temporal Transformer with self-attention across the temporal window.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.pos_encoder = PositionalEncoding(feature_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(feature_dim),
                "ffn": nn.Sequential(
                    nn.Linear(feature_dim, dim_feedforward),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, feature_dim),
                    nn.Dropout(dropout)
                ),
                "norm2": nn.LayerNorm(feature_dim)
            })
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, T, d)
        Returns:
            z: Tensor of shape (B, T, d)
            temporal_attn: Tensor of shape (B, T, T) temporal attention weights for XAI
        """
        h = self.pos_encoder(x)
        last_attn = None

        for layer in self.layers:
            attn_module = layer["attn"]
            attn_out, attn_w = attn_module(query=h, key=h, value=h, need_weights=True)
            last_attn = attn_w  # (B, T, T)
            h = layer["norm1"](h + attn_out)
            h = layer["norm2"](h + layer["ffn"](h))

        z = self.final_norm(h)
        return z, last_attn
