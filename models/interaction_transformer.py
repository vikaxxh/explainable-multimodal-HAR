"""
Dynamic Interaction Transformer and Fusion.
Phases 16 & 17 of implementation plan:
Processes dynamic graphs G_1, ..., G_T across interacting agents over time:
- Dynamic relational message passing between primary agent and neighbors
- Computes F_{interaction} in R^{B x T x d}
- Fuses multimodal representation F_{multi} and F_{interaction}:
  F_{final} = Fusion(F_{multi}, F_{interaction})
- Extracts interaction edge weights I_{i,j}(t) for Phase 21 XAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class DynamicInteractionTransformer(nn.Module):
    """
    Spatiotemporal Graph Attention Transformer over dynamic agent interactions.
    Incorporates edge features (distance, relative speed, heading, TTC) directly into attention.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        edge_embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Project neighbor trajectory coordinates to agent tokens
        self.neighbor_projector = nn.Sequential(
            nn.Linear(5, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

        # Edge feature integration into query-key relation
        self.edge_to_attn = nn.Linear(edge_embed_dim, feature_dim)

        # Relational Cross-Attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(feature_dim),
                "ffn": nn.Sequential(
                    nn.Linear(feature_dim, feature_dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim * 2, feature_dim),
                    nn.Dropout(dropout)
                ),
                "norm2": nn.LayerNorm(feature_dim)
            })
            for _ in range(num_layers)
        ])

        # Fusion network to combine F_{multi} and F_{interaction}
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        self.fusion_projector = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        f_multi: torch.Tensor,            # (B, T, d) Primary agent multimodal representation
        neighbor_trajs: torch.Tensor,     # (B, T, N, 5)
        edge_embed: torch.Tensor,         # (B, T, N, edge_embed_dim)
        adj_mask: torch.Tensor            # (B, T, N) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f_multi: (B, T, d)
            neighbor_trajs: (B, T, N, 5)
            edge_embed: (B, T, N, edge_embed_dim)
            adj_mask: (B, T, N) bool
        Returns:
            f_final: (B, T, d) combined representation
            f_interaction: (B, T, d) interaction-specific representation
            interaction_weights: (B, T, N) edge attribution importance for XAI
        """
        B, T, N, _ = neighbor_trajs.shape

        # Encode neighbor tokens: (B*T, N, d)
        n_tokens = self.neighbor_projector(neighbor_trajs)  # (B, T, N, d)
        edge_bias = self.edge_to_attn(edge_embed)           # (B, T, N, d)
        n_tokens = n_tokens + edge_bias

        # Primary agent tokens: (B, T, 1, d)
        p_tokens = f_multi.unsqueeze(2)

        # Prepare for attention over N neighbors:
        # Query = Primary agent (B*T, 1, d)
        # Key, Value = Neighbor agents (B*T, N, d)
        q = p_tokens.view(B * T, 1, self.feature_dim)
        kv = n_tokens.view(B * T, N, self.feature_dim)

        # Mask invalid neighbors: key_padding_mask is True for padded/invalid keys
        # adj_mask: (B, T, N) -> False means disconnected / pad
        key_padding_mask = ~adj_mask.view(B * T, N)

        # If a row has all neighbors masked out, unmask first to avoid NaN in softmax
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        h = q
        last_attn_weights = None

        for layer in self.layers:
            attn_module = layer["attn"]
            attn_out, attn_w = attn_module(
                query=h,
                key=kv,
                value=kv,
                key_padding_mask=key_padding_mask,
                need_weights=True
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0)
            attn_w = torch.nan_to_num(attn_w, nan=0.0)
            last_attn_weights = attn_w  # (B*T, 1, N)
            h = layer["norm1"](h + attn_out)
            h = layer["norm2"](h + layer["ffn"](h))

        f_interaction = h.view(B, T, self.feature_dim)  # (B, T, d)
        f_interaction = torch.nan_to_num(f_interaction, nan=0.0)
        interaction_weights = last_attn_weights.view(B, T, N) # (B, T, N)

        # Fuse F_{multi} and F_{interaction} (Phase 17)
        concat_rep = torch.cat([f_multi, f_interaction], dim=-1)
        alpha = self.fusion_gate(concat_rep)
        f_final = alpha * f_multi + (1.0 - alpha) * f_interaction
        f_final = self.fusion_projector(concat_rep)
        f_final = torch.nan_to_num(f_final, nan=0.0)

        return f_final, f_interaction, interaction_weights
