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

        # Learned token used when a pedestrian has NO valid neighbour at a timestep.
        # (Previously padded slot 0 was un-masked, so "no neighbour" windows attended to a zero agent
        #  whose edge features re-encoded the pedestrian's own position/speed/heading.)
        self.no_neighbor_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Residual fusion: F_final = LN(F_multi + Proj(F_interaction)).
        # With no neighbours the added term is a learned constant, so the only extra information the
        # interaction model receives over the isolated model comes from actual neighbours.
        self.fusion_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim)
        )
        self.fusion_norm = nn.LayerNorm(feature_dim)

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
        q = p_tokens.reshape(B * T, 1, self.feature_dim)
        kv = n_tokens.reshape(B * T, N, self.feature_dim)

        # Mask invalid neighbors: key_padding_mask is True for padded/invalid keys
        key_padding_mask = ~adj_mask.reshape(B * T, N)

        # Rows with no valid neighbour: unmask slot 0 only to keep softmax finite; their outputs are
        # replaced by the learned no-neighbour token below and their attention weights set to 0.
        all_masked = key_padding_mask.all(dim=1)
        safe_mask = key_padding_mask.clone()
        safe_mask[all_masked, 0] = False

        h = q
        last_attn_weights = None

        for layer in self.layers:
            attn_out, attn_w = layer["attn"](
                query=h, key=kv, value=kv,
                key_padding_mask=safe_mask,
                need_weights=True
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0)
            last_attn_weights = torch.nan_to_num(attn_w, nan=0.0)  # (B*T, 1, N)
            h = layer["norm1"](h + attn_out)
            h = layer["norm2"](h + layer["ffn"](h))

        empty = all_masked.view(B * T, 1, 1)
        h = torch.where(empty, self.no_neighbor_token.expand(B * T, 1, self.feature_dim).to(h.dtype), h)
        last_attn_weights = last_attn_weights.masked_fill(empty, 0.0)

        f_interaction = torch.nan_to_num(h.reshape(B, T, self.feature_dim), nan=0.0)
        interaction_weights = last_attn_weights.reshape(B, T, N)

        # Residual fusion (Phase 17)
        f_final = self.fusion_norm(f_multi + self.fusion_projector(f_interaction))
        f_final = torch.nan_to_num(f_final, nan=0.0)

        return f_final, f_interaction, interaction_weights
