"""
Dynamic Interaction Graph Construction.
Phase 15 of implementation plan:
At every timestep t, creates interaction graph G_t = (V_t, E_t):
- Node v_i: agent representation [F_i^{rgb}, F_i^{pose}, F_i^{traj}]
- Edge e_{ij}: dynamic pairwise relational features [d_ij, delta_v_ij, delta_theta_ij, TTC_ij]
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class DynamicInteractionGraph(nn.Module):
    """
    Constructs dynamic multi-agent interaction graphs and edge embeddings.
    Calculates pairwise spatial distance, relative velocity, heading difference, and TTC.
    """

    def __init__(
        self,
        edge_dim: int = 4,
        edge_embed_dim: int = 64,
        distance_threshold: float = 12.0
    ):
        super().__init__()
        self.edge_dim = edge_dim
        self.edge_embed_dim = edge_embed_dim
        self.distance_threshold = distance_threshold

        # Edge feature MLP: projects [dist, delta_v, delta_heading, ttc] to edge_embed_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(edge_embed_dim, edge_embed_dim)
        )

    def compute_relational_edges(
        self,
        primary_traj: torch.Tensor,       # (B, T, 5) [x, y, vx, vy, heading]
        neighbor_trajs: torch.Tensor,     # (B, T, N, 5)
        neighbor_mask: torch.Tensor       # (B, T, N) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes dynamic edge features between primary agent (i=0) and all neighbors j in 1..N.

        Args:
            primary_traj: (B, T, 5)
            neighbor_trajs: (B, T, N, 5)
            neighbor_mask: (B, T, N)
        Returns:
            edge_feats: (B, T, N, 4) raw relational metrics
            edge_embed: (B, T, N, edge_embed_dim) projected features
            adj_mask: (B, T, N) boolean connectivity mask (valid neighbor and within threshold)
        """
        B, T, N, _ = neighbor_trajs.shape
        eps = 1e-5

        p_pos = primary_traj[:, :, :2].unsqueeze(2)      # (B, T, 1, 2)
        n_pos = neighbor_trajs[:, :, :, :2]              # (B, T, N, 2)

        p_vel = primary_traj[:, :, 2:4].unsqueeze(2)     # (B, T, 1, 2)
        n_vel = neighbor_trajs[:, :, :, 2:4]             # (B, T, N, 2)

        p_head = primary_traj[:, :, 4:5].unsqueeze(2)    # (B, T, 1, 1)
        n_head = neighbor_trajs[:, :, :, 4:5]            # (B, T, N, 1)

        # 1. Distance
        rel_pos = p_pos - n_pos                          # (B, T, N, 2)
        dist = torch.norm(rel_pos, dim=-1, keepdim=True) # (B, T, N, 1)

        # 2. Relative speed
        rel_vel = p_vel - n_vel                          # (B, T, N, 2)
        delta_v = torch.norm(rel_vel, dim=-1, keepdim=True) # (B, T, N, 1)

        # 3. Heading angular difference
        diff_head = torch.remainder(torch.abs(p_head - n_head), 2 * math.pi)
        delta_theta = torch.minimum(diff_head, 2 * math.pi - diff_head) # (B, T, N, 1)

        # 4. Closing speed & TTC
        unit_rel = rel_pos / (dist + eps)
        closing_speed = -torch.sum(rel_vel * unit_rel, dim=-1, keepdim=True)
        closing_speed = torch.clamp(closing_speed, min=0.0)

        ttc = torch.where(closing_speed > 0.1, dist / (closing_speed + eps), torch.tensor(10.0, device=dist.device))
        ttc = torch.clamp(ttc, 0.0, 10.0)                # (B, T, N, 1)

        # Combined edge features
        edge_feats = torch.cat([dist, delta_v, delta_theta, ttc], dim=-1) # (B, T, N, 4)
        edge_feats = torch.nan_to_num(edge_feats, nan=0.0, posinf=10.0, neginf=0.0)

        # Connectivity mask: must be within threshold and marked active in neighbor_mask
        adj_mask = (dist.squeeze(-1) <= self.distance_threshold) & neighbor_mask

        # Project edge features
        edge_embed = self.edge_mlp(edge_feats)

        return edge_feats, edge_embed, adj_mask
