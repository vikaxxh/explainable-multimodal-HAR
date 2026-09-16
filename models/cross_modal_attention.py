"""
Cross-Modal Attention Module.
Phase 13 of implementation plan:
Instead of raw concatenation F = [F1; F2; F3; F4], applies multi-head cross-attention:
Q = F^{traj}, K = F^{rgb}, V = F^{rgb}
A_{traj -> rgb} = softmax(Q K^T / sqrt(d)) * V
Repeated across modality pairs with residual connections and attention map extraction for XAI.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class CrossModalAttentionBlock(nn.Module):
    """
    Multi-head cross-modal attention block where Query attends to Key/Value.
    Includes residual connection, LayerNorm, and Feed-Forward Network.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, T, d)
            key_value: (B, T, d)
        Returns:
            out: (B, T, d)
            attn_weights: (B, T, T)
        """
        attn_out, attn_weights = self.attn(query, key_value, key_value, need_weights=True)
        x = self.norm1(query + attn_out)
        out = self.norm2(x + self.ffn(x))
        return out, attn_weights


class CrossModalFusion(nn.Module):
    """
    Comprehensive cross-modal fusion layer between:
    - Trajectory <-> RGB
    - Trajectory <-> Pose
    - Trajectory <-> Scene
    - Pose <-> RGB
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

        # Pairwise cross-attentions anchored on trajectory and visual representations
        self.cross_traj_rgb = CrossModalAttentionBlock(feature_dim, num_heads, dim_feedforward, dropout)
        self.cross_traj_pose = CrossModalAttentionBlock(feature_dim, num_heads, dim_feedforward, dropout)
        self.cross_traj_scene = CrossModalAttentionBlock(feature_dim, num_heads, dim_feedforward, dropout)
        self.cross_pose_rgb = CrossModalAttentionBlock(feature_dim, num_heads, dim_feedforward, dropout)

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_pose: torch.Tensor,
        f_traj: torch.Tensor,
        f_scene: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            f_rgb, f_pose, f_traj, f_scene: Each (B, T, d)
        Returns:
            enhanced_modalities: Dict with keys 'rgb', 'pose', 'traj', 'scene'
            attention_maps: Dict of cross-attention matrices for XAI
        """
        # Cross-attentions
        traj_from_rgb, attn_tr = self.cross_traj_rgb(f_traj, f_rgb)
        traj_from_pose, attn_tp = self.cross_traj_pose(f_traj, f_pose)
        traj_from_scene, attn_ts = self.cross_traj_scene(f_traj, f_scene)
        pose_from_rgb, attn_pr = self.cross_pose_rgb(f_pose, f_rgb)

        # Enhanced fused representations per modality
        enhanced = {
            "rgb": f_rgb,
            "pose": (f_pose + pose_from_rgb) * 0.5,
            "traj": (f_traj + traj_from_rgb + traj_from_pose + traj_from_scene) / 4.0,
            "scene": f_scene
        }

        attn_maps = {
            "traj_rgb": attn_tr,
            "traj_pose": attn_tp,
            "traj_scene": attn_ts,
            "pose_rgb": attn_pr
        }

        return enhanced, attn_maps
