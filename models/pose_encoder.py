"""
Pose Feature Encoder (ST-GCN / Spatial-Temporal Joint Attention).
Phase 6 & 12 of implementation plan:
Encodes skeletal keypoints P in R^{B x T x J x D} (J=18, D=3)
into pose representations F^{pose} in R^{B x T x d} (d=256).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialJointAttention(nn.Module):
    """Computes spatial attention across the J skeletal joints for each frame."""
    def __init__(self, joint_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.fc_in = nn.Linear(joint_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, J, joint_dim)
        h = torch.tanh(self.fc_in(x))      # (B*T, J, hidden_dim)
        scores = self.attn(h)              # (B*T, J, 1)
        weights = F.softmax(scores, dim=1) # (B*T, J, 1)
        pooled = torch.sum(x * weights, dim=1) # (B*T, joint_dim)
        return pooled, weights


class PoseEncoder(nn.Module):
    """
    Spatio-Temporal Pose Encoder combining joint projection,
    spatial joint attention pooling, and temporal smoothing to produce (B, T, d).
    """

    def __init__(
        self,
        num_joints: int = 18,
        joint_dim: int = 3,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        self.feature_dim = feature_dim

        # Per-joint feature projection
        self.joint_embed = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Multi-joint self-attention
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Frame-level projection to feature_dim
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim * num_joints, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, pose_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_seq: Tensor of shape (B, T, J, D)
        Returns:
            f_pose: Tensor of shape (B, T, feature_dim)
        """
        B, T, J, D = pose_seq.shape
        x = pose_seq.view(B * T, J, D)

        # Project joints to hidden dimension
        h = self.joint_embed(x)  # (B*T, J, hidden_dim)

        # Spatial self-attention among joints
        attn_out, _ = self.spatial_attn(h, h, h)  # (B*T, J, hidden_dim)
        h = h + attn_out

        # Flatten joints and project to feature_dim
        flattened = h.view(B * T, J * h.shape[-1])
        out = self.projector(flattened)  # (B*T, feature_dim)

        return out.view(B, T, self.feature_dim)
