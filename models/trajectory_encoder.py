"""
Trajectory and Motion Feature Encoder.
Phase 5 & 12 of implementation plan:
Encodes kinematic trajectory sequences [x, y, vx, vy, heading] in R^{B x T x 5}
into motion representations F^{traj} in R^{B x T x d} (d=256).
"""

import torch
import torch.nn as nn
from typing import Optional


class TrajectoryEncoder(nn.Module):
    """
    Encodes kinematic state sequences into rich temporal motion embeddings.
    Combines 1D temporal convolution with a bidirectional GRU and projection layer.
    """

    def __init__(
        self,
        input_dim: int = 5,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        # Input projection and temporal convolution
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Bidirectional GRU for continuous kinematic context
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Final projection to common feature dimension d
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, traj_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            traj_seq: Tensor of shape (B, T, input_dim)
        Returns:
            f_traj: Tensor of shape (B, T, feature_dim)
        """
        B, T, D = traj_seq.shape

        # 1D Conv expects (B, D, T)
        x = traj_seq.transpose(1, 2)
        x_conv = self.conv1d(x)          # (B, hidden_dim, T)
        x_conv = x_conv.transpose(1, 2)  # (B, T, hidden_dim)

        # Bidirectional GRU
        gru_out, _ = self.gru(x_conv)    # (B, T, hidden_dim)

        # Project to feature_dim
        f_traj = self.projector(gru_out) # (B, T, feature_dim)
        return f_traj
