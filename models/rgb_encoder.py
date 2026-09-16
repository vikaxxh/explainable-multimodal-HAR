"""
RGB Visual Feature Encoder.
Phase 12 of implementation plan:
Projects video clips [B, T, 3, H, W] to visual token features F^{rgb} in R^{B x T x d} (d=256).
Supports lightweight CNN backbone or ResNet with temporal projection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class LightweightVisualBackbone(nn.Module):
    """Fast 2D CNN backbone for processing video frames efficiently."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W)
        feat = self.conv(x)  # (N, 256, 1, 1)
        feat = feat.flatten(1)
        return self.fc(feat)


class RGBEncoder(nn.Module):
    """
    Encodes video sequence (B, T, 3, H, W) into spatio-temporal visual embeddings (B, T, d).
    """

    def __init__(
        self,
        backbone_type: str = "resnet18",
        pretrained: bool = False,
        feature_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.backbone_type = backbone_type

        if backbone_type == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            # Remove final classification FC
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            in_dim = resnet.fc.in_features
            self.projector = nn.Sequential(
                nn.Linear(in_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.backbone = LightweightVisualBackbone(out_dim=feature_dim)
            self.projector = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout)
            )

    def forward(self, rgb_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_seq: Tensor of shape (B, T, C, H, W)
        Returns:
            f_rgb: Tensor of shape (B, T, feature_dim)
        """
        B, T, C, H, W = rgb_seq.shape
        # Collapse B and T to process through 2D CNN
        x = rgb_seq.view(B * T, C, H, W)

        if self.backbone_type == "resnet18":
            feat = self.backbone(x)  # (B*T, 512, 1, 1)
            feat = feat.view(B * T, -1)
            out = self.projector(feat)  # (B*T, feature_dim)
        else:
            feat = self.backbone(x)
            out = self.projector(feat)

        return out.view(B, T, self.feature_dim)
