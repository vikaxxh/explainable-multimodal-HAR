"""
Baseline Models for Benchmark Comparison and Reproduction.
Phases 9 & 10 of implementation plan:
1. RGBBaselineModel: Single-modality RGB -> Visual Encoder -> Temporal Transformer -> Classifier
2. TrajectoryBaselineModel: Single-modality Trajectory -> Motion Encoder -> Temporal Transformer -> Classifier
3. PoseBaselineModel: Single-modality Pose -> Pose Encoder -> Temporal Transformer -> Classifier
4. IntentFormerReproductionModel: RGB + Trajectory + Scene -> Multimodal Fusion -> Temporal Transformer -> Classifier
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from models.rgb_encoder import RGBEncoder
from models.trajectory_encoder import TrajectoryEncoder
from models.pose_encoder import PoseEncoder
from models.scene_encoder import SceneEncoder
from models.temporal_transformer import TemporalTransformer
from heads.pedestrian_behavior import PedestrianBehaviorHead
from datasets.taxonomy import PEDESTRIAN_CLASSES


class RGBBaselineModel(nn.Module):
    """Phase 9: Baseline 1 - Single-modality RGB Video Classifier."""
    def __init__(self, feature_dim: int = 256, num_classes: int = len(PEDESTRIAN_CLASSES), dropout: float = 0.1):
        super().__init__()
        self.rgb_encoder = RGBEncoder(feature_dim=feature_dim, dropout=dropout)
        self.temporal_transformer = TemporalTransformer(feature_dim=feature_dim, num_layers=2, dropout=dropout)
        self.head = PedestrianBehaviorHead(feature_dim=feature_dim, num_classes=num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f_rgb = self.rgb_encoder(batch["rgb"])
        z, attn = self.temporal_transformer(f_rgb)
        logits = self.head(z)
        return {"ped_logits": logits, "temporal_attn": attn, "features": z}


class TrajectoryBaselineModel(nn.Module):
    """Phase 9: Baseline 2 - Single-modality Trajectory Kinematics Classifier."""
    def __init__(self, feature_dim: int = 256, num_classes: int = len(PEDESTRIAN_CLASSES), dropout: float = 0.1):
        super().__init__()
        self.traj_encoder = TrajectoryEncoder(input_dim=5, feature_dim=feature_dim, dropout=dropout)
        self.temporal_transformer = TemporalTransformer(feature_dim=feature_dim, num_layers=2, dropout=dropout)
        self.head = PedestrianBehaviorHead(feature_dim=feature_dim, num_classes=num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f_traj = self.traj_encoder(batch["trajectory"])
        z, attn = self.temporal_transformer(f_traj)
        logits = self.head(z)
        return {"ped_logits": logits, "temporal_attn": attn, "features": z}


class PoseBaselineModel(nn.Module):
    """Phase 9: Baseline 3 - Single-modality Skeletal Pose Keypoints Classifier."""
    def __init__(self, feature_dim: int = 256, num_classes: int = len(PEDESTRIAN_CLASSES), dropout: float = 0.1):
        super().__init__()
        self.pose_encoder = PoseEncoder(feature_dim=feature_dim, dropout=dropout)
        self.temporal_transformer = TemporalTransformer(feature_dim=feature_dim, num_layers=2, dropout=dropout)
        self.head = PedestrianBehaviorHead(feature_dim=feature_dim, num_classes=num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f_pose = self.pose_encoder(batch["pose"])
        z, attn = self.temporal_transformer(f_pose)
        logits = self.head(z)
        return {"ped_logits": logits, "temporal_attn": attn, "features": z}


class IntentFormerReproductionModel(nn.Module):
    """
    Phase 10: IntentFormer-style Base Paper Architecture.
    RGB + Trajectory + Scene -> Concatenation Projection -> Temporal Transformer -> Classifier.
    """
    def __init__(self, feature_dim: int = 256, num_classes: int = len(PEDESTRIAN_CLASSES), dropout: float = 0.1):
        super().__init__()
        self.rgb_encoder = RGBEncoder(feature_dim=feature_dim, dropout=dropout)
        self.traj_encoder = TrajectoryEncoder(input_dim=5, feature_dim=feature_dim, dropout=dropout)
        self.scene_encoder = SceneEncoder(num_classes=10, feature_dim=feature_dim, dropout=dropout)

        # Early concatenation and projection
        self.fusion_projector = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.temporal_transformer = TemporalTransformer(feature_dim=feature_dim, num_layers=3, dropout=dropout)
        self.head = PedestrianBehaviorHead(feature_dim=feature_dim, num_classes=num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f_rgb = self.rgb_encoder(batch["rgb"])
        f_traj = self.traj_encoder(batch["trajectory"])
        f_scene = self.scene_encoder(batch["scene"])

        concat = torch.cat([f_rgb, f_traj, f_scene], dim=-1)
        f_fused = self.fusion_projector(concat)

        z, attn = self.temporal_transformer(f_fused)
        logits = self.head(z)
        return {"ped_logits": logits, "temporal_attn": attn, "features": z}
