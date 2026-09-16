"""
X-MIST / EMIT-HAR: Explainable Multimodal Interaction-aware Spatio-Temporal Transformer.
Complete Proposed Research Architecture.
Integrates:
1. Multimodal Encoders (RGB, Pose, Trajectory, Scene)
2. Cross-Modal Attention (Trajectory <-> RGB, Trajectory <-> Pose, etc.)
3. Adaptive Modality Gating (Learned w_m weights)
4. Dynamic Interaction Graph (relational edge features: distance, speed diff, heading diff, TTC)
5. Dynamic Interaction Transformer (agent-to-agent relational reasoning)
6. Spatio-Temporal Sequence Transformer (temporal transitions)
7. Multi-Task Behavior Recognition Heads:
   - Head 1: Pedestrian Behavior (8 classes)
   - Head 2: Micromobility Behavior (9 classes)
   - Head 3: Interaction Behavior (6 classes)
8. Explainability Evidence Output (Temporal, Spatial, Modality, Interaction)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from models.rgb_encoder import RGBEncoder
from models.pose_encoder import PoseEncoder
from models.trajectory_encoder import TrajectoryEncoder
from models.scene_encoder import SceneEncoder
from models.cross_modal_attention import CrossModalFusion
from models.modality_fusion import ModalityGating
from models.interaction_graph import DynamicInteractionGraph
from models.interaction_transformer import DynamicInteractionTransformer
from models.temporal_transformer import TemporalTransformer
from heads.pedestrian_behavior import PedestrianBehaviorHead
from heads.micromobility_behavior import MicromobilityBehaviorHead
from heads.interaction import InteractionBehaviorHead
from datasets.taxonomy import PEDESTRIAN_CLASSES, MICROMOBILITY_CLASSES, INTERACTION_CLASSES


class ProposedXMISTModel(nn.Module):
    """
    Explainable Multimodal Interaction-aware Spatio-Temporal Transformer (X-MIST / EMIT-HAR).
    Fully modular to support research ablations A1 through A9.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        use_rgb: bool = True,
        use_pose: bool = True,
        use_traj: bool = True,
        use_scene: bool = True,
        use_cross_modal: bool = True,
        use_interaction: bool = True,
        use_modality_gating: bool = True,
        num_ped_classes: int = len(PEDESTRIAN_CLASSES),
        num_micro_classes: int = len(MICROMOBILITY_CLASSES),
        num_inter_classes: int = len(INTERACTION_CLASSES),
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_rgb = use_rgb
        self.use_pose = use_pose
        self.use_traj = use_traj
        self.use_scene = use_scene
        self.use_cross_modal = use_cross_modal
        self.use_interaction = use_interaction
        self.use_modality_gating = use_modality_gating

        # 1. Modality Encoders
        if use_rgb:
            self.rgb_encoder = RGBEncoder(feature_dim=feature_dim, dropout=dropout)
        if use_pose:
            self.pose_encoder = PoseEncoder(feature_dim=feature_dim, dropout=dropout)
        if use_traj:
            self.traj_encoder = TrajectoryEncoder(input_dim=5, feature_dim=feature_dim, dropout=dropout)
        if use_scene:
            self.scene_encoder = SceneEncoder(num_classes=10, feature_dim=feature_dim, dropout=dropout)

        # 2. Cross-Modal Attention
        if use_cross_modal:
            self.cross_modal_fusion = CrossModalFusion(feature_dim=feature_dim, dropout=dropout)

        # 3. Modality Gating & Adaptive Fusion
        if use_modality_gating:
            self.modality_gating = ModalityGating(feature_dim=feature_dim, num_modalities=4)
        else:
            self.simple_fusion = nn.Linear(feature_dim * 4, feature_dim)

        # 4. Dynamic Interaction Graph & Transformer
        if use_interaction:
            self.interaction_graph = DynamicInteractionGraph(edge_dim=4, edge_embed_dim=64, distance_threshold=12.0)
            self.interaction_transformer = DynamicInteractionTransformer(feature_dim=feature_dim, edge_embed_dim=64, dropout=dropout)

        # 5. Spatio-Temporal Sequence Transformer
        self.temporal_transformer = TemporalTransformer(feature_dim=feature_dim, num_layers=3, dropout=dropout)

        # 6. Multi-Task Behavior Heads
        self.ped_head = PedestrianBehaviorHead(feature_dim=feature_dim, num_classes=num_ped_classes, dropout=dropout)
        self.micro_head = MicromobilityBehaviorHead(feature_dim=feature_dim, num_classes=num_micro_classes, dropout=dropout)
        self.inter_head = InteractionBehaviorHead(feature_dim=feature_dim, num_classes=num_inter_classes, dropout=dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Full progressive forward pass returning predictions and XAI attribution artifacts.
        """
        device = next(self.parameters()).device
        B = batch["trajectory"].shape[0]
        T = batch["trajectory"].shape[1]

        # 1. Encode Modalities
        f_rgb = self.rgb_encoder(batch["rgb"]) if self.use_rgb else torch.zeros(B, T, self.feature_dim, device=device)
        f_pose = self.pose_encoder(batch["pose"]) if self.use_pose else torch.zeros(B, T, self.feature_dim, device=device)
        f_traj = self.traj_encoder(batch["trajectory"]) if self.use_traj else torch.zeros(B, T, self.feature_dim, device=device)
        f_scene = self.scene_encoder(batch["scene"]) if self.use_scene else torch.zeros(B, T, self.feature_dim, device=device)

        # 2. Cross-Modal Attention
        cross_attn_maps = {}
        if self.use_cross_modal and self.use_rgb and self.use_pose and self.use_traj and self.use_scene:
            enhanced_mods, cross_attn_maps = self.cross_modal_fusion(f_rgb, f_pose, f_traj, f_scene)
        else:
            enhanced_mods = {"rgb": f_rgb, "pose": f_pose, "traj": f_traj, "scene": f_scene}

        # 3. Modality Gating
        modality_weights = None
        if self.use_modality_gating:
            f_multi, modality_weights = self.modality_gating(enhanced_mods)
        else:
            cat = torch.cat([enhanced_mods["rgb"], enhanced_mods["pose"], enhanced_mods["traj"], enhanced_mods["scene"]], dim=-1)
            f_multi = self.simple_fusion(cat)

        # 4. Interaction Modeling
        interaction_weights = None
        f_interaction = None
        f_fused = f_multi

        if self.use_interaction and "neighbor_agents" in batch and "neighbor_mask" in batch:
            edge_feats, edge_embed, adj_mask = self.interaction_graph.compute_relational_edges(
                primary_traj=batch["trajectory"],
                neighbor_trajs=batch["neighbor_agents"],
                neighbor_mask=batch["neighbor_mask"]
            )
            f_fused, f_interaction, interaction_weights = self.interaction_transformer(
                f_multi=f_multi,
                neighbor_trajs=batch["neighbor_agents"],
                edge_embed=edge_embed,
                adj_mask=adj_mask
            )

        # 5. Spatio-Temporal Sequence Transformer
        z, temporal_attn = self.temporal_transformer(f_fused)

        # 6. Multi-task Heads
        ped_logits = self.ped_head(z)
        micro_logits = self.micro_head(z)
        inter_logits = self.inter_head(z)

        return {
            # Predictions
            "ped_logits": ped_logits,
            "micro_logits": micro_logits,
            "inter_logits": inter_logits,
            "latent_z": z,
            # XAI Attribution Artifacts (Phases 20-22)
            "temporal_attn": temporal_attn,             # (B, T, T) Temporal Evidence
            "modality_weights": modality_weights,       # (B, T, 4) Modality Evidence
            "interaction_weights": interaction_weights, # (B, T, N) Interaction Evidence
            "cross_attn_maps": cross_attn_maps,         # Cross-modal Evidence
            "f_multi": f_multi,
            "f_interaction": f_interaction
        }
