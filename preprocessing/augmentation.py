"""
Multimodal Data Augmentation.

Coordinates are image-plane box centres normalised to [0, 1] (x / width, y / height), so only
augmentations that keep a plausible dashcam geometry are enabled by default:

1. Horizontal flip (default OFF; set prob_hflip for [0, 1]-normalised data):  x -> 1 - x, vx -> -vx, heading -> pi - heading
   (applied identically to the pedestrian and all valid neighbours).
2. Rotation (default OFF): rotating image coordinates about the image corner is not physically
   meaningful for a forward-facing camera. Kept only for synthetic / bird's-eye data. The rotation is
   now applied consistently to positions, velocities and heading (previously positions/velocities
   were rotated by -theta while heading was shifted by +theta).
3. Velocity scaling (default OFF): scaling velocities without rescaling positions produces
   physically inconsistent tracks.
4. Joint masking / modality dropout: only act on modalities present in the batch.
"""

import math
import torch
import numpy as np
from typing import Dict, Any


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class MultimodalAugmentor:

    def __init__(
        self,
        prob_hflip: float = 0.0,
        prob_rotation: float = 0.0,
        prob_speed_scale: float = 0.0,
        prob_joint_mask: float = 0.3,
        prob_modality_dropout: float = 0.0,
    ):
        self.prob_hflip = prob_hflip
        self.prob_rotation = prob_rotation
        self.prob_speed_scale = prob_speed_scale
        self.prob_joint_mask = prob_joint_mask
        self.prob_modality_dropout = prob_modality_dropout

    @staticmethod
    def hflip_tracks(traj: torch.Tensor) -> torch.Tensor:
        """traj (..., 5) [x, y, vx, vy, heading] in normalised image coords."""
        out = traj.clone()
        out[..., 0] = 1.0 - traj[..., 0]
        out[..., 2] = -traj[..., 2]
        out[..., 4] = _wrap(math.pi - traj[..., 4])
        return out

    @staticmethod
    def rotate_tracks(traj: torch.Tensor, angle: float) -> torch.Tensor:
        """Counter-clockwise rotation by `angle` applied consistently to position, velocity and heading."""
        c, s = math.cos(angle), math.sin(angle)
        out = traj.clone()
        x, y, vx, vy = traj[..., 0], traj[..., 1], traj[..., 2], traj[..., 3]
        out[..., 0] = c * x - s * y
        out[..., 1] = s * x + c * y
        out[..., 2] = c * vx - s * vy
        out[..., 3] = s * vx + c * vy
        out[..., 4] = _wrap(traj[..., 4] + angle)
        return out

    def augment_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        aug = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        B = aug["trajectory"].shape[0]
        has_nb = "neighbor_agents" in aug and "neighbor_mask" in aug

        # 1. Per-sample horizontal flip
        if self.prob_hflip > 0:
            flip = torch.from_numpy(np.random.rand(B) < self.prob_hflip).to(aug["trajectory"].device)
            if flip.any():
                aug["trajectory"][flip] = self.hflip_tracks(aug["trajectory"][flip])
                if has_nb:
                    nb = aug["neighbor_agents"][flip]
                    nb_flipped = self.hflip_tracks(nb)
                    valid = aug["neighbor_mask"][flip].unsqueeze(-1)
                    aug["neighbor_agents"][flip] = torch.where(valid, nb_flipped, nb)  # keep padding at zero
                if "bbox" in aug:
                    bb = aug["bbox"][flip]
                    bb[..., 0] = 1.0 - bb[..., 0] - bb[..., 2]
                    aug["bbox"][flip] = bb

        # 2. Rotation (batch-wide angle), off by default
        if self.prob_rotation > 0 and np.random.rand() < self.prob_rotation:
            angle = float(np.random.uniform(-math.pi, math.pi))
            aug["trajectory"] = self.rotate_tracks(aug["trajectory"], angle)
            if has_nb:
                rot = self.rotate_tracks(aug["neighbor_agents"], angle)
                aug["neighbor_agents"] = torch.where(aug["neighbor_mask"].unsqueeze(-1), rot, aug["neighbor_agents"])

        # 3. Velocity scaling, off by default
        if self.prob_speed_scale > 0 and np.random.rand() < self.prob_speed_scale:
            scale = float(np.random.uniform(0.85, 1.15))
            aug["trajectory"][..., 2:4] *= scale
            if has_nb:
                aug["neighbor_agents"][..., 2:4] *= scale

        # 4. Joint masking (only if real pose present)
        if "pose" in aug and self.prob_joint_mask > 0 and np.random.rand() < self.prob_joint_mask:
            num_joints = aug["pose"].shape[2]
            masked = np.random.choice(num_joints, size=min(2, num_joints), replace=False)
            aug["pose"][:, :, masked, :] = 0.0

        # 5. Modality dropout (only among modalities present)
        if self.prob_modality_dropout > 0 and np.random.rand() < self.prob_modality_dropout:
            present = [m for m in ["rgb", "pose", "scene"] if m in aug]
            if present:
                aug[np.random.choice(present)].zero_()

        return aug
