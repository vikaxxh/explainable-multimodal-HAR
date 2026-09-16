"""
Synthetic Dataset Generator for Rapid HPC Verification and Unit Testing.
Generates realistic multi-agent sequences with:
- RGB video crops: (T, 3, 224, 224)
- Skeleton keypoints: (T, 18, 3)
- Trajectories: (T, 5) [x, y, vx, vy, heading]
- Scene semantic distributions: (T, 10)
- Interacting neighbor agents: (T, N_max, 5)
- Ground-truth multi-task labels (Pedestrian, Micromobility, Interaction)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, Optional
from datasets.taxonomy import (
    PEDESTRIAN_CLASSES,
    MICROMOBILITY_CLASSES,
    INTERACTION_CLASSES,
    AGENT_TYPES,
    SCENE_CLASSES
)


class SyntheticMultimodalDataset(Dataset):
    """
    In-memory or disk-cacheable synthetic dataset for quick benchmarking,
    ablation dry-runs, and multi-GPU DDP validation.
    """

    def __init__(
        self,
        num_samples: int = 128,
        window_size: int = 32,
        num_joints: int = 18,
        max_agents: int = 4,
        image_size: tuple = (224, 224),
        seed: int = 42
    ):
        super().__init__()
        self.num_samples = num_samples
        self.window_size = window_size
        self.num_joints = num_joints
        self.max_agents = max_agents
        self.image_size = image_size
        self.rng = np.random.RandomState(seed)

        # Pre-generate lightweight synthetic samples
        self.samples = []
        for i in range(num_samples):
            self.samples.append(self._generate_single_sample(i))

    def _generate_single_sample(self, idx: int) -> Dict[str, Any]:
        T = self.window_size
        H, W = self.image_size
        N = self.max_agents

        # Multi-task ground truth labels
        ped_label = self.rng.randint(0, len(PEDESTRIAN_CLASSES))
        micro_label = self.rng.randint(0, len(MICROMOBILITY_CLASSES))
        inter_label = self.rng.randint(0, len(INTERACTION_CLASSES))

        # 1. Primary agent trajectory: (T, 5) [x, y, vx, vy, heading]
        t_steps = np.linspace(0, 3.2, T, dtype=np.float32)
        base_speed = 1.2 + 0.3 * self.rng.randn()
        angle = self.rng.uniform(-np.pi, np.pi)
        vx = float(base_speed * np.cos(angle))
        vy = float(base_speed * np.sin(angle))
        x = np.cumsum(np.full(T, vx * 0.1, dtype=np.float32)) + self.rng.uniform(10, 50)
        y = np.cumsum(np.full(T, vy * 0.1, dtype=np.float32)) + self.rng.uniform(10, 50)
        heading = np.full(T, angle, dtype=np.float32)
        trajectory = np.stack([x, y, np.full(T, vx), np.full(T, vy), heading], axis=1).astype(np.float32)

        # 2. Skeletal pose keypoints: (T, 18, 3) [x_norm, y_norm, confidence]
        # Simulate oscillating walking/standing joints
        pose = np.zeros((T, self.num_joints, 3), dtype=np.float32)
        cycle = np.sin(2 * np.pi * 1.5 * t_steps)
        for j in range(self.num_joints):
            # normalized around center (0.5, 0.5)
            jx = 0.5 + 0.1 * np.sin(j) + 0.05 * cycle * (1.0 if j % 2 == 0 else -1.0)
            jy = 0.2 + 0.04 * j
            conf = 0.85 + 0.1 * self.rng.rand(T)
            pose[:, j, 0] = np.clip(jx, 0.0, 1.0)
            pose[:, j, 1] = np.clip(jy, 0.0, 1.0)
            pose[:, j, 2] = conf

        # 3. Scene semantic context: (T, 10)
        # Distribution over 10 classes
        scene = np.zeros((T, len(SCENE_CLASSES)), dtype=np.float32)
        scene[:, 1] = 0.6  # sidewalk
        scene[:, 0] = 0.3  # road
        scene[:, 2] = 0.1  # crosswalk
        noise = self.rng.dirichlet(np.ones(len(SCENE_CLASSES)), size=T).astype(np.float32) * 0.1
        scene = scene + noise
        scene = scene / np.sum(scene, axis=1, keepdims=True)

        # 4. Interacting neighbor agents: (T, N_max, 5)
        # Include an approaching e-scooter or bicycle
        neighbor_agents = np.zeros((T, N, 5), dtype=np.float32)
        neighbor_mask = np.zeros((T, N), dtype=bool)

        num_active = self.rng.randint(1, N + 1)
        for k in range(num_active):
            offset_dist = self.rng.uniform(2.0, 8.0)
            approach_speed = 3.5  # scooter / bike speed
            nx = x + offset_dist - t_steps * approach_speed * 0.5
            ny = y + offset_dist * 0.5
            nvx = np.full(T, -approach_speed * 0.5, dtype=np.float32)
            nvy = np.zeros(T, dtype=np.float32)
            nhead = np.full(T, np.pi, dtype=np.float32)
            neighbor_agents[:, k, :] = np.stack([nx, ny, nvx, nvy, nhead], axis=1)
            neighbor_mask[:, k] = True

        # 5. RGB video tensor: (T, 3, H, W)
        # Lightweight representation for training without overloading RAM during tests
        rgb = np.zeros((T, 3, H, W), dtype=np.float32)
        # Create subtle gradient pattern to have non-zero gradients
        grid_y, grid_x = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
        for c in range(3):
            rgb[:, c, :, :] = (grid_y * (c + 1) / 3.0).astype(np.float32)

        return {
            "rgb": rgb,
            "pose": pose,
            "trajectory": trajectory,
            "scene": scene,
            "neighbor_agents": neighbor_agents,
            "neighbor_mask": neighbor_mask,
            "ped_label": ped_label,
            "micro_label": micro_label,
            "inter_label": inter_label,
            "agent_type": "pedestrian"
        }

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            "rgb": torch.from_numpy(sample["rgb"]),
            "pose": torch.from_numpy(sample["pose"]),
            "trajectory": torch.from_numpy(sample["trajectory"]),
            "scene": torch.from_numpy(sample["scene"]),
            "neighbor_agents": torch.from_numpy(sample["neighbor_agents"]),
            "neighbor_mask": torch.from_numpy(sample["neighbor_mask"]),
            "ped_label": torch.tensor(sample["ped_label"], dtype=torch.long),
            "micro_label": torch.tensor(sample["micro_label"], dtype=torch.long),
            "inter_label": torch.tensor(sample["inter_label"], dtype=torch.long)
        }
