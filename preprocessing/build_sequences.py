"""
Temporal Window Construction.
Phase 8 of implementation plan:
Creates fixed-length temporal sequence windows (e.g. T=16, 32, 64 frames):
Each sample becomes:
- RGB[T, C, H, W]
- Pose[T, J, D]
- Trajectory[T, 5]
- Scene[T, K]
- Agents[T, N, 5] (neighboring interacting agents)
- Labels: pedestrian, micromobility, interaction
"""

import os
import numpy as np
from typing import List, Dict, Any, Generator, Optional
from datasets.taxonomy import PED_TO_IDX, MICRO_TO_IDX, INTER_TO_IDX


def slice_temporal_window(
    total_frames: int,
    window_size: int = 32,
    stride: int = 8
) -> List[slice]:
    """Generates slice objects for sliding temporal windows."""
    slices = []
    if total_frames < window_size:
        # Pad or single slice
        slices.append(slice(0, total_frames))
        return slices

    for start in range(0, total_frames - window_size + 1, stride):
        slices.append(slice(start, start + window_size))
    return slices


def pad_or_truncate_sequence(
    arr: np.ndarray,
    target_len: int,
    pad_mode: str = "edge"
) -> np.ndarray:
    """
    Pads or truncates leading dimension to match target_len.
    arr shape: (T, ...)
    """
    current_len = arr.shape[0]
    if current_len == target_len:
        return arr
    elif current_len > target_len:
        return arr[:target_len]
    else:
        # Need padding
        pad_width = [(0, target_len - current_len)] + [(0, 0)] * (arr.ndim - 1)
        if pad_mode == "edge" and current_len > 0:
            return np.pad(arr, pad_width, mode="edge")
        else:
            return np.pad(arr, pad_width, mode="constant")


def build_sample_dict(
    rgb_frames: np.ndarray,          # (T, 3, H, W)
    pose_seq: np.ndarray,            # (T, J, 3)
    trajectory: np.ndarray,          # (T, 5)
    scene_context: np.ndarray,       # (T, 10)
    neighbor_agents: np.ndarray,     # (T, N_max, 5)
    neighbor_mask: np.ndarray,       # (T, N_max) bool
    ped_label: int = 0,
    micro_label: int = 0,
    inter_label: int = 0,
    agent_type: str = "pedestrian",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Packs a processed sequence into a standardized dictionary.
    """
    return {
        "rgb": rgb_frames.astype(np.float32),
        "pose": pose_seq.astype(np.float32),
        "trajectory": trajectory.astype(np.float32),
        "scene": scene_context.astype(np.float32),
        "neighbor_agents": neighbor_agents.astype(np.float32),
        "neighbor_mask": neighbor_mask.astype(bool),
        "ped_label": int(ped_label),
        "micro_label": int(micro_label),
        "inter_label": int(inter_label),
        "agent_type": str(agent_type),
        "metadata": metadata or {}
    }
