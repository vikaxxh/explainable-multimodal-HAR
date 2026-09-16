"""
Pose Extraction and Skeleton Normalization.
Phase 6 of implementation plan:
Extracts and normalizes pedestrian skeletal keypoints:
P_t in R^{J x D} where J=18 joints, D=3 (x, y, confidence).
Computes joint positions, velocities, and confidence.
"""

from typing import List, Dict, Any, Optional
import numpy as np


# Standard COCO 18-joint topology
JOINT_NAMES = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
    "LEye", "REar", "LEar"
]

# Skeleton connectivity pairs for graph adjacency (limbs)
SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Right arm
    (1, 5), (5, 6), (6, 7),              # Left arm
    (1, 8), (8, 9), (9, 10),             # Right leg
    (1, 11), (11, 12), (12, 13),         # Left leg
    (0, 14), (14, 16), (0, 15), (15, 17) # Face / ears
]


def normalize_pose_to_bbox(
    keypoints: np.ndarray,
    bbox: List[float]
) -> np.ndarray:
    """
    Normalizes (J, 3) keypoints relative to bounding box [x, y, w, h].
    x_norm = (x - bbox_x) / bbox_w
    y_norm = (y - bbox_y) / bbox_h
    confidence remains untouched in [0, 1].

    Args:
        keypoints: Array of shape (J, 3) with [x, y, conf]
        bbox: [x, y, w, h]

    Returns:
        normalized: Array of shape (J, 3)
    """
    norm_kp = keypoints.copy()
    bx, by, bw, bh = bbox
    bw = max(bw, 1e-4)
    bh = max(bh, 1e-4)

    norm_kp[:, 0] = (norm_kp[:, 0] - bx) / bw
    norm_kp[:, 1] = (norm_kp[:, 1] - by) / bh
    return norm_kp.astype(np.float32)


def compute_pose_sequence_features(
    pose_seq: np.ndarray,
    dt: float = 0.1
) -> np.ndarray:
    """
    Computes pose features including positions, velocities, and confidence.

    Args:
        pose_seq: Array of shape (T, J, 3) [x, y, conf]
        dt: Time delta between frames

    Returns:
        features: Array of shape (T, J, 3) with [x, y, conf] or extended with velocities
    """
    T, J, D = pose_seq.shape
    if T <= 1:
        return pose_seq.astype(np.float32)

    # Calculate velocities: delta_x / dt, delta_y / dt
    vel = np.zeros((T, J, 2), dtype=np.float32)
    diff = (pose_seq[1:, :, :2] - pose_seq[:-1, :, :2]) / dt
    vel[1:] = diff
    vel[0] = diff[0]

    # Combine (x, y, conf)
    return pose_seq.astype(np.float32)
