"""
Trajectory and Kinematics Extraction.
Phase 5 of implementation plan:
Calculates:
- Position: p_t = (x_t, y_t)
- Velocity: v_t = (p_t - p_{t-1}) / dt
- Acceleration: a_t = (v_t - v_{t-1}) / dt
- Heading: theta_t = atan2(y_t - y_{t-1}, x_t - x_{t-1})
"""

import numpy as np
from typing import List, Dict, Any


def compute_kinematics(positions: np.ndarray, dt: float = 0.1) -> np.ndarray:
    """
    Computes kinematics (position, velocity, acceleration, heading) from 2D coordinates.

    Args:
        positions: Array of shape (T, 2) containing [x_t, y_t]
        dt: Time delta between frames in seconds (e.g. 0.1 for 10 FPS)

    Returns:
        features: Array of shape (T, 5) containing [x_t, y_t, vx_t, vy_t, heading_t]
    """
    T = len(positions)
    if T == 0:
        return np.zeros((0, 5), dtype=np.float32)

    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    headings = np.zeros((T, 1), dtype=np.float32)

    if T > 1:
        # First order forward/central difference for velocity
        diff = np.diff(positions, axis=0) / dt
        velocities[1:] = diff
        velocities[0] = diff[0]  # replicate initial velocity

        # Heading angle in radians [-pi, pi]
        dx = diff[:, 0]
        dy = diff[:, 1]
        headings[1:, 0] = np.arctan2(dy, dx)
        headings[0, 0] = headings[1, 0]

        # Acceleration
        acc_diff = np.diff(velocities, axis=0) / dt
        accelerations[1:] = acc_diff
        accelerations[0] = acc_diff[0]

    features = np.hstack([
        positions,               # (T, 2)
        velocities,              # (T, 2)
        headings                 # (T, 1)
    ])  # Total dimension = 5

    return features.astype(np.float32)


def compute_edge_features(
    traj_i: np.ndarray,
    traj_j: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Computes dynamic pairwise interaction edge features between two agents i and j over time.

    Edge feature vector e_ij(t) = [d_ij, delta_v_ij, delta_theta_ij, TTC_ij]
    where:
    - d_ij: Euclidean distance ||p_i - p_j||
    - delta_v_ij: Relative speed ||v_i - v_j||
    - delta_theta_ij: Angular difference in headings
    - TTC_ij: Estimated Time-to-Collision d_ij / (closing_speed + eps)

    Args:
        traj_i: (T, 5) array for agent i [x, y, vx, vy, heading]
        traj_j: (T, 5) array for agent j [x, y, vx, vy, heading]

    Returns:
        edge_feats: (T, 4) array
    """
    pos_i, pos_j = traj_i[:, :2], traj_j[:, :2]
    vel_i, vel_j = traj_i[:, 2:4], traj_j[:, 2:4]
    head_i, head_j = traj_i[:, 4], traj_j[:, 4]

    # Distance
    rel_pos = pos_i - pos_j
    dist = np.linalg.norm(rel_pos, axis=1, keepdims=True)  # (T, 1)

    # Relative speed
    rel_vel = vel_i - vel_j
    speed_diff = np.linalg.norm(rel_vel, axis=1, keepdims=True)  # (T, 1)

    # Heading difference wrapped to [-pi, pi]
    heading_diff = np.abs(head_i - head_j) % (2 * np.pi)
    heading_diff = np.minimum(heading_diff, 2 * np.pi - heading_diff)[:, np.newaxis]

    # Closing velocity along the line-of-sight
    unit_rel_pos = rel_pos / (dist + eps)
    closing_speed = -np.sum(rel_vel * unit_rel_pos, axis=1, keepdims=True)
    closing_speed = np.maximum(closing_speed, 0.0)

    # Time To Collision (TTC) clipped to [0, 10] seconds for numerical stability
    ttc = np.where(closing_speed > 0.1, dist / (closing_speed + eps), 10.0)
    ttc = np.clip(ttc, 0.0, 10.0)

    edge_feats = np.hstack([dist, speed_diff, heading_diff, ttc])
    return edge_feats.astype(np.float32)
