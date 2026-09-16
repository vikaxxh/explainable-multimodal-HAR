"""
Trajectory and Multi-Agent Interaction Spatial Visualizer.
Plots:
1. Primary agent trajectory curve in 2D with heading vectors.
2. Interacting neighbor trajectories and proximity radii.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_agent_trajectories(
    primary_traj: np.ndarray,          # (T, 5) [x, y, vx, vy, heading]
    neighbor_trajs: np.ndarray,        # (T, N, 5)
    save_path: Optional[str] = None,
    title: str = "Multi-Agent Spatial Trajectories"
):
    """
    Plots primary agent and neighbor paths in the 2D coordinate plane.
    """
    plt.figure(figsize=(7, 6))

    # Primary agent
    px = primary_traj[:, 0]
    py = primary_traj[:, 1]
    plt.plot(px, py, "b-o", label="Primary Pedestrian", linewidth=2.5, markersize=4)
    plt.scatter([px[0]], [py[0]], color="blue", marker="s", s=80, label="Pedestrian Start")
    plt.scatter([px[-1]], [py[-1]], color="darkblue", marker="*", s=120, label="Pedestrian End")

    # Neighbor agents
    if neighbor_trajs is not None and neighbor_trajs.ndim == 3:
        T, N, _ = neighbor_trajs.shape
        colors = ["red", "green", "orange", "purple"]
        for k in range(N):
            nx = neighbor_trajs[:, k, 0]
            ny = neighbor_trajs[:, k, 1]
            if np.any(nx != 0) or np.any(ny != 0):
                c = colors[k % len(colors)]
                plt.plot(nx, ny, "--", color=c, label=f"Neighbor {k+1} Path", alpha=0.8)
                plt.scatter([nx[-1]], [ny[-1]], color=c, marker="^", s=80)

    plt.xlabel("X Coordinate (m)", fontsize=10)
    plt.ylabel("Y Coordinate (m)", fontsize=10)
    plt.title(title, fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()
