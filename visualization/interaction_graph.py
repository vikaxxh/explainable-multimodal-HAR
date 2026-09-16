"""
Interaction Graph Topology Visualizer.
Plots the multi-agent graph G_t = (V_t, E_t) showing:
- Nodes (Pedestrians, Micromobility, Vehicles)
- Weighted edges representing interaction attention strengths.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_interaction_graph(
    node_positions: np.ndarray,         # (N_total, 2)
    node_labels: List[str],            # Names for each node
    edge_weights: np.ndarray,          # (N_total, N_total) interaction weights
    save_path: Optional[str] = None,
    title: str = "Dynamic Interaction Graph"
):
    """
    Renders graph visualization of agents and relational edges.
    """
    plt.figure(figsize=(6, 6))
    num_nodes = len(node_labels)

    # Draw nodes
    for i in range(num_nodes):
        x, y = node_positions[i]
        plt.scatter(x, y, s=400, color="#2980b9" if i == 0 else "#e67e22", zorder=3)
        plt.text(x, y + 0.3, node_labels[i], fontsize=10, ha="center", weight="bold")

    # Draw edges with thickness proportional to attention weight
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and edge_weights[i, j] > 0.05:
                w = edge_weights[i, j]
                plt.plot(
                    [node_positions[i, 0], node_positions[j, 0]],
                    [node_positions[i, 1], node_positions[j, 1]],
                    color="#2c3e50",
                    alpha=min(1.0, w * 2.0),
                    linewidth=max(1.0, w * 6.0),
                    zorder=2
                )

    plt.title(title, fontsize=12)
    plt.axis("off")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()
