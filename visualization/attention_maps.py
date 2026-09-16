"""
Attention Maps and Modality Attribution Visualizer.
Plots:
1. Temporal Self-Attention Matrix Heatmap (T x T)
2. Modality Percentage Contribution Bar / Donut Chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def plot_temporal_attention(
    attn_matrix: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Temporal Attention Heatmap"
):
    """
    Plots (T, T) temporal attention map.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.title(title, fontsize=12)
    plt.xlabel("Key Timestep (t)", fontsize=10)
    plt.ylabel("Query Timestep (t)", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_modality_breakdown(
    modality_pcts: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Modality Attribution Breakdown"
):
    """
    Plots bar chart of modality contributions.
    """
    names = list(modality_pcts.keys())
    values = list(modality_pcts.values())

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, values, color=["#3498db", "#2ecc71", "#e74c3c", "#f39c12"])
    plt.ylabel("Contribution (%)", fontsize=10)
    plt.title(title, fontsize=12)
    plt.ylim(0, max(values) + 15)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()
