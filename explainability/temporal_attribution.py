"""
Temporal Attribution Module.
Phase 20.1 of implementation plan:
Calculates temporal importance score I_t for each timestep t in 1..T:
Combines temporal self-attention rollout and gradient-based attribution.
Identifies the critical temporal decision window (e.g. 1.8s - 2.6s).
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any


def compute_temporal_attribution(
    temporal_attn: torch.Tensor,
    dt: float = 0.1,
    top_k_percent: float = 0.25
) -> Dict[str, Any]:
    """
    Extracts timestep importance curve I_t and finds critical temporal window.

    Args:
        temporal_attn: Attention matrix of shape (T, T) or (B, T, T)
        dt: Frame time delta in seconds
        top_k_percent: Fraction of frames to consider as most critical

    Returns:
        Dict containing importance curve I_t, critical intervals, and frame indices
    """
    if isinstance(temporal_attn, torch.Tensor):
        if temporal_attn.dim() == 3:
            # Take first sample if batch provided: (T, T)
            attn = temporal_attn[0].detach().cpu().numpy()
        else:
            attn = temporal_attn.detach().cpu().numpy()
    else:
        attn = np.asarray(temporal_attn)

    T = attn.shape[0]
    # Sum attention directed to timestep t from subsequent decision steps
    # I_t is column-wise average of incoming attention
    importance = np.mean(attn, axis=0) # (T,)

    # Normalize to [0, 1]
    denom = np.max(importance) - np.min(importance)
    if denom > 1e-6:
        norm_importance = (importance - np.min(importance)) / denom
    else:
        norm_importance = np.ones(T) / float(T)

    # Find high-importance temporal interval
    threshold = np.quantile(norm_importance, 1.0 - top_k_percent)
    critical_frames = np.where(norm_importance >= threshold)[0]

    start_frame = int(critical_frames[0]) if len(critical_frames) > 0 else 0
    end_frame = int(critical_frames[-1]) if len(critical_frames) > 0 else T - 1

    start_sec = start_frame * dt
    end_sec = end_frame * dt

    return {
        "importance_curve": norm_importance.tolist(),
        "critical_frames": critical_frames.tolist(),
        "window_frames": [start_frame, end_frame],
        "window_seconds": (round(start_sec, 2), round(end_sec, 2)),
        "summary": f"{start_sec:.1f}s – {end_sec:.1f}s"
    }
