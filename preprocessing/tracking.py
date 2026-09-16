"""
Multi-Object Tracking and Trajectory Aggregation.
Phase 5 of implementation plan:
Connects detections across frames into coherent agent trajectories:
T_i = {(x_t, y_t, v_t, a_t, theta_t)}_{t=1}^T
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from preprocessing.detection import AgentDetection
from preprocessing.trajectory import compute_kinematics


class MultiObjectTracker:
    """
    Connects detections across frames into tracklets and extracts kinematics.
    """

    def __init__(self, iou_threshold: float = 0.3, dt: float = 0.1):
        self.iou_threshold = iou_threshold
        self.dt = dt

    @staticmethod
    def bbox_iou(boxA: List[float], boxB: List[float]) -> float:
        """Computes IoU between [x, y, w, h] boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def group_by_agent_id(self, detections: List[AgentDetection]) -> Dict[int, List[AgentDetection]]:
        """
        Groups already-associated detections by agent_id sorted by frame_id.
        """
        agent_tracks: Dict[int, List[AgentDetection]] = {}
        for det in detections:
            if det.agent_id not in agent_tracks:
                agent_tracks[det.agent_id] = []
            agent_tracks[det.agent_id].append(det)

        # Sort each track chronologically
        for agent_id in agent_tracks:
            agent_tracks[agent_id].sort(key=lambda d: d.frame_id)

        return agent_tracks

    def extract_trajectory(self, track: List[AgentDetection]) -> Dict[str, Any]:
        """
        Takes a sequence of detections for one agent, extracts centers,
        and computes [x, y, vx, vy, heading] kinematic features.
        """
        if not track:
            return {"kinematics": np.zeros((0, 5), dtype=np.float32), "frames": []}

        centers = np.array([d.center for d in track], dtype=np.float32)
        frames = [d.frame_id for d in track]
        agent_type = track[0].agent_type
        agent_id = track[0].agent_id

        kinematics = compute_kinematics(centers, dt=self.dt)

        return {
            "agent_id": agent_id,
            "agent_type": agent_type,
            "frames": frames,
            "boxes": [d.bbox for d in track],
            "kinematics": kinematics  # (T, 5)
        }
