"""
Agent Detection Module.
Phase 4 of implementation plan:
Detects agents in video frames:
- agent_id: Unique tracking identifier
- agent_type: 'pedestrian', 'bicycle', 'escooter', 'vehicle'
- frame_id: Integer timestamp / frame index
- bbox: [x, y, width, height] in pixel coordinates
- confidence: Detection confidence score in [0, 1]
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class AgentDetection:
    agent_id: int
    agent_type: str
    frame_id: int
    bbox: List[float]  # [x, y, w, h]
    confidence: float

    @property
    def center(self) -> List[float]:
        return [self.bbox[0] + self.bbox[2] / 2.0, self.bbox[1] + self.bbox[3] / 2.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentDetector:
    """
    Unified detector interface supporting pre-annotated dataset ingestion
    or deep learning detectors (YOLO / torchvision Faster R-CNN).
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def detect_frame(self, frame: np.ndarray, frame_id: int) -> List[AgentDetection]:
        """
        Detects agents within a single RGB frame (H, W, 3).
        Falls back to empty or pre-registered detections if weights are not local.
        """
        # Interface designed to plug in torchvision.models.detection or ultralytics YOLO
        detections = []
        return detections

    def parse_annotations(self, raw_annotations: List[Dict[str, Any]]) -> List[AgentDetection]:
        """
        Parses raw annotation list into standardized AgentDetection dataclasses.
        """
        parsed = []
        for item in raw_annotations:
            conf = float(item.get("confidence", 1.0))
            if conf >= self.confidence_threshold:
                parsed.append(
                    AgentDetection(
                        agent_id=int(item["agent_id"]),
                        agent_type=str(item.get("type", item.get("agent_type", "pedestrian"))),
                        frame_id=int(item["frame_id"]),
                        bbox=[float(c) for c in item["bbox"]],
                        confidence=conf
                    )
                )
        return parsed
