"""
Scene and Context Representation.
Phase 7 of implementation plan:
Extracts semantic context:
- Road, Sidewalk, Crosswalk, Traffic signal, Buildings,
  Obstacles, Vehicles, Cycling lane, Pedestrian area, Background.

Provides:
1. Global scene semantic distribution vector (K-dim)
2. Local agent contextual patch distribution (K-dim under agent bbox)
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from datasets.taxonomy import SCENE_CLASSES, SCENE_TO_IDX


class SceneContextExtractor:
    """
    Extracts semantic scene context distributions for temporal sequences.
    """

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.class_names = SCENE_CLASSES

    def extract_local_context(
        self,
        seg_mask: np.ndarray,
        bbox: List[float]
    ) -> np.ndarray:
        """
        Extracts semantic category distribution within and immediately beneath an agent's bounding box.

        Args:
            seg_mask: (H, W) array with integer class IDs [0..num_classes-1]
            bbox: [x, y, w, h]

        Returns:
            dist: Normalized class distribution vector of shape (num_classes,)
        """
        H, W = seg_mask.shape
        x, y, w, h = bbox
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(W, int(x + w))
        y2 = min(H, int(y + h))

        if x2 <= x1 or y2 <= y1:
            # Fallback uniform / background
            dist = np.zeros(self.num_classes, dtype=np.float32)
            dist[SCENE_TO_IDX.get("background", 9)] = 1.0
            return dist

        crop = seg_mask[y1:y2, x1:x2]
        counts = np.bincount(crop.flatten(), minlength=self.num_classes)[:self.num_classes]
        total = float(np.sum(counts))
        if total > 0:
            dist = counts.astype(np.float32) / total
        else:
            dist = np.zeros(self.num_classes, dtype=np.float32)
            dist[SCENE_TO_IDX.get("background", 9)] = 1.0

        return dist

    def extract_sequence_context(
        self,
        seg_masks: List[np.ndarray],
        boxes: List[List[float]]
    ) -> np.ndarray:
        """
        Extracts local context across a temporal sequence of T frames.

        Args:
            seg_masks: List of T segmentation masks each (H, W)
            boxes: List of T bounding boxes [x, y, w, h]

        Returns:
            seq_context: Array of shape (T, num_classes)
        """
        T = len(boxes)
        seq_context = np.zeros((T, self.num_classes), dtype=np.float32)

        for t in range(T):
            if t < len(seg_masks) and seg_masks[t] is not None:
                seq_context[t] = self.extract_local_context(seg_masks[t], boxes[t])
            else:
                # Default prior: road / sidewalk distribution
                seq_context[t, SCENE_TO_IDX.get("sidewalk", 1)] = 0.6
                seq_context[t, SCENE_TO_IDX.get("road", 0)] = 0.4

        return seq_context
