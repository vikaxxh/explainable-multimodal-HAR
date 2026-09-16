"""
Pedestrian Dataset Adapter for Benchmark Datasets (JAAD, PIE, TITAN).
Provides single-modality and paired pedestrian trajectory representations.
"""

import os
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from datasets.taxonomy import PEDESTRIAN_CLASSES, PED_TO_IDX


class PedestrianBenchmarkDataset(Dataset):
    """
    Adapter for JAAD / PIE pedestrian intention and action benchmarks.
    Extracts trajectory sequences, bounding boxes, and intention/action labels.
    """

    def __init__(
        self,
        annotations_file: Optional[str] = None,
        window_size: int = 32,
        dt: float = 0.1
    ):
        super().__init__()
        self.window_size = window_size
        self.dt = dt
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
