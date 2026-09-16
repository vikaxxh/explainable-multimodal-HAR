"""
Interaction Dataset Adapter.
Focuses on agent pairs:
- Pedestrian-Pedestrian
- Pedestrian-Bicycle
- Pedestrian-E-scooter
- Pedestrian-Vehicle
- Micromobility-Vehicle
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from datasets.taxonomy import INTERACTION_CLASSES, INTER_TO_IDX


class AgentInteractionDataset(Dataset):
    """
    Dataset wrapper focusing on pairwise interaction dynamics,
    relative kinematics, and interaction classification.
    """

    def __init__(self, samples: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
