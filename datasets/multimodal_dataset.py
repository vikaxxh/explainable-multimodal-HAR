"""
Multimodal Dataset Loader.
Supports loading preprocessed .npz / .pt sequence windows from disk:
- RGB video clips [T, C, H, W]
- Skeleton pose keypoints [T, J, D]
- Trajectory kinematics [T, 5]
- Semantic scene context [T, K]
- Interacting neighbor agents [T, N, 5]
- Multi-task targets: pedestrian, micromobility, interaction
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional
from datasets.synthetic_dataset import SyntheticMultimodalDataset


class MultimodalSequenceDataset(Dataset):
    """
    Loads saved multimodal sequence files from processed directory.
    Falls back gracefully to synthetic generation if directory is empty or during testing.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        window_size: int = 32,
        fallback_to_synthetic: bool = True,
        synthetic_samples: int = 128
    ):
        super().__init__()
        self.data_dir = os.path.join(data_dir, split) if os.path.exists(os.path.join(data_dir, split)) else data_dir
        self.split = split
        self.window_size = window_size

        # Find .npz or .pt sequence files
        self.files = []
        if os.path.exists(self.data_dir):
            candidates = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")) + glob.glob(os.path.join(self.data_dir, "*.pt")))
            # Ignore 0-byte or truncated files
            self.files = [f for f in candidates if os.path.getsize(f) > 500]

        self.synthetic_backup = None
        if len(self.files) == 0:
            if fallback_to_synthetic:
                print(f"[Dataset] Notice: No valid sequence files found in {self.data_dir}. Initializing synthetic dataset ({synthetic_samples} samples) for split '{split}'.")
                self.synthetic_backup = SyntheticMultimodalDataset(
                    num_samples=synthetic_samples,
                    window_size=window_size
                )
            else:
                raise FileNotFoundError(f"No processed data files found in {self.data_dir}")

    def __len__(self) -> int:
        if self.synthetic_backup is not None:
            return len(self.synthetic_backup)
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.synthetic_backup is not None:
            return self.synthetic_backup[idx]

        file_path = self.files[idx]
        try:
            if file_path.endswith(".npz"):
                with np.load(file_path, allow_pickle=True) as data:
                    return {
                        "rgb": torch.from_numpy(data["rgb"]).float(),
                        "pose": torch.from_numpy(data["pose"]).float(),
                        "trajectory": torch.from_numpy(data["trajectory"]).float(),
                        "scene": torch.from_numpy(data["scene"]).float(),
                        "neighbor_agents": torch.from_numpy(data["neighbor_agents"]).float(),
                        "neighbor_mask": torch.from_numpy(data["neighbor_mask"]).bool(),
                        "ped_label": torch.tensor(int(data["ped_label"]), dtype=torch.long),
                        "micro_label": torch.tensor(int(data["micro_label"]), dtype=torch.long),
                        "inter_label": torch.tensor(int(data["inter_label"]), dtype=torch.long)
                    }
            else:
                return torch.load(file_path)
        except Exception as e:
            # Graceful fallback to next sample if an individual file is corrupted
            alt_idx = (idx + 1) % len(self.files)
            if alt_idx == idx:
                raise e
            return self.__getitem__(alt_idx)


def collate_multimodal_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Standard batch collator stacking tensors across batch dimension.
    """
    collated = {}
    for key in batch[0].keys():
        collated[key] = torch.stack([item[key] for item in batch], dim=0)
    return collated
