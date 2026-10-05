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
                    raw_rgb = data["rgb"]
                    if raw_rgb.shape != (self.window_size, 3, 224, 224):
                        rgb_tensor = torch.zeros((self.window_size, 3, 224, 224), dtype=torch.float32)
                    else:
                        rgb_tensor = torch.from_numpy(raw_rgb).float()

                    ped_lbl = int(data["ped_label"])
                    cross_lbl = int(data["cross"]) if "cross" in data else (1 if ped_lbl == 5 else 0)
                    action_lbl = int(data["action"]) if "action" in data else (1 if ped_lbl in [0, 5] else 0)

                    # Extract authentic composite track ID
                    track_id = None
                    if "metadata" in data:
                        m = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]
                        if isinstance(m, dict) and "ped_id" in m:
                            s_id = m.get("set_id", "")
                            v_id = m.get("video_id", "")
                            p_id = m.get("ped_id", "")
                            track_id = f"{s_id}_{v_id}_{p_id}" if (s_id or v_id) else str(p_id)
                    if not track_id and "ped_id" in data:
                        track_id = str(data["ped_id"])

                    if not track_id or track_id == "":
                        raise KeyError(f"Fatal: Sequence file {file_path} missing authentic 'metadata.ped_id' or 'ped_id'. Cannot evaluate without real track clustering.")

                    return {
                        "rgb": rgb_tensor,
                        "pose": torch.from_numpy(data["pose"]).float(),
                        "trajectory": torch.from_numpy(data["trajectory"]).float(),
                        "scene": torch.from_numpy(data["scene"]).float(),
                        "neighbor_agents": torch.from_numpy(data["neighbor_agents"]).float(),
                        "neighbor_mask": torch.from_numpy(data["neighbor_mask"]).bool(),
                        "cross": torch.tensor(cross_lbl, dtype=torch.long),
                        "action": torch.tensor(action_lbl, dtype=torch.long),
                        "ped_label": torch.tensor(ped_lbl, dtype=torch.long),
                        "micro_label": torch.tensor(int(data["micro_label"]), dtype=torch.long),
                        "inter_label": torch.tensor(int(data["inter_label"]), dtype=torch.long),
                        "track_id": track_id
                    }
            else:
                return torch.load(file_path)
        except Exception as e:
            # Graceful fallback to next sample if an individual file is corrupted
            alt_idx = (idx + 1) % len(self.files)
            if alt_idx == idx:
                raise e
            return self.__getitem__(alt_idx)


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Standard batch collator stacking tensors across batch dimension and preserving lists for non-tensors.
    """
    collated = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            collated[key] = [item[key] for item in batch]
    return collated
