"""
Multimodal Dataset Loader.
Loads preprocessed .npz sequence windows from disk:
- Trajectory kinematics [T, 5]   (required)
- Interacting neighbor agents [T, N, 5] + mask [T, N]   (required)
- RGB [T, C, H, W], pose [T, J, D], scene [T, K]   (optional; only returned when present on disk
  with a real shape, so trajectory-only experiments do not allocate placeholder tensors)
- Targets: cross (primary), action, ped_label, micro_label, inter_label

Errors are raised loudly: no silent synthetic fallback and no silent substitution of corrupt files.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List
from datasets.synthetic_dataset import SyntheticMultimodalDataset


class MultimodalSequenceDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        window_size: int = 32,
        fallback_to_synthetic: bool = False,
        synthetic_samples: int = 128,
        load_rgb: bool = False,
    ):
        super().__init__()
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            if fallback_to_synthetic:
                split_dir = None
            else:
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
        self.data_dir = split_dir
        self.split = split
        self.window_size = window_size
        self.load_rgb = load_rgb

        self.files: List[str] = []
        if self.data_dir is not None:
            candidates = sorted(glob.glob(os.path.join(self.data_dir, "*.npz")))
            self.files = [f for f in candidates if os.path.getsize(f) > 0]

        self.synthetic_backup = None
        if len(self.files) == 0:
            if fallback_to_synthetic:
                print(f"[Dataset] WARNING: no data for split '{split}'. Using SYNTHETIC data ({synthetic_samples} samples).")
                self.synthetic_backup = SyntheticMultimodalDataset(num_samples=synthetic_samples, window_size=window_size)
            else:
                raise FileNotFoundError(f"No .npz sequence files found in {self.data_dir}")

    def __len__(self) -> int:
        if self.synthetic_backup is not None:
            return len(self.synthetic_backup)
        return len(self.files)

    @staticmethod
    def _track_id(data) -> str:
        track_id = None
        if "metadata" in data.files:
            m = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]
            if isinstance(m, dict) and m.get("ped_id", "") != "":
                s_id = m.get("set_id", "") or m.get("clip_id", "")
                v_id = m.get("video_id", "")
                p_id = m.get("ped_id", "")
                track_id = f"{s_id}_{v_id}_{p_id}" if (s_id or v_id) else str(p_id)
        if not track_id and "ped_id" in data.files and str(data["ped_id"]) != "":
            track_id = str(data["ped_id"])
        return track_id

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.synthetic_backup is not None:
            return self.synthetic_backup[idx]

        file_path = self.files[idx]
        try:
            with np.load(file_path, allow_pickle=True) as data:
                traj = data["trajectory"]
                T = traj.shape[0]
                track_id = self._track_id(data)
                if not track_id:
                    raise KeyError("missing metadata.ped_id / ped_id (needed for track-clustered statistics)")

                item = {
                    "trajectory": torch.from_numpy(traj).float(),
                    "neighbor_agents": torch.from_numpy(data["neighbor_agents"]).float(),
                    "neighbor_mask": torch.from_numpy(data["neighbor_mask"]).bool(),
                    "cross": torch.tensor(int(data["cross"]), dtype=torch.long),
                    "action": torch.tensor(int(data["action"]) if "action" in data.files else 0, dtype=torch.long),
                    "ped_label": torch.tensor(int(data["ped_label"]) if "ped_label" in data.files else 0, dtype=torch.long),
                    "micro_label": torch.tensor(int(data["micro_label"]) if "micro_label" in data.files else 0, dtype=torch.long),
                    "inter_label": torch.tensor(int(data["inter_label"]) if "inter_label" in data.files else 0, dtype=torch.long),
                    "track_id": track_id,
                }
                # Optional modalities: only when real data with matching temporal length exists
                if "pose" in data.files and data["pose"].ndim == 3 and data["pose"].shape[0] == T:
                    item["pose"] = torch.from_numpy(data["pose"]).float()
                if "scene" in data.files and data["scene"].ndim == 2 and data["scene"].shape[0] == T:
                    item["scene"] = torch.from_numpy(data["scene"]).float()
                if self.load_rgb and "rgb" in data.files and data["rgb"].ndim == 4 and data["rgb"].shape[0] == T:
                    item["rgb"] = torch.from_numpy(data["rgb"]).float()
                if "bbox" in data.files:
                    item["bbox"] = torch.from_numpy(data["bbox"]).float()
                return item
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}") from e


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stacks tensors present in every sample; keeps non-tensors as lists."""
    collated = {}
    keys = set(batch[0].keys())
    for item in batch[1:]:
        keys &= set(item.keys())
    for key in batch[0].keys():
        if key not in keys:
            continue
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            collated[key] = [item[key] for item in batch]
    return collated
