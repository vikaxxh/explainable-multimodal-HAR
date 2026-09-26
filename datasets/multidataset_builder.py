"""
Multi-Dataset Builder: Unified Ingestion of PIE, JAAD, TITAN, and MicroVision.
Combines 4 major urban benchmarks:
1. PIE: Pedestrian + Bicycle tracks and intentional actions
2. JAAD: Dense pedestrian behavior clips (346 videos, 2.7k tracks)
3. TITAN: Honda Research complex actions & vehicle/cyclist interactions
4. MicroVision: Targeted e-scooter and micromobility dynamics
Massively scales sample diversity and rare interaction instances to achieve 95+% accuracy.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from datasets.pie_adapter import PIEAdapter
from datasets.jaad_adapter import JAADAdapter
from datasets.titan_adapter import TITANAdapter
from datasets.microvision_adapter import MicroVisionAdapter


class MultiDatasetBuilder:
    """
    Constructs unified multi-dataset corpus from PIE, JAAD, TITAN, and MicroVision.
    """

    def __init__(
        self,
        output_dir: str = "data/processed",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ):
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.rng = np.random.RandomState(seed)

    def build(
        self,
        pie_dir: Optional[str] = None,
        jaad_dir: Optional[str] = None,
        titan_dir: Optional[str] = None,
        microvision_dir: Optional[str] = None,
        max_pie_samples: int = 5000,
        max_jaad_samples: int = 5000,
        max_titan_samples: int = 1000,
        max_micro_samples: int = 500,
        stride: int = 8,
        clean_existing: bool = False
    ) -> Dict[str, int]:
        """
        Extracts samples from all configured datasets, merges them, and saves to disk.
        Set max_*_samples <= 0 for unlimited (extract all available tracks).
        Use stride=4 for dense stride or stride=8 for standard stride.
        """
        import glob
        if clean_existing:
            print(f"[MultiDataset] Cleaning existing .npz files in {self.output_dir}...")
            for s in ["train", "val", "test"]:
                sdir = os.path.join(self.output_dir, s)
                if os.path.isdir(sdir):
                    for f in glob.glob(os.path.join(sdir, "*.npz")):
                        try:
                            os.remove(f)
                        except OSError:
                            pass

        all_samples: List[Dict[str, Any]] = []

        # 1. Ingest PIE (Pedestrians + Bicycles)
        if max_pie_samples != -1:
            print("\n[MultiDataset] --- Ingesting PIE Dataset ---")
            pie_adapter = PIEAdapter(pie_root=pie_dir or "data/raw/PIE", stride=stride)
            pie_raw = pie_adapter.load_annotations()
            pie_lim = None if max_pie_samples <= 0 else max_pie_samples
            pie_samples = pie_adapter.extract_sequences_from_annotations(pie_raw, max_samples=pie_lim)
            print(f"[MultiDataset] Extracted {len(pie_samples)} sequences from PIE.")
            all_samples.extend(pie_samples)

        # 2. Ingest JAAD (Pedestrian Companion Dataset)
        if max_jaad_samples != -1:
            print(f"\n[MultiDataset] --- Ingesting JAAD Dataset (stride={stride}) ---")
            jaad_adapter = JAADAdapter(jaad_root=jaad_dir or "data/raw/JAAD", stride=stride)
            jaad_raw = jaad_adapter.load_annotations()
            jaad_lim = None if max_jaad_samples <= 0 else max_jaad_samples
            jaad_samples = jaad_adapter.extract_sequences_from_annotations(jaad_raw, max_samples=jaad_lim)
            print(f"[MultiDataset] Extracted {len(jaad_samples)} sequences from JAAD.")
            all_samples.extend(jaad_samples)

        # 3. Ingest TITAN (Honda Research Complex Interactions)
        if (titan_dir or max_titan_samples != 0) and max_titan_samples != -1:
            print("\n[MultiDataset] --- Ingesting Honda TITAN Dataset ---")
            titan_adapter = TITANAdapter(titan_root=titan_dir or "data/raw/TITAN")
            titan_raw = titan_adapter.load_annotations()
            titan_lim = None if max_titan_samples <= 0 else max_titan_samples
            titan_samples = titan_adapter.extract_sequences(titan_raw, max_samples=titan_lim)
            print(f"[MultiDataset] Extracted {len(titan_samples)} sequences from TITAN.")
            all_samples.extend(titan_samples)

        # 4. Ingest MicroVision (E-scooter Dynamics)
        if (microvision_dir or max_micro_samples != 0) and max_micro_samples != -1:
            print("\n[MultiDataset] --- Ingesting MicroVision Dataset ---")
            micro_adapter = MicroVisionAdapter(data_root=microvision_dir or "data/raw/MicroVision")
            micro_lim = None if max_micro_samples <= 0 else max_micro_samples
            micro_samples = micro_adapter.extract_scooter_sequences(max_samples=micro_lim)
            print(f"[MultiDataset] Extracted {len(micro_samples)} sequences from MicroVision.")
            all_samples.extend(micro_samples)

        # 5. Shuffle and Partition
        self.rng.shuffle(all_samples)
        n_total = len(all_samples)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        splits = {
            "train": all_samples[:n_train],
            "val": all_samples[n_train:n_train + n_val],
            "test": all_samples[n_train + n_val:]
        }

        # 6. Save to disk as .npz records
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(iterable, desc=""):
                return iterable

        summary_counts = {}
        for split_name, samples in splits.items():
            split_dir = os.path.join(self.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            summary_counts[split_name] = len(samples)

            for idx, sample in enumerate(tqdm(samples, desc=f"[Saving {split_name}]")):
                file_name = f"seq_{idx:05d}.npz"
                file_path = os.path.join(split_dir, file_name)
                np.savez(
                    file_path,
                    rgb=sample["rgb"],
                    pose=sample["pose"],
                    trajectory=sample["trajectory"],
                    scene=sample["scene"],
                    neighbor_agents=sample["neighbor_agents"],
                    neighbor_mask=sample["neighbor_mask"],
                    ped_label=sample["ped_label"],
                    micro_label=sample["micro_label"],
                    inter_label=sample["inter_label"],
                    agent_type=sample["agent_type"]
                )

        print(f"\n[MultiDataset] Complete! Saved {n_total} unified multi-dataset samples into {self.output_dir}/")
        for s_name, count in summary_counts.items():
            print(f"  • {s_name}: {count} sequences")

        return summary_counts
