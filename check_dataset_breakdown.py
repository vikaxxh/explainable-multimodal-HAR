"""
Dataset Inspection Utility for X-MIST / EMIT-HAR.
Reports:
1. Available raw datasets in data/raw/
2. Total processed sequences per split (train, val, test)
3. Dataset breakdown per split (PIE, JAAD, TITAN, MicroVision)
4. Behavior class distribution
"""

import os
import glob
from collections import Counter
import numpy as np

def check_raw_datasets():
    print("=" * 65)
    print("RAW DATASETS IN data/raw/")
    print("=" * 65)
    raw_dirs = {
        "PIE": "data/raw/PIE",
        "JAAD": "data/raw/JAAD",
        "TITAN": "data/raw/TITAN",
        "MicroVision": "data/raw/MicroVision"
    }
    for name, path in raw_dirs.items():
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "**", "*.*"), recursive=True)
            xmls = [f for f in files if f.endswith(".xml")]
            jsons = [f for f in files if f.endswith(".json")]
            pkls = [f for f in files if f.endswith(".pkl")]
            print(f"  • {name:12s}: FOUND ({len(files)} total files | XML: {len(xmls)}, JSON: {len(jsons)}, PKL: {len(pkls)})")
        else:
            print(f"  • {name:12s}: NOT FOUND")
    print()

def check_processed_datasets():
    print("=" * 65)
    print("PROCESSED SEQUENCES IN data/processed/")
    print("=" * 65)

    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = os.path.join("data", "processed", split)
        files = glob.glob(os.path.join(split_dir, "*.npz"))
        print(f"\n--- Split: [{split.upper()}] ({len(files)} total sequences) ---")

        if not files:
            print("  (No sequences found)")
            continue

        dataset_counts = Counter()
        ped_counts = Counter()

        for f in files:
            try:
                data = np.load(f, allow_pickle=True)
                if "metadata" in data and data["metadata"].ndim == 0:
                    meta = data["metadata"].item()
                    d_name = meta.get("dataset", "Unknown")
                else:
                    d_name = "Unknown"
                dataset_counts[d_name] += 1

                if "ped_label" in data:
                    ped_counts[int(data["ped_label"])] += 1
            except Exception:
                continue

        print("  Dataset Breakdown:")
        for d_name, cnt in dataset_counts.most_common():
            pct = (cnt / len(files)) * 100
            print(f"    • {d_name:15s}: {cnt:5d} sequences ({pct:5.1f}%)")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    check_raw_datasets()
    check_processed_datasets()
