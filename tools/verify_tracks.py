#!/usr/bin/env python3
"""
tools/verify_tracks.py: Fast Dataset Split & Unique Track Verifier.
Reads processed PIE .npz files and reports window counts and unique tracks per split.
"""

import os
import glob
import numpy as np


def verify_splits(data_dir: str = "data/processed_pie"):
    print("=" * 75)
    print(f"VERIFYING DATASET SPLITS & UNIQUE TRACKS ({data_dir})")
    print("=" * 75)

    expected = {"test": 719, "val": 239, "train": 884}

    for split in ["test", "val", "train"]:
        files = glob.glob(os.path.join(data_dir, split, "*.npz"))
        if not files:
            print(f"Split {split.upper():5s}: No files found in {os.path.join(data_dir, split)}")
            continue

        unique_ids = set()
        for f in files:
            d = np.load(f, allow_pickle=True)
            if "metadata" in d.files:
                m = d["metadata"].item() if hasattr(d["metadata"], "item") else d["metadata"]
                if isinstance(m, dict):
                    s_id = m.get("set_id", "")
                    v_id = m.get("video_id", "")
                    p_id = m.get("ped_id", "")
                    unique_ids.add(f"{s_id}_{v_id}_{p_id}" if (s_id or v_id) else str(p_id))
                    continue
            if "ped_id" in d.files:
                unique_ids.add(str(d["ped_id"]))

        exp = expected.get(split, 0)
        print(f"Split {split.upper():5s} | Windows: {len(files):6,d} | Unique Tracks: {len(unique_ids):4d} (Benchmark Exp: {exp})")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_pie", help="Processed data directory")
    args = parser.parse_args()
    verify_splits(args.data_dir)
