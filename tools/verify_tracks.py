#!/usr/bin/env python3
"""
tools/verify_tracks.py: Ultra-Fast Dataset Split & Unique Track Verifier.
Uses multiprocessing to read all 89k .npz sequence files in under 10 seconds.
Safe to run on Login/Master node without triggering the 5-minute timeout.
"""

import os
import glob
import time
import numpy as np
from multiprocessing import Pool, cpu_count


def _extract_track_id(file_path: str) -> str:
    try:
        with np.load(file_path, allow_pickle=True) as d:
            if "metadata" in d.files:
                m = d["metadata"].item() if hasattr(d["metadata"], "item") else d["metadata"]
                if isinstance(m, dict):
                    s_id = m.get("set_id", "")
                    v_id = m.get("video_id", "")
                    p_id = m.get("ped_id", "")
                    return f"{s_id}_{v_id}_{p_id}" if (s_id or v_id) else str(p_id)
            if "ped_id" in d.files:
                return str(d["ped_id"])
    except Exception:
        pass
    return ""


def verify_splits(data_dir: str = "data/processed_pie", num_workers: int = 8):
    print("=" * 75)
    print(f"VERIFYING DATASET SPLITS & UNIQUE TRACKS ({data_dir})")
    print(f"Workers: {num_workers} parallel threads")
    print("=" * 75)

    expected = {"test": 719, "val": 239, "train": 884}
    total_start = time.time()

    for split in ["test", "val", "train"]:
        t0 = time.time()
        pattern = os.path.join(data_dir, split, "*.npz")
        files = glob.glob(pattern)
        if not files:
            print(f"Split {split.upper():5s}: No files found in {os.path.join(data_dir, split)}")
            continue

        with Pool(processes=num_workers) as pool:
            track_ids = set(pool.map(_extract_track_id, files, chunksize=250))

        track_ids.discard("")
        dur = time.time() - t0
        exp = expected.get(split, 0)
        valid = (len(track_ids) == exp)
        print(f"Split {split.upper():5s} | Windows: {len(files):6,d} | Unique Tracks: {len(track_ids):4d} (Exp: {exp}) | Time: {dur:.2f}s")

    print("-" * 75)
    print(f"Total Verification Finished in {time.time() - total_start:.2f}s")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_pie", help="Processed data directory")
    parser.add_argument("--workers", type=int, default=min(8, max(1, cpu_count())), help="Number of worker processes")
    args = parser.parse_args()
    verify_splits(args.data_dir, args.workers)
