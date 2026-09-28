"""
Measure Neighbor Distribution for Hypothesis 1 (H1).
Computes exact empirical neighbor density (K=0, K=1, K>=2) at both window-level
and pedestrian track-level on official PIE and JAAD splits.

Usage:
  python tools/measure_neighbor_distribution.py --data_dir data/processed_pie_clean --split test
"""

import os
import glob
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, Any, List


def parse_args():
    parser = argparse.ArgumentParser(description="Measure Neighbor Distribution (H1)")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed sequences directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train", "all"], help="Split to analyze")
    parser.add_argument("--output_file", type=str, default=None, help="Optional text file to save results")
    return parser.parse_args()


def analyze_split(split_dir: str) -> Dict[str, Any]:
    """Analyzes all .npz files in a split directory to compute K distribution."""
    npz_files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
    if not npz_files:
        return {"error": f"No .npz files found in {split_dir}"}

    total_windows = len(npz_files)
    k_counts_window = {0: 0, 1: 0, "2+": 0}
    track_windows = defaultdict(list)

    for fpath in npz_files:
        try:
            data = np.load(fpath, allow_pickle=True)
            # Neighbor mask: (T, N)
            mask = data["neighbor_mask"]
            # Active neighbors at final observation frame
            k_active = int(mask[-1].sum())

            if k_active == 0:
                k_counts_window[0] += 1
            elif k_active == 1:
                k_counts_window[1] += 1
            else:
                k_counts_window["2+"] += 1

            # Extract track / pedestrian ID if available (fallback to filename prefix)
            ped_id = str(data.get("ped_id", os.path.basename(fpath).split("_")[0]))
            track_windows[ped_id].append(k_active)
        except Exception as e:
            continue

    # Track-level aggregation: maximum active neighbors observed along the track
    k_counts_track = {0: 0, 1: 0, "2+": 0}
    for ped_id, k_list in track_windows.items():
        max_k = max(k_list)
        if max_k == 0:
            k_counts_track[0] += 1
        elif max_k == 1:
            k_counts_track[1] += 1
        else:
            k_counts_track["2+"] += 1

    total_tracks = len(track_windows)

    return {
        "total_windows": total_windows,
        "total_tracks": total_tracks,
        "window_level": {
            "K=0": (k_counts_window[0], round((k_counts_window[0] / total_windows) * 100.0, 2)),
            "K=1": (k_counts_window[1], round((k_counts_window[1] / total_windows) * 100.0, 2)),
            "K>=2": (k_counts_window["2+"], round((k_counts_window["2+"] / total_windows) * 100.0, 2)),
        },
        "track_level": {
            "K=0": (k_counts_track[0], round((k_counts_track[0] / max(1, total_tracks)) * 100.0, 2)),
            "K=1": (k_counts_track[1], round((k_counts_track[1] / max(1, total_tracks)) * 100.0, 2)),
            "K>=2": (k_counts_track["2+"], round((k_counts_track["2+"] / max(1, total_tracks)) * 100.0, 2)),
        }
    }


def main():
    args = parse_args()

    splits = ["test", "val", "train"] if args.split == "all" else [args.split]

    print("=" * 80)
    print("HYPOTHESIS 1 (H1): EMPIRICAL NEIGHBOR AVAILABILITY MEASUREMENT")
    print(f"Data Source: {args.data_dir}")
    print("=" * 80)

    for sp in splits:
        sp_dir = os.path.join(args.data_dir, sp)
        if not os.path.isdir(sp_dir):
            print(f"[Warning] Split directory not found: {sp_dir}")
            continue

        res = analyze_split(sp_dir)
        if "error" in res:
            print(f"[{sp.upper()}] Error: {res['error']}")
            continue

        print(f"\n--- Split: {sp.upper()} (Total Windows: {res['total_windows']:,} | Tracks: {res['total_tracks']:,}) ---")
        print(f"{'Stratum':<15} | {'Window Count':>15} | {'Window Share':>14} | {'Track Count':>15} | {'Track Share':>13}")
        print("-" * 80)
        for k in ["K=0", "K=1", "K>=2"]:
            w_cnt, w_pct = res["window_level"][k]
            t_cnt, t_pct = res["track_level"][k]
            print(f"{k:<15} | {w_cnt:>15,} | {w_pct:>13.2f}% | {t_cnt:>15,} | {t_pct:>12.2f}%")

        # Pre-registered H1 decision criterion
        pct_active = res["window_level"]["K=1"][1] + res["window_level"]["K>=2"][1]
        print("-" * 80)
        if pct_active < 50.0:
            print(f"[H1 Finding]: Non-ego neighbors exist in a MINORITY of windows ({pct_active:.2f}% active, {res['window_level']['K=0'][1]:.2f}% isolated).")
            print("  >>> Sparsity confirmed. Interaction claims must be evaluated stratified by neighbor presence.")
        else:
            print(f"[H1 Finding]: Non-ego neighbors exist in a MAJORITY of windows ({pct_active:.2f}% active).")
            print("  >>> Sparsity hypothesis refuted. Benchmark contains dense interactive traffic.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
