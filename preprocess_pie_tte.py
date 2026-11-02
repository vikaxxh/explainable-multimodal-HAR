"""
Builds the PIE crossing-intention benchmark used for the CVMI 2027 paper.

Protocol (standard PIE intention setup, Kotseruba et al. WACV 2021):
  - Observation: 16 frames (0.53 s at 30 fps)
  - Prediction horizon: the observation ends 30-60 frames (1-2 s) before the crossing event
  - Splits: train = set01, set02, set04 | val = set05, set06 | test = set03
  - Pedestrians annotated only as 'crossing-irrelevant' are excluded
  - Kinematics computed with dt = 1/30 s on box centres normalised by image size

Usage:
  python preprocess_pie_tte.py --pie_dir data/raw/PIE --output_dir data/processed_pie_tte --clean
"""

import os
import glob
import json
import argparse
import numpy as np

from datasets.pie_adapter import PIEAdapter

SPLITS = {
    "train": ["set01", "set02", "set04"],
    "val": ["set05", "set06"],
    "test": ["set03"],
}


def parse_args():
    p = argparse.ArgumentParser(description="PIE time-to-event intention benchmark preprocessor")
    p.add_argument("--pie_dir", default="data/raw/PIE")
    p.add_argument("--output_dir", default="data/processed_pie_tte")
    p.add_argument("--obs_len", type=int, default=16)
    p.add_argument("--tte_min", type=int, default=30)
    p.add_argument("--tte_max", type=int, default=60)
    p.add_argument("--overlap", type=float, default=0.6)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--num_neighbors", type=int, default=4)
    p.add_argument("--neighbor_radius", type=float, default=0.20)
    p.add_argument("--clean", action="store_true", help="Delete existing .npz files in output_dir first")
    return p.parse_args()


def main():
    args = parse_args()
    adapter = PIEAdapter(pie_root=args.pie_dir, window_size=args.obs_len)
    pie_data = adapter.load_annotations()
    if not pie_data:
        raise SystemExit(f"No PIE annotations found under {args.pie_dir}. Aborting (no synthetic fallback).")

    found_sets = sorted(pie_data.keys())
    missing = [s for sets in SPLITS.values() for s in sets if s not in found_sets]
    if missing:
        raise SystemExit(f"Missing PIE sets {missing}; found {found_sets}. Aborting.")

    summary = {"protocol": vars(args), "splits": {}}
    for split, sets in SPLITS.items():
        out_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        if args.clean:
            for f in glob.glob(os.path.join(out_dir, "*.npz")):
                os.remove(f)
        elif glob.glob(os.path.join(out_dir, "*.npz")):
            raise SystemExit(f"{out_dir} already contains .npz files. Use --clean to overwrite.")

        seqs = adapter.extract_tte_sequences(
            pie_data,
            set_filter=sets,
            obs_len=args.obs_len,
            tte_range=(args.tte_min, args.tte_max),
            overlap=args.overlap,
            fps=args.fps,
            num_neighbors=args.num_neighbors,
            neighbor_radius=args.neighbor_radius,
        )
        for i, s in enumerate(seqs):
            md = s["metadata"]
            np.savez(
                os.path.join(out_dir, f"seq_{i:06d}.npz"),
                trajectory=s["trajectory"],
                bbox=s["bbox"],
                neighbor_agents=s["neighbor_agents"],
                neighbor_mask=s["neighbor_mask"],
                cross=s["cross"],
                action=s["action"],
                ped_label=s["ped_label"],
                micro_label=s["micro_label"],
                inter_label=s["inter_label"],
                agent_type=s["agent_type"],
                metadata=md,
                ped_id=str(md["ped_id"]),
            )
        tracks = {f'{s["metadata"]["set_id"]}_{s["metadata"]["video_id"]}_{s["metadata"]["ped_id"]}' for s in seqs}
        pos_tracks = {f'{s["metadata"]["set_id"]}_{s["metadata"]["video_id"]}_{s["metadata"]["ped_id"]}' for s in seqs if s["cross"] == 1}
        k_last = [int(s["neighbor_mask"][-1].sum()) for s in seqs]
        summary["splits"][split] = {
            "sets": sets,
            "windows": len(seqs),
            "crossing_windows": int(sum(s["cross"] for s in seqs)),
            "tracks": len(tracks),
            "crossing_tracks": len(pos_tracks),
            "windows_K0": int(sum(1 for k in k_last if k == 0)),
            "windows_K1": int(sum(1 for k in k_last if k == 1)),
            "windows_K2plus": int(sum(1 for k in k_last if k >= 2)),
        }
        print(f"[{split}] {summary['splits'][split]}")

    with open(os.path.join(args.output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {os.path.join(args.output_dir, 'dataset_summary.json')}")


if __name__ == "__main__":
    main()
