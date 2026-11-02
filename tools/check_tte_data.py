#!/usr/bin/env python3
"""
Sanity checks for the PIE time-to-event dataset (run after preprocess_pie_tte.py, before training).

Reports per split: windows, tracks, crossing share (window and track level), neighbour count K,
speed distribution, share of neighbour edges with a finite TTC, time-to-event range, and checks
that no track appears in more than one split.

Usage: python tools/check_tte_data.py --data_dir data/processed_pie_tte
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def closing_ttc(p, n, min_closing=1e-3, cap=10.0):
    rel = p[:, None, :2] - n[..., :2]
    dist = np.linalg.norm(rel, axis=-1)
    rv = p[:, None, 2:4] - n[..., 2:4]
    closing = -np.sum(rv * rel / (dist[..., None] + 1e-5), axis=-1)
    ttc = np.where(closing > min_closing, dist / (np.maximum(closing, min_closing) + 1e-5), cap)
    return np.clip(ttc, 0, cap), closing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_pie_tte")
    ap.add_argument("--min_closing_speed", type=float, default=1e-3)
    ap.add_argument("--tte_min", type=int, default=30)
    ap.add_argument("--tte_max", type=int, default=60)
    args = ap.parse_args()

    report, tracks_by_split = {}, {}
    for split in ["train", "val", "test"]:
        files = sorted(glob.glob(os.path.join(args.data_dir, split, "*.npz")))
        if not files:
            print(f"[{split}] no files")
            continue
        y, ks, speeds, ttcs, ttes, tracks, pos_tracks = [], [], [], [], [], set(), set()
        for f in files:
            d = np.load(f, allow_pickle=True)
            m = d["metadata"].item()
            tid = f'{m["set_id"]}_{m["video_id"]}_{m["ped_id"]}'
            tracks.add(tid)
            c = int(d["cross"])
            y.append(c)
            if c:
                pos_tracks.add(tid)
            mask = d["neighbor_mask"]
            ks.append(int(mask[-1].sum()))
            tr = d["trajectory"]
            speeds.append(float(np.linalg.norm(tr[:, 2:4], axis=-1).mean()))
            ttes.append(int(m.get("tte_frames", -1)))
            if mask.any():
                ttc, _ = closing_ttc(tr, d["neighbor_agents"], args.min_closing_speed)
                ttcs.extend(ttc[mask].tolist())
        y, ks, speeds, ttcs = np.array(y), np.array(ks), np.array(speeds), np.array(ttcs)
        tracks_by_split[split] = tracks
        report[split] = {
            "windows": len(files),
            "tracks": len(tracks),
            "crossing_window_share": round(float(y.mean()), 4),
            "crossing_tracks": len(pos_tracks),
            "K0_share": round(float((ks == 0).mean()), 4),
            "K1_share": round(float((ks == 1).mean()), 4),
            "K2plus_share": round(float((ks >= 2).mean()), 4),
            "ped_speed_percentiles_p10_p50_p90": [round(float(v), 5) for v in np.percentile(speeds, [10, 50, 90])],
            "finite_ttc_edge_share": round(float((ttcs < 10.0).mean()), 4) if len(ttcs) else None,
            "ttc_percentiles_p10_p50_p90": [round(float(v), 3) for v in np.percentile(ttcs, [10, 50, 90])] if len(ttcs) else None,
            "tte_frames_min_max": [int(min(ttes)), int(max(ttes))],
        }
        print(f"[{split}] " + json.dumps(report[split]))

    # Leakage check
    names = list(tracks_by_split)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = tracks_by_split[names[i]] & tracks_by_split[names[j]]
            status = "OK" if not inter else f"LEAK: {len(inter)} shared tracks"
            print(f"Track overlap {names[i]} vs {names[j]}: {status}")
            report[f"overlap_{names[i]}_{names[j]}"] = len(inter)

    errors, warnings = [], []
    for name, n in report.items():
        if name.startswith("overlap_") and n:
            errors.append(f"{name}: {n} tracks appear in more than one split")
    for s, r in report.items():
        if not isinstance(r, dict):
            continue
        lo, hi = r["tte_frames_min_max"]
        if lo < args.tte_min or hi > args.tte_max:
            errors.append(f"{s}: time-to-event {lo}-{hi} frames outside [{args.tte_min}, {args.tte_max}]")
        if r["crossing_window_share"] in (0.0, 1.0):
            errors.append(f"{s}: only one class present")
        if r["finite_ttc_edge_share"] is not None and r["finite_ttc_edge_share"] < 0.05:
            warnings.append(f"{s}: <5% of edges have finite TTC -> lower min_closing_speed")
    print("WARNINGS:" if warnings else "No warnings.", *warnings, sep="\n  ")
    report["errors"], report["warnings"] = errors, warnings
    with open(os.path.join(args.data_dir, "sanity_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    if errors:
        print("ERRORS:", *errors, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
