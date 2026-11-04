"""
Builds the JAAD crossing-intention benchmark (replication of the PIE TTE protocol for CVMI 2027).

Same protocol as preprocess_pie_tte.py, reusing PIEAdapter.extract_tte_sequences:
  - Observation 16 frames (30 fps); observation ends 30-60 frames (1-2 s) before the crossing event
    (first 'crossing' frame) or before the last annotated frame for non-crossing pedestrians
  - Pedestrians annotated only as 'crossing-irrelevant' are excluded
  - Official video-disjoint splits from data/splits/jaad_<subset>/{train,val,test}.txt
Differences to PIE (JAAD has no vehicle boxes):
  - Primary agents: 'pedestrian' tracks (JAAD_beh) or also 'ped' tracks (JAAD_all)
  - Neighbours: all other pedestrian tracks ('pedestrian', 'ped', 'people')
  - Per-video image size read from the XML meta (JAAD mixes 1920x1080 and 1280x720 clips)

Usage:
  python preprocess_jaad_tte.py --jaad_dir data/raw/JAAD --output_dir data/processed_jaad_tte --clean
"""

import os
import glob
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np

from datasets.pie_adapter import PIEAdapter

PRIMARY_LABELS = {"beh": {"pedestrian"}, "all": {"pedestrian", "ped"}}
NEIGHBOUR_LABELS = {"pedestrian", "ped", "people"}


def parse_args():
    p = argparse.ArgumentParser(description="JAAD time-to-event intention benchmark preprocessor")
    p.add_argument("--jaad_dir", default="data/raw/JAAD", help="Folder containing annotations/*.xml")
    p.add_argument("--output_dir", default="data/processed_jaad_tte")
    p.add_argument("--subset", default="beh", choices=["beh", "all"])
    p.add_argument("--splits_dir", default="data/splits")
    p.add_argument("--obs_len", type=int, default=16)
    p.add_argument("--tte_min", type=int, default=30)
    p.add_argument("--tte_max", type=int, default=60)
    p.add_argument("--overlap", type=float, default=0.6)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--num_neighbors", type=int, default=4)
    p.add_argument("--neighbor_radius", type=float, default=0.20)
    p.add_argument("--clean", action="store_true", help="Delete existing .npz files in output_dir first")
    return p.parse_args()


def cross_value(text):
    t = (text or "").strip().lower()
    if "irrelevant" in t or t == "-1":
        return -1
    return 1 if t in ("1", "true", "yes", "crossing") else 0


def parse_video(xml_path, primary_labels):
    root = ET.parse(xml_path).getroot()
    w = root.findtext(".//original_size/width")
    h = root.findtext(".//original_size/height")
    if w is None or h is None:
        raise SystemExit(f"{xml_path}: no <original_size> in meta; cannot normalise coordinates.")
    peds, others = {}, {}
    for n, track in enumerate(root.findall(".//track")):
        label = track.get("label", "").lower()
        if label not in NEIGHBOUR_LABELS:
            continue
        tid, frames, boxes, cross, actions = None, [], [], [], []
        for box in track.findall("box"):
            if box.get("outside", "0") == "1":
                continue
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            frames.append(int(box.get("frame")))
            boxes.append([xtl, ytl, max(1.0, xbr - xtl), max(1.0, ybr - ytl)])
            c, a = 0, 1
            for attr in box.findall("attribute"):
                name = attr.get("name", "").lower()
                if name == "id" and tid is None:
                    tid = (attr.text or "").strip()
                elif name == "cross":
                    c = cross_value(attr.text)
                elif name == "action":
                    a = 1 if "walk" in (attr.text or "").lower() else 0
            cross.append(c)
            actions.append(a)
        if not frames:
            continue
        tid = tid or f"{label}_{n}"
        if label in primary_labels:
            peds[tid] = {"frames": frames, "bbox": boxes, "cross": cross, "actions": actions}
        else:
            # Non-primary pedestrians only act as neighbours (stored in the generic agent slot)
            others[tid] = {"frames": frames, "bbox": boxes, "type": "pedestrian"}
    return {"ped_annotations": peds, "vehicle_annotations": others, "image_size": (float(w), float(h))}


def main():
    args = parse_args()
    xml_files = sorted(glob.glob(os.path.join(args.jaad_dir, "annotations", "*.xml")))
    if not xml_files:
        raise SystemExit(f"No JAAD XML files under {args.jaad_dir}/annotations. Aborting (no synthetic fallback).")
    videos = {os.path.splitext(os.path.basename(f))[0]: f for f in xml_files}

    split_dir = os.path.join(args.splits_dir, f"jaad_{args.subset}")
    split_ids = {}
    for s in ("train", "val", "test"):
        path = os.path.join(split_dir, f"{s}.txt")
        if not os.path.exists(path):
            raise SystemExit(f"Missing split file {path}")
        split_ids[s] = [l.strip() for l in open(path) if l.strip()]
    for a in split_ids:
        for b in split_ids:
            if a < b and set(split_ids[a]) & set(split_ids[b]):
                raise SystemExit(f"Split files {a}/{b} share videos")
    missing = [v for ids in split_ids.values() for v in ids if v not in videos]
    if missing:
        raise SystemExit(f"{len(missing)} split videos have no XML (e.g. {missing[:3]}). Aborting.")

    adapter = PIEAdapter(pie_root=args.jaad_dir, window_size=args.obs_len)
    summary = {"protocol": vars(args), "splits": {}}
    for split, ids in split_ids.items():
        out_dir = os.path.join(args.output_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        if args.clean:
            for f in glob.glob(os.path.join(out_dir, "*.npz")):
                os.remove(f)
        elif glob.glob(os.path.join(out_dir, "*.npz")):
            raise SystemExit(f"{out_dir} already contains .npz files. Use --clean to overwrite.")

        data = {"jaad": {v: parse_video(videos[v], PRIMARY_LABELS[args.subset]) for v in ids}}
        seqs = adapter.extract_tte_sequences(
            data, obs_len=args.obs_len, tte_range=(args.tte_min, args.tte_max), overlap=args.overlap,
            fps=args.fps, num_neighbors=args.num_neighbors, neighbor_radius=args.neighbor_radius,
            dataset_name="JAAD",
        )
        for i, s in enumerate(seqs):
            md = s["metadata"]
            np.savez(
                os.path.join(out_dir, f"seq_{i:06d}.npz"),
                trajectory=s["trajectory"], bbox=s["bbox"],
                neighbor_agents=s["neighbor_agents"], neighbor_mask=s["neighbor_mask"],
                cross=s["cross"], action=s["action"], ped_label=s["ped_label"],
                micro_label=s["micro_label"], inter_label=s["inter_label"],
                agent_type=s["agent_type"], metadata=md, ped_id=str(md["ped_id"]),
            )
        key = lambda s: f'{s["metadata"]["set_id"]}_{s["metadata"]["video_id"]}_{s["metadata"]["ped_id"]}'
        k_last = [int(s["neighbor_mask"][-1].sum()) for s in seqs]
        summary["splits"][split] = {
            "videos": len(ids),
            "windows": len(seqs),
            "crossing_windows": int(sum(s["cross"] for s in seqs)),
            "tracks": len({key(s) for s in seqs}),
            "crossing_tracks": len({key(s) for s in seqs if s["cross"] == 1}),
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
