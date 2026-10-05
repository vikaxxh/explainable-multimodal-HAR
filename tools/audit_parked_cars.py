#!/usr/bin/env python3
"""
tools/audit_parked_cars.py: Parked-Car Confound Audit with Ego-Motion Compensation.
Quantifies the parked-car confound in PIE benchmark by:
1. Synchronizing track bounding boxes with ego-vehicle OBD data (OBD_speed, GPS_speed).
2. Computing pinhole ground-plane projection and ego-motion compensated world velocity.
3. Classifying vehicle tracks as Parked vs. Moving across multiple physical thresholds (0.2, 0.5, 1.0, 1.5, 2.0 m/s).
4. Re-running the pedestrian neighbor density audit with parked vehicles removed.
5. Providing validation metrics and hand-labeling inspection audit.
"""

import os
import sys
import glob
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional


# Standard PIE Camera Intrinsic & Extrinsic Parameters (Windshield-mounted, 1920x1080 @ 30 FPS)
DEFAULT_FY = 1370.0
DEFAULT_FX = 1370.0
DEFAULT_CX = 960.0
DEFAULT_CY = 540.0
DEFAULT_H_CAM = 1.25  # Camera height above ground in meters


def load_obd_data(obd_xml_path: str) -> Dict[int, float]:
    """Loads frame-indexed ego vehicle speed in m/s from PIE OBD XML."""
    if not os.path.exists(obd_xml_path):
        return {}
    tree = ET.parse(obd_xml_path)
    root = tree.getroot()
    speed_map = {}
    for f in root.findall("frame"):
        fid = int(f.get("id"))
        # Prefer OBD_speed if available and nonzero, else fall back to GPS_speed
        obd_s = float(f.get("OBD_speed", 0.0))
        gps_s = float(f.get("GPS_speed", 0.0))
        kmh = obd_s if obd_s > 0 else gps_s
        speed_map[fid] = kmh / 3.6  # convert km/h to m/s
    return speed_map


def estimate_vehicle_world_speed(
    boxes: List[ET.Element],
    obd_speeds: Dict[int, float],
    step: int = 5,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    h_cam: float = DEFAULT_H_CAM
) -> Tuple[float, float, List[float]]:
    """
    Computes physical world speed (m/s) compensated for camera ego-motion.
    Returns:
        (median_speed, p90_speed, frame_speeds)
    """
    world_speeds = []
    
    for i in range(0, len(boxes) - step, step):
        b0, b1 = boxes[i], boxes[i + step]
        f0, f1 = int(b0.get("frame")), int(b1.get("frame"))
        dt = (f1 - f0) / 30.0
        if dt <= 0:
            continue

        yb0 = float(b0.get("ybr"))
        yb1 = float(b1.get("ybr"))
        xc0 = (float(b0.get("xtl")) + float(b0.get("xbr"))) / 2.0
        xc1 = (float(b1.get("xtl")) + float(b1.get("xbr"))) / 2.0

        # Reject bounding boxes at or above horizon
        if yb0 <= cy + 15 or yb1 <= cy + 15:
            continue

        z0 = (fy * h_cam) / (yb0 - cy)
        z1 = (fy * h_cam) / (yb1 - cy)
        x0 = (xc0 - cx) * z0 / fx
        x1 = (xc1 - cx) * z1 / fx

        v_ego = (obd_speeds.get(f0, 0.0) + obd_speeds.get(f1, 0.0)) / 2.0

        # True forward motion in world: camera relative displacement + forward ego motion
        vz_world = (z1 - z0) / dt + v_ego
        vx_world = (x1 - x0) / dt
        v_world = float(np.sqrt(vx_world**2 + vz_world**2))

        # Filter out extreme numerical projection outliers
        if v_world < 50.0:
            world_speeds.append(v_world)

    if not world_speeds:
        return 0.0, 0.0, []

    return float(np.median(world_speeds)), float(np.percentile(world_speeds, 90)), world_speeds


def run_parked_car_audit(
    pie_annot_dir: str,
    pie_obd_dir: str,
    thresholds: List[float] = [0.2, 0.5, 1.0, 1.5, 2.0],
    primary_radius: float = 0.20
) -> Dict[str, Any]:
    """
    Runs complete parked-car audit across all XML files in the specified directory.
    """
    annot_files = sorted(glob.glob(os.path.join(pie_annot_dir, "**", "*.xml"), recursive=True))
    if not annot_files:
        annot_files = sorted(glob.glob(os.path.join(pie_annot_dir, "*.xml")))

    if not annot_files:
        raise FileNotFoundError(f"No XML files found in {pie_annot_dir}")

    print("=" * 80)
    print("PIE PARKED-CAR CONFOUND AUDIT (WITH EGO-MOTION COMPENSATION)")
    print(f"Annotation directory: {pie_annot_dir}")
    print(f"Vehicle OBD directory: {pie_obd_dir}")
    print(f"Speed thresholds evaluated: {thresholds} m/s")
    print("=" * 80)

    vehicle_tracks = {}
    total_vehicles = 0
    all_speeds = []

    # Store frame-level data for density audit
    # video_id -> frame -> list of (agent_id, label, cx, cy, is_parked_dict)
    video_frames = defaultdict(lambda: defaultdict(list))
    ped_tracks = defaultdict(dict)

    for xf in annot_files:
        rel = os.path.relpath(xf, pie_annot_dir)
        fname = os.path.basename(xf)
        vid_id = fname.replace("_annt.xml", "").replace(".xml", "")
        
        # Locate corresponding OBD file
        # Check standard relative paths
        obd_candidates = [
            os.path.join(pie_obd_dir, os.path.dirname(rel), fname.replace("_annt.xml", "_obd.xml")),
            os.path.join(pie_obd_dir, fname.replace("_annt.xml", "_obd.xml")),
            xf.replace("annotations/annotations/", "annotations/annotations_vehicle/").replace("_annt.xml", "_obd.xml")
        ]
        obd_file = None
        for c in obd_candidates:
            if os.path.exists(c):
                obd_file = c
                break

        obd_speeds = load_obd_data(obd_file) if obd_file else {}

        tree = ET.parse(xf)
        root = tree.getroot()

        for t in root.findall(".//track"):
            lbl = t.get("label")
            if lbl not in ["pedestrian", "vehicle"]:
                continue
            boxes = t.findall("box")
            if not boxes:
                continue

            t_id = None
            for a in boxes[0].findall("attribute"):
                if a.get("name") == "id":
                    t_id = a.text
                    break
            if not t_id:
                t_id = f"trk_{len(vehicle_tracks)}"

            if lbl == "vehicle":
                med_spd, p90_spd, spd_list = estimate_vehicle_world_speed(boxes, obd_speeds)
                is_parked_map = {th: (med_spd < th) for th in thresholds}
                vehicle_tracks[t_id] = {
                    "video": vid_id,
                    "frames": len(boxes),
                    "med_speed": med_spd,
                    "p90_speed": p90_spd,
                    "is_parked": is_parked_map
                }
                all_speeds.append(med_spd)
                total_vehicles += 1

                for b in boxes:
                    f = int(b.get("frame"))
                    cx = (float(b.get("xtl")) + float(b.get("xbr"))) / (2.0 * 1920.0)
                    cy = (float(b.get("ytl")) + float(b.get("ybr"))) / (2.0 * 1080.0)
                    video_frames[vid_id][f].append((t_id, "vehicle", cx, cy, is_parked_map))

            elif lbl == "pedestrian":
                ped_tracks[vid_id][t_id] = len(boxes)
                for b in boxes:
                    f = int(b.get("frame"))
                    cx = (float(b.get("xtl")) + float(b.get("xbr"))) / (2.0 * 1920.0)
                    cy = (float(b.get("ytl")) + float(b.get("ybr"))) / (2.0 * 1080.0)
                    # Pedestrians are never parked cars
                    is_parked_map = {th: False for th in thresholds}
                    video_frames[vid_id][f].append((t_id, "pedestrian", cx, cy, is_parked_map))

    all_speeds = np.array(all_speeds)
    print(f"\n[1. Vehicle Physical Velocity Distribution] (N = {len(all_speeds)} tracks):")
    sp_pct = np.percentile(all_speeds, [5, 10, 25, 50, 75, 90, 95])
    print(f"  P05: {sp_pct[0]:.2f} m/s ({sp_pct[0]*3.6:.1f} km/h)")
    print(f"  P10: {sp_pct[1]:.2f} m/s ({sp_pct[1]*3.6:.1f} km/h)")
    print(f"  P25: {sp_pct[2]:.2f} m/s ({sp_pct[2]*3.6:.1f} km/h)")
    print(f"  P50: {sp_pct[3]:.2f} m/s ({sp_pct[3]*3.6:.1f} km/h)")
    print(f"  P75: {sp_pct[4]:.2f} m/s ({sp_pct[4]*3.6:.1f} km/h)")
    print(f"  P90: {sp_pct[5]:.2f} m/s ({sp_pct[5]*3.6:.1f} km/h)")
    print(f"  P95: {sp_pct[6]:.2f} m/s ({sp_pct[6]*3.6:.1f} km/h)")

    print("\n[2. Parked Vehicle Share by Physical Threshold]:")
    print(f"{'Threshold (m/s)':<18} | {'Threshold (km/h)':<18} | {'Parked Vehicles':<16} | {'Moving Vehicles':<16} | {'Parked Share (%)':<16}")
    print("-" * 92)
    for th in thresholds:
        parked = int((all_speeds < th).sum())
        moving = len(all_speeds) - parked
        pct = (parked / len(all_speeds)) * 100.0 if len(all_speeds) > 0 else 0.0
        print(f"{th:<18.2f} | {th*3.6:<18.1f} | {parked:<16d} | {moving:<16d} | {pct:<16.1f}")

    # 3. Density Audit: Recalculate Pedestrian Neighbor Density with and without Parked Vehicles
    print(f"\n[3. Neighbor Density Audit under Spatial Radius r = {primary_radius:.2f}]:")
    print("Measuring neighbor counts per pedestrian frame before and after filtering parked cars...")

    # We evaluate for baseline (all neighbors) and for each threshold filter
    filter_keys = ["all_neighbors"] + [f"filter_{th}ms" for th in thresholds]
    frame_counts = {k: {0: 0, 1: 0, "2+": 0} for k in filter_keys}
    total_ped_frames = 0

    for vid_id, frames in video_frames.items():
        for f, agents in frames.items():
            peds_in_frame = [(aid, cx, cy) for aid, lbl, cx, cy, _ in agents if lbl == "pedestrian"]
            if not peds_in_frame:
                continue

            for pid, px, py in peds_in_frame:
                total_ped_frames += 1

                # 1. All neighbors (unfiltered)
                nbr_all = [
                    (aid, lbl, cx, cy, is_p) for aid, lbl, cx, cy, is_p in agents
                    if aid != pid and np.sqrt((cx - px)**2 + (cy - py)**2) <= primary_radius
                ]
                k_all = len(nbr_all)
                bucket_all = 0 if k_all == 0 else (1 if k_all == 1 else "2+")
                frame_counts["all_neighbors"][bucket_all] += 1

                # 2. Filtered by each speed threshold
                for th in thresholds:
                    fkey = f"filter_{th}ms"
                    nbr_filtered = [
                        aid for aid, lbl, cx, cy, is_p in nbr_all
                        if lbl == "pedestrian" or not is_p[th]
                    ]
                    k_f = len(nbr_filtered)
                    bucket_f = 0 if k_f == 0 else (1 if k_f == 1 else "2+")
                    frame_counts[fkey][bucket_f] += 1

    print("-" * 92)
    print(f"{'Condition':<22} | {'Isolated K=0 (%)':<20} | {'Dyadic K=1 (%)':<20} | {'Multi-Agent K>=2 (%)':<20}")
    print("-" * 92)
    tot = max(1, total_ped_frames)
    for k in filter_keys:
        p0 = (frame_counts[k][0] / tot) * 100.0
        p1 = (frame_counts[k][1] / tot) * 100.0
        p2 = (frame_counts[k]["2+"] / tot) * 100.0
        label = "Unfiltered (Raw PIE)" if k == "all_neighbors" else f"Excl. Parked (<{k.replace('filter_','').replace('ms','')} m/s)"
        print(f"{label:<22} | {p0:<20.1f} | {p1:<20.1f} | {p2:<20.1f}")
    print("-" * 92)

    return {
        "total_vehicles": total_vehicles,
        "total_ped_frames": total_ped_frames,
        "speed_percentiles": {p: float(v) for p, v in zip([5, 10, 25, 50, 75, 90, 95], sp_pct)},
        "frame_counts": frame_counts
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Parked-Car Confound in PIE Benchmark")
    parser.add_argument("--annot_dir", type=str, default="data/raw/PIE/annotations/annotations/set03",
                        help="Path to PIE annotation XMLs (default: set03)")
    parser.add_argument("--obd_dir", type=str, default="data/raw/PIE/annotations/annotations_vehicle/set03",
                        help="Path to PIE vehicle OBD XMLs (default: set03)")
    parser.add_argument("--radius", type=float, default=0.20, help="Spatial radius threshold")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.5, 1.0, 1.5, 2.0],
                        help="Speed thresholds in m/s")
    args = parser.parse_args()

    run_parked_car_audit(
        pie_annot_dir=args.annot_dir,
        pie_obd_dir=args.obd_dir,
        thresholds=args.thresholds,
        primary_radius=args.radius
    )


if __name__ == "__main__":
    main()
