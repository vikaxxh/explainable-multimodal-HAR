import os
import xml.etree.ElementTree as ET
import numpy as np
from datasets.pie_adapter import PIEAdapter

print("=" * 70)
print("TRACE PIE MATCHING ON video_0001_annt.xml")
print("=" * 70)

# 1. Check annotations_vehicle/set01
veh_set01 = "data/raw/PIE/annotations/annotations_vehicle/set01"
if os.path.exists(veh_set01):
    print("Files in annotations_vehicle/set01:", os.listdir(veh_set01)[:5])
else:
    print(f"Directory not found: {veh_set01}")

# 2. Parse single XML file using PIEAdapter
adapter = PIEAdapter(pie_root="data/raw/PIE", window_size=32, stride=4, distance_threshold=0.20)
xf = "data/raw/PIE/annotations/annotations/set01/video_0001_annt.xml"
data = adapter._parse_pie_xml_files([xf])

peds = data.get("set01", {}).get("video_0001", {}).get("ped_annotations", {})
vehs = data.get("set01", {}).get("video_0001", {}).get("vehicle_annotations", {})
print(f"Extracted ped_tracks: {len(peds)} | vehicle_tracks: {len(vehs)}")

# 3. Trace candidate neighbor matching
reasons = {"missing_frames": 0, "distance_too_large": 0, "accepted": 0}
min_distances = []

for pid, pdata in peds.items():
    boxes = pdata["bbox"]
    frames = pdata["frames"]
    if len(boxes) < 32:
        continue
    for start in range(0, len(boxes) - 32 + 1, 4):
        win_frames = frames[start:start + 32]
        if win_frames[-1] - win_frames[0] != 31:
            continue
        c_p = np.array([[(b[0] + b[2] / 2.0) / 1920.0, (b[1] + b[3] / 2.0) / 1080.0] for b in boxes[start:start + 32]])
        for vid, vdata in vehs.items():
            v_dict = {f: b for f, b in zip(vdata["frames"], vdata["bbox"])}
            if not all(f in v_dict for f in win_frames):
                reasons["missing_frames"] += 1
            else:
                c_v = np.array([[(b[0] + b[2] / 2.0) / 1920.0, (b[1] + b[3] / 2.0) / 1080.0] for b in [v_dict[f] for f in win_frames]])
                d = float(np.mean(np.linalg.norm(c_p - c_v, axis=-1)))
                min_distances.append(d)
                if d <= 0.20:
                    reasons["accepted"] += 1
                else:
                    reasons["distance_too_large"] += 1

print("\nNeighbor candidate rejection analysis for video_0001:")
print(f"  • Rejected because vehicle missing in >= 1 of 32 frames: {reasons['missing_frames']:,}")
print(f"  • Vehicle in all 32 frames, but distance > 0.20:        {reasons['distance_too_large']:,}")
print(f"  • Valid neighbors accepted (all 32 frames & d <= 0.20):   {reasons['accepted']:,}")
if min_distances:
    print(f"  • Mean distance across 32-frame cotemporal pairs:     {np.mean(min_distances):.3f}")
    print(f"  • Minimum distance found across any pair:             {np.min(min_distances):.3f}")
print("=" * 70)
