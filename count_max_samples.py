"""
Dry-run utility to calculate the EXACT maximum number of real sequences
extractable from local raw datasets (JAAD + PIE) without saving anything to disk.
"""

import os
import glob
import numpy as np

def count_max_jaad_samples(jaad_root="data/raw/JAAD", window_size=32, strides=[8, 4]):
    xml_dir = os.path.join(jaad_root, "annotations")
    if not os.path.isdir(xml_dir):
        return None

    import xml.etree.ElementTree as ET
    xml_files = sorted(glob.glob(os.path.join(xml_dir, "*.xml")))
    if not xml_files:
        return None

    total_tracks = 0
    valid_tracks = 0
    total_frames = 0
    windows_per_stride = {s: 0 for s in strides}

    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            for track in root.findall(".//track"):
                if track.get("label", "") != "pedestrian":
                    continue
                total_tracks += 1
                boxes = [b for b in track.findall("box") if b.get("outside", "0") != "1"]
                t_len = len(boxes)
                total_frames += t_len
                if t_len >= window_size:
                    valid_tracks += 1
                    for s in strides:
                        n_win = (t_len - window_size) // s + 1
                        windows_per_stride[s] += n_win
        except Exception:
            continue

    return {
        "xml_files": len(xml_files),
        "total_tracks": total_tracks,
        "valid_tracks": valid_tracks,
        "total_frames": total_frames,
        "windows": windows_per_stride
    }

def count_max_pie_samples(pie_root="data/raw/PIE", window_size=32, strides=[8, 4]):
    from datasets.pie_adapter import PIEAdapter
    if not os.path.isdir(pie_root):
        return None

    adapter = PIEAdapter(pie_root=pie_root, window_size=window_size)
    pie_data = adapter.load_annotations()
    if not pie_data:
        return None

    total_ped_tracks = 0
    valid_ped_tracks = 0
    windows_per_stride = {s: 0 for s in strides}

    for set_id, videos in pie_data.items():
        if not isinstance(videos, dict):
            continue
        for v_id, v_content in videos.items():
            peds = v_content.get("ped_annotations", {})
            for pid, pdata in peds.items():
                boxes = pdata.get("bbox", [])
                t_len = len(boxes)
                total_ped_tracks += 1
                if t_len >= window_size:
                    valid_ped_tracks += 1
                    for s in strides:
                        n_win = (t_len - window_size) // s + 1
                        windows_per_stride[s] += n_win

    return {
        "total_tracks": total_ped_tracks,
        "valid_tracks": valid_ped_tracks,
        "windows": windows_per_stride
    }

def main():
    print("=" * 70)
    print("MAXIMUM EXTRACTABLE SAMPLE CAPACITY AUDIT")
    print("=" * 70)

    jaad_res = count_max_jaad_samples()
    pie_res = count_max_pie_samples()

    total_stride_8 = 0
    total_stride_4 = 0

    if jaad_res:
        print("\n[JAAD Dataset]")
        print(f"  • Annotation XML files parsed: {jaad_res['xml_files']}")
        print(f"  • Total raw pedestrian tracks: {jaad_res['total_tracks']}")
        print(f"  • Valid tracks (>= 32 frames): {jaad_res['valid_tracks']}")
        print(f"  • Max sequences (Default Stride = 8): {jaad_res['windows'][8]:,}")
        print(f"  • Max sequences (Dense Stride = 4)  : {jaad_res['windows'][4]:,}")
        total_stride_8 += jaad_res['windows'][8]
        total_stride_4 += jaad_res['windows'][4]
    else:
        print("\n[JAAD Dataset]: No raw files found in data/raw/JAAD")

    if pie_res:
        print("\n[PIE Dataset]")
        print(f"  • Total pedestrian tracks    : {pie_res['total_tracks']}")
        print(f"  • Valid tracks (>= 32 frames): {pie_res['valid_tracks']}")
        print(f"  • Max sequences (Default Stride = 8): {pie_res['windows'][8]:,}")
        print(f"  • Max sequences (Dense Stride = 4)  : {pie_res['windows'][4]:,}")
        total_stride_8 += pie_res['windows'][8]
        total_stride_4 += pie_res['windows'][4]
    else:
        print("\n[PIE Dataset]: No raw files found in data/raw/PIE")

    print("\n" + "=" * 70)
    print("TOTAL MAXIMUM SAMPLES AVAILABLE ON THIS SYSTEM:")
    print(f"  >>> With Default Stride (s=8): {total_stride_8:,} total sequences")
    print(f"      • Train (70%): int({total_stride_8 * 0.7:,.0f}) sequences")
    print(f"      • Val   (15%): int({total_stride_8 * 0.15:,.0f}) sequences")
    print(f"      • Test  (15%): int({total_stride_8 * 0.15:,.0f}) sequences")
    print()
    print(f"  >>> With Dense Stride (s=4)  : {total_stride_4:,} total sequences")
    print("=" * 70)

if __name__ == "__main__":
    main()
