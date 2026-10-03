import os
import glob
import xml.etree.ElementTree as ET
import numpy as np

# 1. Dataset stats
test_files = sorted(glob.glob("data/processed_pie/test/*.npz"))
train_files = sorted(glob.glob("data/processed_pie/train/*.npz"))
print(f"Total test files: {len(test_files)}")
print(f"Total train files: {len(train_files)}")

test_active = sum(int(np.load(f)["neighbor_mask"].sum()) for f in test_files[:500])
train_active = sum(int(np.load(f)["neighbor_mask"].sum()) for f in train_files[:500])
print(f"Active neighbor frames in 500 Test files:  {test_active}")
print(f"Active neighbor frames in 500 Train files: {train_active}")

# 2. Raw annotation audit
xml_files = sorted(glob.glob("data/raw/PIE/annotations/**/*.xml", recursive=True) + glob.glob("data/raw/PIE/annotations/*.xml"))
print(f"\nTotal raw XML files in data/raw/PIE/annotations: {len(xml_files)}")
if xml_files:
    print(f"First 3 XML files: {xml_files[:3]}")
    labels = set()
    for xf in xml_files[:10]:
        try:
            tree = ET.parse(xf)
            for track in tree.getroot().findall(".//track"):
                lbl = track.get("label")
                if lbl:
                    labels.add(lbl.lower())
        except Exception:
            pass
    print(f"Unique track labels found across first 10 XMLs: {labels}")

# Check if vehicle_annotations directory exists
veh_dirs = glob.glob("data/raw/PIE/*vehicle*") + glob.glob("data/raw/PIE/annotations/*vehicle*")
print(f"Vehicle annotation directories found: {veh_dirs}")
