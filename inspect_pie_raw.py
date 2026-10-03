import os
import xml.etree.ElementTree as ET
from collections import Counter

print("--- 1. Track labels inside video_0001_annt.xml ---")
xml_path = "data/raw/PIE/annotations/annotations/set01/video_0001_annt.xml"
if os.path.exists(xml_path):
    tree = ET.parse(xml_path)
    counts = Counter(t.get("label") for t in tree.getroot().iter("track"))
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
else:
    print(f"File not found: {xml_path}")

print("\n--- 2. Contents of data/raw/PIE/annotations/annotations_vehicle/ ---")
veh_path = "data/raw/PIE/annotations/annotations_vehicle"
if os.path.exists(veh_path):
    items = sorted(os.listdir(veh_path))
    print(f"Total entries: {len(items)}")
    for item in items[:15]:
        sub = os.path.join(veh_path, item)
        print(f"  {item} {'(dir)' if os.path.isdir(sub) else '(file)'}")
else:
    print(f"Directory not found: {veh_path}")
