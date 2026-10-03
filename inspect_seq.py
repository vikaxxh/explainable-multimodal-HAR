import glob
import os
import time
import numpy as np

print("=" * 70)
print("INSPECTING PROCESSED SEQUENCES ON DISK")
print("=" * 70)

test_files = sorted(glob.glob("data/processed_pie/test/*.npz"))
train_files = sorted(glob.glob("data/processed_pie/train/*.npz"))

if test_files:
    f0 = test_files[0]
    mtime = time.ctime(os.path.getmtime(f0))
    d0 = np.load(f0, allow_pickle=True)
    print(f"Sample test file: {f0}")
    print(f"File modification time: {mtime}")
    print(f"Keys in .npz: {d0.files}")
    if "metadata" in d0:
        print(f"Metadata: {d0['metadata'].item()}")
    print(f"neighbor_mask shape: {d0['neighbor_mask'].shape} | sum: {d0['neighbor_mask'].sum()}")
    print(f"neighbor_agents shape: {d0['neighbor_agents'].shape} | non-zero: {np.count_nonzero(d0['neighbor_agents'])}")

# Check across ALL test files (fast vectorized check)
total_active = 0
active_seqs = 0
for f in test_files[:2000]:
    d = np.load(f, allow_pickle=True)
    s = int(d["neighbor_mask"].sum())
    if s > 0:
        active_seqs += 1
        total_active += s

print(f"\nAcross first 2,000 test files:")
print(f"  • Sequences with >= 1 active neighbor frame: {active_seqs}")
print(f"  • Total active neighbor frames:              {total_active}")

# Also test a Set 03 XML file directly through PIEAdapter
set03_xmls = sorted(glob.glob("data/raw/PIE/annotations/annotations/set03/*.xml"))
print(f"\nSet 03 XML files found: {len(set03_xmls)}")
if set03_xmls:
    print(f"First Set 03 XML: {set03_xmls[0]}")
    from datasets.pie_adapter import PIEAdapter
    adapter = PIEAdapter(pie_root="data/raw/PIE", window_size=32, stride=4, distance_threshold=0.20)
    data = adapter._parse_pie_xml_files([set03_xmls[0]])
    vid = list(data.get("set03", {}).keys())[0] if data.get("set03") else "none"
    peds = data.get("set03", {}).get(vid, {}).get("ped_annotations", {})
    vehs = data.get("set03", {}).get(vid, {}).get("vehicle_annotations", {})
    print(f"Parsed {set03_xmls[0]}: {len(peds)} pedestrians, {len(vehs)} vehicles")
print("=" * 70)
