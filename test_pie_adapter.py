from datasets.pie_adapter import PIEAdapter

print("=" * 70)
print("TESTING FULL PIE ADAPTER LOAD")
print("=" * 70)

adapter = PIEAdapter(pie_root="data/raw/PIE", window_size=32, stride=4)
data = adapter.load_annotations()
print(f"Loaded sets: {list(data.keys())}")

for s in sorted(data.keys()):
    vids = data[s]
    total_peds = sum(len(v["ped_annotations"]) for v in vids.values())
    total_vehs = sum(len(v["vehicle_annotations"]) for v in vids.values())
    print(f"  • {s}: {len(vids):2d} videos | {total_peds:4d} pedestrian tracks | {total_vehs:4d} vehicle tracks")
print("=" * 70)
