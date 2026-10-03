from datasets.pie_adapter import PIEAdapter
import numpy as np

print("=" * 70)
print("TESTING SEQUENCE EXTRACTION ON SET 03 (TEST SET)")
print("=" * 70)

adapter = PIEAdapter(pie_root="data/raw/PIE", window_size=32, stride=4, distance_threshold=0.20)
pie_data = adapter.load_annotations()

# Extract 200 test sequences from Set 03
test_seqs = adapter.extract_sequences_from_annotations(pie_data, max_samples=200, set_filter=["set03"])
print(f"\nExtracted {len(test_seqs)} sample sequences from Set 03")

active_counts = [int(s["neighbor_mask"].sum()) for s in test_seqs]
active_seqs = sum(1 for c in active_counts if c > 0)
total_active_frames = sum(active_counts)

print(f"  • Sequences with >= 1 active neighbor: {active_seqs}/{len(test_seqs)} ({active_seqs / len(test_seqs) * 100:.1f}%)")
print(f"  • Total active neighbor frames:        {total_active_frames}")
print("=" * 70)
