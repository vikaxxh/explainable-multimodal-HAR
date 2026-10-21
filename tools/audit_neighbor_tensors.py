#!/usr/bin/env python3
"""
tools/audit_neighbor_tensors.py: Comprehensive neighbor tensor audit across processed PIE splits.
Verifies that .npz sequence files contain non-zero neighbor data at both frame and window levels.
"""
import os
import glob
import numpy as np

splits = ["train", "val", "test"]
base_dir = "data/processed_pie"

print("=" * 85)
print("PROCESSED NEIGHBOR TENSOR AUDIT: data/processed_pie")
print("=" * 85)

for split in splits:
    split_path = os.path.join(base_dir, split)
    files = sorted(glob.glob(os.path.join(split_path, "*.npz")))
    total_files = len(files)
    
    if total_files == 0:
        print(f"\n[Split: {split.upper()}] No .npz files found in {split_path}")
        continue
    
    # Audit full split or representative random sample of up to 1,000 files
    sample_size = min(total_files, 1000)
    indices = np.linspace(0, total_files - 1, sample_size, dtype=int)
    
    windows_with_neighbors = 0
    k0_windows = 0
    k1_windows = 0
    k2_plus_windows = 0
    
    total_active_frames = 0
    total_frame_slots = 0
    max_k_seen = 0
    
    for idx in indices:
        f = files[idx]
        with np.load(f, allow_pickle=True) as data:
            if "neighbor_mask" not in data:
                continue
            mask = data["neighbor_mask"] # Shape: (T, K)
            
            # Window-level check
            has_nbr = bool(mask.any())
            if has_nbr:
                windows_with_neighbors += 1
            
            # Frame-level active count at final timestep (t = T - 1)
            final_k = int(mask[-1].sum())
            if final_k == 0:
                k0_windows += 1
            elif final_k == 1:
                k1_windows += 1
            else:
                k2_plus_windows += 1
                
            # Full sequence active slots
            total_active_frames += int(mask.sum())
            total_frame_slots += int(mask.size)
            max_k_seen = max(max_k_seen, int(mask.sum(axis=-1).max()))
            
    pct_windows = (windows_with_neighbors / sample_size) * 100.0
    pct_k0 = (k0_windows / sample_size) * 100.0
    pct_k1 = (k1_windows / sample_size) * 100.0
    pct_k2 = (k2_plus_windows / sample_size) * 100.0
    slot_density = (total_active_frames / max(1, total_frame_slots)) * 100.0
    
    print(f"\n[Split: {split.upper()}] Audited {sample_size:,} of {total_files:,} files")
    print(f"  Windows with >=1 Active Neighbor (any frame):  {windows_with_neighbors:,}/{sample_size:,} ({pct_windows:.1f}%)")
    print(f"  Final Frame Density (t = T-1):")
    print(f"    - K = 0 (Isolated):                          {k0_windows:,} ({pct_k0:.1f}%)")
    print(f"    - K = 1 (Dyadic):                            {k1_windows:,} ({pct_k1:.1f}%)")
    print(f"    - K >= 2 (Dense Multi-Agent):                {k2_plus_windows:,} ({pct_k2:.1f}%)")
    print(f"  Overall (T x K) Mask Slot Density:             {slot_density:.2f}% active")
    print(f"  Maximum Concurrent Neighbors in Single Frame:  {max_k_seen}")

print("\n" + "=" * 85)
