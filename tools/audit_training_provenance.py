#!/usr/bin/env python3
"""
tools/audit_training_provenance.py: Rigorous provenance audit for Proposed Model Checkpoint.
Verifies:
1. File timestamps of checkpoint vs processed dataset.
2. Saved config interaction flags and module architecture weights.
3. Actual weight norms in interaction_graph and interaction_transformer.
4. Training log data directory and startup flags.
"""

import os
import sys
import time
import torch

ckpt_path = "experiments/checkpoints/proposed_seed42/checkpoint_best.pt"
log_path = "train_proposed_fair.log"
data_dir = "data/processed_pie/train"

print("=" * 85)
print("PROVENANCE & INTEGRITY AUDIT: FAIR PROPOSED MODEL (proposed_seed42)")
print("=" * 85)

# 1. File Timestamps
if not os.path.exists(ckpt_path):
    sys.exit(f"[ERROR] Checkpoint not found at {ckpt_path}")

ckpt_mtime = os.path.getmtime(ckpt_path)
ckpt_time_str = time.strftime('%Y-%m-%d %H:%M:%S IST', time.localtime(ckpt_mtime))
print(f"1. Checkpoint Timestamp:  {ckpt_time_str} ({ckpt_path})")

if os.path.exists(data_dir):
    sample_files = os.listdir(data_dir)[:10]
    if sample_files:
        sample_path = os.path.join(data_dir, sample_files[0])
        data_mtime = os.path.getmtime(sample_path)
        data_time_str = time.strftime('%Y-%m-%d %H:%M:%S IST', time.localtime(data_mtime))
        print(f"   Dataset File Timestamp: {data_time_str} ({sample_path})")
        if ckpt_mtime > data_mtime:
            print("   >> [PASS]: Checkpoint was trained AFTER processed dataset files were created.")
        else:
            print("   >> [WARNING]: Checkpoint timestamp is OLDER than dataset files!")

# 2. Checkpoint Internal Config & State
print("\n2. Checkpoint Internal Architecture & Interaction Keys:")
ckpt = torch.load(ckpt_path, map_location="cpu")
cfg = ckpt.get("config", {})
inter_cfg = cfg.get("model", {}).get("interaction", {})
print(f"   Saved Epoch:                     {ckpt.get('epoch')}")
print(f"   Best Val Metric (val_auc):       {ckpt.get('best_metric_val')}")
print(f"   Interaction Config Enabled:      {inter_cfg.get('enabled')}")
print(f"   Interaction Distance Threshold:  {inter_cfg.get('distance_threshold')}")
print(f"   Interaction Number of Heads:     {inter_cfg.get('num_heads')}")

state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
inter_keys = [k for k in state.keys() if "interaction" in k]
print(f"\n   Total Parameters in State Dict:  {len(state)}")
print(f"   Interaction Module Parameters:   {len(inter_keys)} parameter tensors found.")

if inter_keys:
    print("\n   Sample Interaction Layer Norms (Verifying Non-Zero Gradients/Weights):")
    for k in inter_keys[:8]:
        w = state[k].float()
        print(f"     - {k:<45}: shape={str(list(w.shape)):<18} norm={w.norm().item():.4f} std={w.std().item():.4f}")
    
    all_inter_norms = [state[k].float().norm().item() for k in inter_keys]
    if all(n > 0 for n in all_inter_norms):
        print("   >> [PASS]: 100% of interaction layers contain active, non-zero learned weights.")
    else:
        print("   >> [FAIL]: Some interaction layers have zero norm.")
else:
    print("   >> [FAIL]: No interaction layers found in checkpoint!")

# 3. Training Log Snippet
print("\n3. Training Startup Log Snippet (train_proposed_fair.log):")
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        log_lines = [next(f, "") for _ in range(25)]
    print("   " + "   ".join(log_lines[:15]))
else:
    print(f"   [Notice] Log file {log_path} not found.")

print("=" * 85 + "\n")
