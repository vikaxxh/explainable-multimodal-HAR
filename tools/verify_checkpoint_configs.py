#!/usr/bin/env python3
"""
tools/verify_checkpoint_configs.py: Side-by-side verification of training configs
and selection rules between Proposed and Isolated checkpoints.
"""
import os
import torch

checkpoints = [
    ("PROPOSED MODEL", "experiments/checkpoints/pie/checkpoint_best.pt"),
    ("ISOLATED BASELINE", "experiments/checkpoints/isolated_seed42/checkpoint_best.pt")
]

print("=" * 85)
print("TRAINING CONFIGURATION & CHECKPOINT SELECTION AUDIT")
print("=" * 85)

for label, path in checkpoints:
    print(f"\n--- {label} ({path}) ---")
    if not os.path.exists(path):
        print(f"  [ERROR] File not found: {path}")
        continue
    
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})
    tr = cfg.get("training", {})
    mdl = cfg.get("model", {})
    
    epoch = ckpt.get("epoch")
    best_metric = ckpt.get("best_metric_val")
    best_val_loss = ckpt.get("best_val_loss")
    early_metric = ckpt.get("early_stop_metric", tr.get("early_stop_metric", "val_loss (legacy)"))
    loss_weights = tr.get("loss_weights", "Not explicitly stored in checkpoint config")
    dropout = mdl.get("dropout", "default")
    weight_decay = tr.get("weight_decay", "default")
    lr = tr.get("lr", "default")
    
    print(f"  Saved Epoch:          {epoch}")
    print(f"  Checkpoint Metric:    {early_metric} = {best_metric}")
    print(f"  Best Val Loss:        {best_val_loss}")
    print(f"  Loss Weights:         {loss_weights}")
    print(f"  Dropout:              {dropout}")
    print(f"  Weight Decay:         {weight_decay}")
    print(f"  Learning Rate:        {lr}")

print("\n" + "=" * 85)
