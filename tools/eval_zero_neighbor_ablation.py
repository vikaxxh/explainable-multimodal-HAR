#!/usr/bin/env python3
"""
tools/eval_zero_neighbor_ablation.py: Check 3B - Zero-Neighbor Ablation Audit.
Tests whether the Proposed Multi-Agent Model actually relies on neighbor inputs at inference time
by comparing intact predictions against an identical forward pass with all neighbors zeroed out.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel


def parse_args():
    parser = argparse.ArgumentParser(description="Check 3B: Zero-Neighbor Ablation on Proposed Model")
    parser.add_argument("--checkpoint", type=str, default="experiments/checkpoints/proposed_seed42/checkpoint_best.pt",
                        help="Path to proposed model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/processed_pie", help="Path to processed sequences")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--frozen", action="store_true", help="Guard flag required if evaluating on test split")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split == "test" and not args.frozen:
        sys.exit("[CRITICAL GUARD] Refusing to evaluate on test split without --frozen flag.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    
    print("=" * 85)
    print("CHECK 3B: ZERO-NEIGHBOR COUNTERFACTUAL ABLATION AUDIT")
    print(f"Model Checkpoint:  {args.checkpoint}")
    print(f"Evaluation Split:  {args.split.upper()}")
    print(f"Device:            {device}")
    print("=" * 85)

    # 1. Load Model
    model = ProposedXMISTModel(feature_dim=config["model"]["feature_dim"], dropout=0.0).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=False)
    model.eval()

    # 2. Load Dataset
    dataset = MultimodalSequenceDataset(data_dir=args.data_dir, split=args.split, window_size=config["data"]["window_size"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch, num_workers=2 if device.type == "cuda" else 0)

    y_true = []
    probs_intact = []
    probs_zeroed = []
    abs_deltas = []

    print(f"Running paired forward passes across {len(dataset):,} windows...")
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            # Intact pass
            batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out_intact = model(batch_gpu)
            p_int = F.softmax(out_intact["crossing_logits"], dim=-1)[:, 1].cpu().numpy()

            # Zeroed pass (Force all neighbor slots and masks to zero)
            batch_zeroed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_gpu.items()}
            batch_zeroed["neighbor_mask"] = torch.zeros_like(batch_gpu["neighbor_mask"])
            batch_zeroed["neighbor_agents"] = torch.zeros_like(batch_gpu["neighbor_agents"])
            out_zeroed = model(batch_zeroed)
            p_zero = F.softmax(out_zeroed["crossing_logits"], dim=-1)[:, 1].cpu().numpy()

            targets = batch["cross"].cpu().numpy() if torch.is_tensor(batch["cross"]) else np.asarray(batch["cross"])
            y_true.extend(targets.tolist())
            probs_intact.extend(p_int.tolist())
            probs_zeroed.extend(p_zero.tolist())
            abs_deltas.extend(np.abs(p_int - p_zero).tolist())

    y_true = np.array(y_true)
    probs_intact = np.array(probs_intact)
    probs_zeroed = np.array(probs_zeroed)
    abs_deltas = np.array(abs_deltas)

    # Compute Metrics
    auc_intact = roc_auc_score(y_true, probs_intact) * 100.0
    auc_zeroed = roc_auc_score(y_true, probs_zeroed) * 100.0
    prauc_intact = average_precision_score(y_true, probs_intact) * 100.0
    prauc_zeroed = average_precision_score(y_true, probs_zeroed) * 100.0
    f1_intact = f1_score(y_true, (probs_intact >= 0.5).astype(int)) * 100.0
    f1_zeroed = f1_score(y_true, (probs_zeroed >= 0.5).astype(int)) * 100.0

    print("\n" + "=" * 85)
    print("CHECK 3B: ZERO-NEIGHBOR COUNTERFACTUAL RESULTS")
    print("=" * 85)
    print(f"{'Condition':<28} | {'ROC-AUC (%)':<14} | {'PR-AUC (%)':<14} | {'F1-Score (%)':<14}")
    print("-" * 85)
    print(f"{'Intact Inputs (Normal)':<28} | {auc_intact:<14.2f} | {prauc_intact:<14.2f} | {f1_intact:<14.2f}")
    print(f"{'Zeroed Neighbors (Masked)':<28} | {auc_zeroed:<14.2f} | {prauc_zeroed:<14.2f} | {f1_zeroed:<14.2f}")
    print(f"{'Ablation Drop (Δ = Intact - Zero)':<28} | {auc_intact - auc_zeroed:<+14.2f} | {prauc_intact - prauc_zeroed:<+14.2f} | {f1_intact - f1_zeroed:<+14.2f}")
    print("-" * 85)
    print("Per-Window Counterfactual Probability Shift (|P_intact - P_zeroed|):")
    print(f"  Median Absolute Shift:  {np.median(abs_deltas):.5f} ({np.median(abs_deltas)*100:.3f}%)")
    print(f"  Mean Absolute Shift:    {np.mean(abs_deltas):.5f} ({np.mean(abs_deltas)*100:.3f}%)")
    print(f"  90th Percentile Shift:  {np.percentile(abs_deltas, 90):.5f} ({np.percentile(abs_deltas, 90)*100:.3f}%)")
    print(f"  Max Absolute Shift:     {np.max(abs_deltas):.5f} ({np.max(abs_deltas)*100:.3f}%)")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
