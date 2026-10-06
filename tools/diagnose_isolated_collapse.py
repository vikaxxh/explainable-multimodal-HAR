#!/usr/bin/env python3
"""
tools/diagnose_isolated_collapse.py: Root-Cause Diagnostic Engine for Isolated Baseline Collapse.
Executes 4 precise checks on the isolated baseline:
1. Raw output probability distribution on validation batch (detects constant/tied collapse).
2. NaN / Inf audit across intermediate feature tensors (f_multi, z, logits).
3. Per-module gradient norm audit (verifies gradient flow through all encoders).
4. Loss function class-weighting inspection (verifies alpha balance in Focal Loss).
"""

import os
import sys
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader

from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel
from training.losses import MultiTaskBehaviorLoss


def run_diagnostics():
    print("=" * 75)
    print("ROOT-CAUSE DIAGNOSTIC AUDIT: ISOLATED BASELINE COLLAPSE")
    print("=" * 75)

    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Dataset (1 batch of real training data)
    print("\n[1/4] Loading sample batch from data/processed_pie/train...")
    dataset = MultimodalSequenceDataset(
        data_dir="data/processed_pie",
        split="train",
        window_size=config["data"]["window_size"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_multimodal_batch
    )
    batch = next(iter(dataloader))
    batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    y_true = batch_gpu["cross"].cpu().numpy()
    print(f"Batch Ground Truth Crossing Labels: {y_true.tolist()} (Crossing Count: {np.sum(y_true)}/16)")

    # 2. Inspect Checkpoint Output Distribution
    ckpt_path = "experiments/checkpoints/isolated_seed42/checkpoint_best.pt"
    print(f"\n[2/4] Inspecting Saved Checkpoint ({ckpt_path})...")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f"      Saved Epoch: {ckpt.get('epoch')}")
        print(f"      Recorded Best Metric: {ckpt.get('best_metric_val')}")

        model_ckpt = ProposedXMISTModel(feature_dim=256, use_interaction=False, dropout=0.3).to(device)
        clean_state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
        model_ckpt.load_state_dict(clean_state, strict=False)
        model_ckpt.eval()

        with torch.no_grad():
            out_ckpt = model_ckpt(batch_gpu)
            c_logits = out_ckpt["crossing_logits"]
            probs = torch.softmax(c_logits, dim=-1)[:, 1].cpu().numpy()
            p_logits = out_ckpt["ped_logits"]
            ped_preds = torch.argmax(p_logits, dim=-1).cpu().numpy()

        print(f"      Raw Crossing Logits:\n{c_logits.cpu().numpy()[:5]}")
        print(f"      Softmax Crossing Probs: min={probs.min():.5f}, max={probs.max():.5f}, mean={probs.mean():.5f}, std={probs.std():.8f}")
        print(f"      Crossing Probs Array: {np.round(probs, 4).tolist()}")
        print(f"      Ped Action Predictions: {ped_preds.tolist()}")

        if probs.std() < 1e-6:
            print("      >> [CRITICAL FINDING]: Output probabilities are 100% IDENTICAL across all samples!")
            print("         This mathematically causes ROC-AUC to equal exactly 50.00% (tied rankings).")
        else:
            print(f"      Probabilities vary: std = {probs.std():.6f}")

    # 3. Check Forward Pass NaN / Inf Across Architecture
    print("\n[3/4] Intermediate Feature Audit (Testing for NaN / Inf)...")
    model = ProposedXMISTModel(feature_dim=256, use_interaction=False, dropout=0.3).to(device)
    model.train()

    out = model(batch_gpu)
    for k in ["f_multi", "latent_z", "crossing_logits", "ped_logits", "action_logits"]:
        if k in out and out[k] is not None:
            t = out[k]
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            print(f"      Tensor '{k:<16}': shape={str(list(t.shape)):<16} NaN={has_nan} Inf={has_inf} mean={t.mean().item():.4f} std={t.std().item():.4f}")

    # 4. Check Gradient Flow & Magnitude per Submodule
    print("\n[4/4] Gradient Flow Audit across Modules...")
    criterion = MultiTaskBehaviorLoss(use_focal_loss=True, gamma=2.0)
    loss_dict = criterion(out, batch_gpu)
    loss = loss_dict["loss_total"]
    print(f"      Loss Total: {loss.item():.4f}")
    for k, v in loss_dict.items():
        if k != "loss_total":
            print(f"        {k}: {v.item():.4f}")

    loss.backward()

    module_grads = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            mod_prefix = name.split(".")[0]
            g_norm = param.grad.norm().item() if param.grad is not None else 0.0
            if mod_prefix not in module_grads:
                module_grads[mod_prefix] = []
            module_grads[mod_prefix].append(g_norm)

    print("\n      Module-level Gradient Norms (Mean | Max):")
    for mod, norms in sorted(module_grads.items()):
        mean_g = np.mean(norms)
        max_g = np.max(norms)
        status = "HEALTHY" if mean_g > 1e-4 else ("VANISHING" if mean_g > 0 else "DEAD (ZERO)")
        print(f"        {mod:<22}: mean={mean_g:.6f}, max={max_g:.6f} [{status}]")

    # 5. Check Loss Class Imbalance Term
    print("\n" + "=" * 75)
    print("FOCAL LOSS CLASS-BALANCE INSPECTION:")
    print("=" * 75)
    focal = criterion.crit_cross
    has_alpha = hasattr(focal, "alpha")
    print(f"Focal Loss Gamma:          {focal.gamma}")
    print(f"Focal Loss Label Smoothing: {focal.label_smoothing}")
    print(f"Has Class Weight (Alpha):  {has_alpha} (Actual alpha: {getattr(focal, 'alpha', None)})")
    if not has_alpha or getattr(focal, "alpha", None) is None:
        print(">> [CRITICAL FINDING]: Focal Loss lacks class weight alpha!")
        print("   On an ~82.6% negative dataset without alpha, Focal Loss penalizes majority-class false positives")
        print("   at (1 - 0.826)^2 = 0.03x, allowing the model to minimize loss by collapsing entirely into the majority class.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_diagnostics()
