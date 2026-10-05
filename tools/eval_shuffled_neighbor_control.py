#!/usr/bin/env python3
"""
tools/eval_shuffled_neighbor_control.py: Shuffled-Neighbor Negative Control Audit.
Sanity check testing whether model reliance on neighbors is authentic to the specific scene constellation:
- Swaps neighbor constellations across unrelated pedestrian windows (batch permutation)
- Evaluates whether crossing AUC and output probabilities drop or shift significantly
- Clustered at the independent pedestrian track level
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats
from typing import Dict, List, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel
from evaluation.classification_metrics import compute_crossing_intention_metrics


def run_shuffled_neighbor_control(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    seed: int = 42
) -> Dict[str, Any]:
    model.eval()
    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("RUNNING SHUFFLED-NEIGHBOR NEGATIVE CONTROL AUDIT")
    print(f"Seed: {seed}")
    print("=" * 80)

    y_true = []
    probs_intact = []
    probs_shuffled = []
    track_diffs: Dict[str, List[float]] = {}

    with torch.no_grad():
        for batch in loader:
            B = batch["trajectory"].shape[0]
            if B < 2:
                continue

            batch_intact = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out_intact = model(batch_intact)
            p_int = F.softmax(out_intact["crossing_logits"], dim=-1)[:, 1].cpu().numpy()

            # Create shuffled neighbor control: permute neighbor tensors across unrelated samples in batch
            perm = rng.permutation(B)
            # Ensure no identity self-matches if B > 1
            if (perm == np.arange(B)).any():
                perm = (perm + 1) % B

            batch_shuffled = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch_intact.items()}
            batch_shuffled["neighbor_agents"] = batch_shuffled["neighbor_agents"][perm]
            batch_shuffled["neighbor_mask"] = batch_shuffled["neighbor_mask"][perm]

            out_shuf = model(batch_shuffled)
            p_shuf = F.softmax(out_shuf["crossing_logits"], dim=-1)[:, 1].cpu().numpy()

            targets = batch["cross"].cpu().numpy()
            y_true.extend(targets.tolist())
            probs_intact.extend(p_int.tolist())
            probs_shuffled.extend(p_shuf.tolist())

            diffs = np.abs(p_int - p_shuf)
            for b in range(B):
                tid = str(batch["track_id"][b])
                track_diffs.setdefault(tid, []).append(float(diffs[b]))

    metrics_intact = compute_crossing_intention_metrics(y_true, probs_intact)
    metrics_shuffled = compute_crossing_intention_metrics(y_true, probs_shuffled)

    track_medians = [float(np.median(v)) for v in track_diffs.values()]
    w_stat, p_val = stats.wilcoxon(track_medians, alternative="greater")

    print("\n" + "=" * 80)
    print("SHUFFLED-NEIGHBOR CONTROL AUDIT REPORT")
    print("=" * 80)
    print(f"Total Windows Evaluated:              {len(y_true):,}")
    print(f"Independent Track Clusters:           {len(track_medians)} tracks")
    print("-" * 80)
    print(f"Intact Neighbors Benchmark:")
    print(f"  Accuracy:                           {metrics_intact['accuracy']:.2f}%")
    print(f"  ROC-AUC:                            {metrics_intact['auc']:.2f}%")
    print(f"  F1-Score:                           {metrics_intact['f1']:.2f}%")
    print(f"Shuffled (Unrelated) Neighbors Control:")
    print(f"  Accuracy:                           {metrics_shuffled['accuracy']:.2f}%")
    print(f"  ROC-AUC:                            {metrics_shuffled['auc']:.2f}%")
    print(f"  F1-Score:                           {metrics_shuffled['f1']:.2f}%")
    print("-" * 80)
    print(f"Performance Gap (Intact - Shuffled):")
    print(f"  ΔAUC:                               {metrics_intact['auc'] - metrics_shuffled['auc']:+.2f}%")
    print(f"  ΔF1:                                {metrics_intact['f1'] - metrics_shuffled['f1']:+.2f}%")
    print(f"Track Median Output Displacement |ΔP|: {float(np.median(track_medians)):.4f}")
    print(f"One-Sided Wilcoxon Displacement Test: W = {w_stat:.1f}, p = {p_val:.5e}")
    print("=" * 80 + "\n")

    return {
        "metrics_intact": metrics_intact,
        "metrics_shuffled": metrics_shuffled,
        "delta_auc": metrics_intact["auc"] - metrics_shuffled["auc"],
        "delta_f1": metrics_intact["f1"] - metrics_shuffled["f1"],
        "median_displacement": float(np.median(track_medians)),
        "p_val": float(p_val)
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Shuffled-Neighbor Negative Control")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    p.add_argument("--data_dir", type=str, default="data/processed_pie", help="Path to processed dataset")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--frozen", action="store_true", help="Required flag when evaluating on held-out test split")
    args = p.parse_args()

    if args.split == "test" and not args.frozen:
        sys.exit("Refusing to execute on test split without --frozen flag.")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint} on {device}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = ProposedXMISTModel(
        feature_dim=config["model"]["feature_dim"],
        dropout=0.0
    ).to(device)

    state_dict = ckpt["model_state_dict"]
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)

    dataset = MultimodalSequenceDataset(
        data_dir=args.data_dir,
        split=args.split,
        window_size=config["data"]["window_size"]
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=2 if device.type == "cuda" else 0
    )

    run_shuffled_neighbor_control(
        model=model,
        loader=loader,
        device=device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
