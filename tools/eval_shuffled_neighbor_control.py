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

    print("=" * 85)
    print("RUNNING SHUFFLED-NEIGHBOR NEGATIVE CONTROL AUDIT")
    print(f"Seed: {seed}")
    print("=" * 85)

    y_true = []
    probs_intact = []
    probs_shuffled = []
    abs_deltas = []
    track_diffs: Dict[str, List[float]] = {}

    with torch.no_grad():
        for batch in loader:
            B = batch["trajectory"].shape[0]
            if B < 2:
                continue

            batch_intact = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            out_intact = model(batch_intact)
            if getattr(model, "_use_ped_head_fallback", False):
                p_int = F.softmax(out_intact["ped_logits"], dim=-1)[:, 5].cpu().numpy()
            else:
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
            if getattr(model, "_use_ped_head_fallback", False):
                p_shuf = F.softmax(out_shuf["ped_logits"], dim=-1)[:, 5].cpu().numpy()
            else:
                p_shuf = F.softmax(out_shuf["crossing_logits"], dim=-1)[:, 1].cpu().numpy()

            targets = batch["cross"].cpu().numpy() if torch.is_tensor(batch["cross"]) else np.asarray(batch["cross"])
            y_true.extend(targets.tolist())
            probs_intact.extend(p_int.tolist())
            probs_shuffled.extend(p_shuf.tolist())

            diffs = np.abs(p_int - p_shuf)
            abs_deltas.extend(diffs.tolist())
            for b in range(B):
                tid = str(batch["track_id"][b])
                track_diffs.setdefault(tid, []).append(float(diffs[b]))

    metrics_intact = compute_crossing_intention_metrics(y_true, probs_intact)
    metrics_shuffled = compute_crossing_intention_metrics(y_true, probs_shuffled)

    abs_deltas = np.array(abs_deltas)
    median_shift = float(np.median(abs_deltas))
    mean_shift = float(np.mean(abs_deltas))
    p90_shift = float(np.percentile(abs_deltas, 90))
    max_shift = float(np.max(abs_deltas))

    track_medians = [float(np.median(v)) for v in track_diffs.values()]
    w_stat, p_val = stats.wilcoxon(track_medians, alternative="greater")

    delta_auc = metrics_intact["auc"] - metrics_shuffled["auc"]
    delta_prauc = metrics_intact["pr_auc"] - metrics_shuffled["pr_auc"]
    delta_f1 = metrics_intact["f1"] - metrics_shuffled["f1"]
    delta_brier = metrics_shuffled["brier_score"] - metrics_intact["brier_score"]

    print("\n" + "=" * 85)
    print("CHECK 3C: SHUFFLED-NEIGHBOR NEGATIVE CONTROL RESULTS")
    print("=" * 85)
    print(f"Total Windows Evaluated:              {len(y_true):,}")
    print(f"Independent Track Clusters:           {len(track_medians)} tracks")
    print("-" * 85)
    print(f"{'Condition':<30} | {'ROC-AUC (%)':<14} | {'PR-AUC (%)':<14} | {'F1-Score (%)':<14} | {'Brier':<8}")
    print("-" * 85)
    print(f"{'Intact Inputs (Normal)':<30} | {metrics_intact['auc']:<14.2f} | {metrics_intact['pr_auc']:<14.2f} | {metrics_intact['f1']:<14.2f} | {metrics_intact['brier_score']:<8.4f}")
    print(f"{'Shuffled Neighbors (Random)':<30} | {metrics_shuffled['auc']:<14.2f} | {metrics_shuffled['pr_auc']:<14.2f} | {metrics_shuffled['f1']:<14.2f} | {metrics_shuffled['brier_score']:<8.4f}")
    print(f"{'Gap (Δ = Intact - Shuffled)':<30} | {delta_auc:<+14.2f} | {delta_prauc:<+14.2f} | {delta_f1:<+14.2f} | {delta_brier:<+8.4f}")
    print("-" * 85)
    print("Per-Window Shuffled Counterfactual Probability Shift (|P_intact - P_shuffled|):")
    print(f"  Median Absolute Shift:  {median_shift:.5f} ({median_shift * 100:.3f}%)")
    print(f"  Mean Absolute Shift:    {mean_shift:.5f} ({mean_shift * 100:.3f}%)")
    print(f"  90th Percentile Shift:  {p90_shift:.5f} ({p90_shift * 100:.3f}%)")
    print(f"  Max Absolute Shift:     {max_shift:.5f} ({max_shift * 100:.3f}%)")
    print("-" * 85)
    print(f"Track Median Output Displacement |ΔP|: {float(np.median(track_medians)):.5f}")
    print(f"One-Sided Wilcoxon Displacement Test: W = {w_stat:.1f}, p = {p_val:.5e}")
    print("=" * 85 + "\n")

    return {
        "metrics_intact": metrics_intact,
        "metrics_shuffled": metrics_shuffled,
        "delta_auc": delta_auc,
        "delta_prauc": delta_prauc,
        "delta_f1": delta_f1,
        "median_displacement": float(np.median(track_medians)),
        "p_val": float(p_val),
        "y_true": np.array(y_true),
        "probs_intact": np.array(probs_intact),
        "probs_shuffled": np.array(probs_shuffled)
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Shuffled-Neighbor Negative Control")
    p.add_argument("--checkpoint", type=str, default="experiments/checkpoints/proposed_seed42/checkpoint_best.pt",
                   help="Path to proposed model checkpoint")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    p.add_argument("--data_dir", type=str, default="data/processed_pie", help="Path to processed dataset")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output_file", type=str, default="", help="Optional path to save npz results")
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

    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    load_res = model.load_state_dict(clean_state, strict=False)
    has_crossing_head = any(k.startswith("crossing_head") for k in clean_state.keys())
    if not has_crossing_head:
        print("[Notice] Checkpoint does not contain 'crossing_head'; using pedestrian behavior Class 5 ('Crossing') logits.")
        model._use_ped_head_fallback = True
    else:
        model._use_ped_head_fallback = False
    if load_res.missing_keys:
        print(f"[Model Loader] Loaded with {len(load_res.missing_keys)} missing optional keys (e.g. auxiliary heads).")

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

    results = run_shuffled_neighbor_control(
        model=model,
        loader=loader,
        device=device,
        seed=args.seed
    )

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        np.savez_compressed(
            args.output_file,
            y_true=results["y_true"],
            probs_intact=results["probs_intact"],
            probs_shuffled=results["probs_shuffled"]
        )
        print(f"Saved shuffled control arrays to: {args.output_file}")


if __name__ == "__main__":
    main()
