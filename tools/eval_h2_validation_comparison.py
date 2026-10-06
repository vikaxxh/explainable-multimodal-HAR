#!/usr/bin/env python3
"""
tools/eval_h2_validation_comparison.py: Paired Track-Clustered Evaluation for Hypothesis H2.

Compares Proposed (Relational Graph) vs. Certified Isolated Baseline on Validation Split (239 Tracks):
- Computes ROC-AUC, PR-AUC (Average Precision), F1, Accuracy, Precision, Recall, Brier Score.
- Computes Point Estimate ΔAUC and ΔPR-AUC.
- Executes 1,000 Paired Track-Clustered Bootstrap iterations across the 239 validation tracks.
- Reports 95% Confidence Intervals and exact two-sided p-values.
- Evaluates against the pre-registered Minimum Meaningful Effect Size (MMES: ΔAUC >= +0.015).
- Saves full per-window predictions to .npz for auditable provenance.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch
import yaml
from typing import Dict, Any, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score, brier_score_loss
from torch.utils.data import DataLoader

from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel


def parse_args():
    parser = argparse.ArgumentParser(description="Paired Track-Clustered H2 Comparison on Validation Split")
    parser.add_argument("--proposed_ckpt", type=str, default="experiments/checkpoints/proposed_seed42/checkpoint_best.pt",
                        help="Path to proposed model checkpoint")
    parser.add_argument("--isolated_ckpt", type=str, default="experiments/checkpoints/isolated_seed42/checkpoint_best.pt",
                        help="Path to isolated baseline checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, default="data/processed_pie", help="Processed data directory")
    parser.add_argument("--split", type=str, default="val", help="Split to evaluate ('val' only; test is quarantined)")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of track-clustered bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap reproducibility")
    parser.add_argument("--output_file", type=str, default="experiments/results/val_h2_comparison_seed42.npz",
                        help="Path to save raw validation predictions and metrics")
    parser.add_argument("--frozen", action="store_true", help="Guard flag (only required if evaluating test split)")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_checkpoint(ckpt_path: str, use_interaction: bool, feature_dim: int, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    model = ProposedXMISTModel(
        feature_dim=feature_dim,
        use_interaction=use_interaction,
        dropout=0.0
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    load_res = model.load_state_dict(clean_state, strict=False)

    has_crossing_head = any(k.startswith("crossing_head") for k in clean_state.keys())
    model._use_ped_head_fallback = not has_crossing_head

    model.eval()
    return model, ckpt.get("epoch", "unknown"), ckpt.get("best_metric_val", None)


@torch.no_grad()
def collect_model_predictions(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs batched inference and returns (y_true, y_probs, track_ids).
    """
    all_targets = []
    all_probs = []
    all_tracks = []

    for batch in dataloader:
        raw = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model(raw)

        if getattr(model, "_use_ped_head_fallback", False):
            # Fallback for older models: class 5 is Crossing
            probs = torch.softmax(out["ped_logits"], dim=-1)[:, 5].cpu().numpy()
        else:
            probs = torch.softmax(out["crossing_logits"], dim=-1)[:, 1].cpu().numpy()

        targets = batch["cross"].cpu().numpy() if torch.is_tensor(batch["cross"]) else np.asarray(batch["cross"])
        tracks = batch["track_id"] if isinstance(batch["track_id"], list) else batch["track_id"].tolist()

        all_probs.extend(probs.tolist())
        all_targets.extend(targets.tolist())
        all_tracks.extend(tracks)

    return np.asarray(all_targets, dtype=int), np.asarray(all_probs, dtype=float), np.asarray(all_tracks)


def compute_metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred)) * 100.0
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    p *= 100.0
    r *= 100.0
    f1 *= 100.0

    try:
        auc = float(roc_auc_score(y_true, y_prob)) * 100.0
        pr_auc = float(average_precision_score(y_true, y_prob)) * 100.0
    except Exception:
        auc = 50.0
        pr_auc = float(np.mean(y_true)) * 100.0

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = 0.0

    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "brier": brier
    }


def run_track_clustered_bootstrap(
    y_true: np.ndarray,
    probs_prop: np.ndarray,
    probs_iso: np.ndarray,
    track_ids: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes paired track-clustered bootstrap across unique pedestrian tracks.
    """
    rng = np.random.RandomState(seed)
    unique_tracks = np.unique(track_ids)
    num_tracks = len(unique_tracks)

    # Pre-index windows by track for speed
    track_to_indices = {t: np.where(track_ids == t)[0] for t in unique_tracks}

    boot_delta_auc = []
    boot_delta_prauc = []
    boot_auc_prop = []
    boot_auc_iso = []

    for _ in range(n_bootstraps):
        resampled_tracks = rng.choice(unique_tracks, size=num_tracks, replace=True)
        idx_b = np.concatenate([track_to_indices[t] for t in resampled_tracks])

        y_b = y_true[idx_b]
        # Skip resamples that lack binary diversity (extremely rare with 239 tracks)
        if len(np.unique(y_b)) < 2:
            continue

        p_prop_b = probs_prop[idx_b]
        p_iso_b = probs_iso[idx_b]

        auc_p = float(roc_auc_score(y_b, p_prop_b)) * 100.0
        auc_i = float(roc_auc_score(y_b, p_iso_b)) * 100.0
        prauc_p = float(average_precision_score(y_b, p_prop_b)) * 100.0
        prauc_i = float(average_precision_score(y_b, p_iso_b)) * 100.0

        boot_auc_prop.append(auc_p)
        boot_auc_iso.append(auc_i)
        boot_delta_auc.append(auc_p - auc_i)
        boot_delta_prauc.append(prauc_p - prauc_i)

    boot_delta_auc = np.array(boot_delta_auc)
    boot_delta_prauc = np.array(boot_delta_prauc)

    ci_delta_auc = np.percentile(boot_delta_auc, [2.5, 97.5])
    ci_delta_prauc = np.percentile(boot_delta_prauc, [2.5, 97.5])

    # Empirical two-sided bootstrap p-value for H2: delta_auc <= 0
    p_val_auc = 2.0 * min(np.mean(boot_delta_auc <= 0.0), np.mean(boot_delta_auc >= 0.0))
    p_val_auc = max(min(p_val_auc, 1.0), 1.0 / n_bootstraps)

    return {
        "num_tracks": num_tracks,
        "ci_delta_auc": ci_delta_auc,
        "ci_delta_prauc": ci_delta_prauc,
        "bootstrap_mean_delta_auc": float(np.mean(boot_delta_auc)),
        "bootstrap_std_delta_auc": float(np.std(boot_delta_auc)),
        "bootstrap_mean_delta_prauc": float(np.mean(boot_delta_prauc)),
        "bootstrap_std_delta_prauc": float(np.std(boot_delta_prauc)),
        "p_value_auc": p_val_auc,
        "boot_delta_auc": boot_delta_auc,
        "boot_delta_prauc": boot_delta_prauc
    }


def main():
    args = parse_args()

    # Pre-registration quarantine check
    if args.split == "test" and not args.frozen:
        sys.exit(
            "\n[CRITICAL ERROR] Refusing to evaluate test split (Set 03) without --frozen flag.\n"
            "Pre-registration protocol strictly quarantines Set 03 until validation comparison is confirmed.\n"
        )

    print("=" * 80)
    print("HYPOTHESIS H2 EMPIRICAL AUDIT: PROPOSED MODEL VS. ISOLATED BASELINE")
    print("=" * 80)
    print(f"Evaluation Split:        {args.split.upper()} (Quarantining held-out test split)")
    print(f"Proposed Checkpoint:     {args.proposed_ckpt}")
    print(f"Isolated Checkpoint:     {args.isolated_ckpt}")
    print(f"Bootstrap Iterations:    {args.n_bootstraps:,} (Clustered by Track ID)")
    print("=" * 80)

    config = load_config(args.config)
    feat_dim = config["model"]["feature_dim"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device:        {device}")

    # 1. Load Proposed Model
    print(f"\n[1/4] Loading Proposed Model (Relational Graph Enabled)...")
    prop_model, prop_epoch, prop_val_score = load_model_from_checkpoint(args.proposed_ckpt, use_interaction=True, feature_dim=feat_dim, device=device)
    print(f"      Loaded Proposed Checkpoint (Saved Epoch: {prop_epoch}, Recorded Val Metric: {prop_val_score})")

    # 2. Load Isolated Baseline
    print(f"\n[2/4] Loading Certified Isolated Baseline (use_interaction=False)...")
    iso_model, iso_epoch, iso_val_score = load_model_from_checkpoint(args.isolated_ckpt, use_interaction=False, feature_dim=feat_dim, device=device)
    print(f"      Loaded Isolated Checkpoint (Saved Epoch: {iso_epoch}, Recorded Val Metric: {iso_val_score})")

    # 3. Load Validation Dataset
    print(f"\n[3/4] Initializing DataLoader for split '{args.split}'...")
    dataset = MultimodalSequenceDataset(
        data_dir=args.data_dir,
        split=args.split,
        window_size=config["data"]["window_size"]
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multimodal_batch,
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda")
    )
    total_windows = len(dataset)
    print(f"      Total Evaluation Windows: {total_windows:,} across {len(set(dataset.files)):,} files")

    # 4. Run Inference
    print(f"\n[4/4] Running paired inference on all {total_windows:,} windows...")
    t0 = time.time()
    y_true_p, probs_prop, tracks_p = collect_model_predictions(prop_model, dataloader, device)
    _, probs_iso, _ = collect_model_predictions(iso_model, dataloader, device)
    dur = time.time() - t0
    print(f"      Inference completed in {dur:.2f}s ({total_windows / max(1e-3, dur):.1f} windows/sec)")

    # Save predictions to disk for provenance
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.savez_compressed(
        args.output_file,
        y_true=y_true_p,
        probs_proposed=probs_prop,
        probs_isolated=probs_iso,
        track_ids=tracks_p
    )
    print(f"      Saved raw prediction arrays to: {args.output_file}")

    # Compute Point Estimates
    metrics_prop = compute_metrics_dict(y_true_p, probs_prop)
    metrics_iso = compute_metrics_dict(y_true_p, probs_iso)

    delta_auc = metrics_prop["auc"] - metrics_iso["auc"]
    delta_prauc = metrics_prop["pr_auc"] - metrics_iso["pr_auc"]
    delta_f1 = metrics_prop["f1"] - metrics_iso["f1"]
    delta_acc = metrics_prop["accuracy"] - metrics_iso["accuracy"]

    # Run Paired Track-Clustered Bootstrap
    print(f"\nExecuting {args.n_bootstraps:,} Track-Clustered Bootstraps across independent tracks...")
    boot_res = run_track_clustered_bootstrap(
        y_true_p, probs_prop, probs_iso, tracks_p,
        n_bootstraps=args.n_bootstraps, seed=args.seed
    )

    # Formal Report
    print("\n" + "=" * 80)
    print("HYPOTHESIS H2 EMPIRICAL AUDIT REPORT (PRE-REGISTRATION EVALUATION)")
    print("=" * 80)
    print(f"Evaluation Dataset Split:          {args.split.upper()} (Quarantining held-out test split)")
    print(f"Total Evaluated Windows:           {total_windows:,}")
    print(f"Independent Track Clusters:        {boot_res['num_tracks']} tracks")
    print(f"Minority Class (Crossing) Rate:    {np.mean(y_true_p) * 100:.2f}% ({np.sum(y_true_p):,} / {len(y_true_p):,})")
    print("-" * 80)
    print(f"{'Metric':<25} | {'Proposed Model':<16} | {'Isolated Baseline':<17} | {'Differential (Δ)':<16}")
    print("-" * 80)
    print(f"{'ROC-AUC (%)':<25} | {metrics_prop['auc']:<16.2f} | {metrics_iso['auc']:<17.2f} | {delta_auc:<+16.2f}")
    print(f"{'PR-AUC (Avg Prec %)':<25} | {metrics_prop['pr_auc']:<16.2f} | {metrics_iso['pr_auc']:<17.2f} | {delta_prauc:<+16.2f}")
    print(f"{'F1 Score (th=0.5 %)':<25} | {metrics_prop['f1']:<16.2f} | {metrics_iso['f1']:<17.2f} | {delta_f1:<+16.2f}")
    print(f"{'Overall Accuracy (%)':<25} | {metrics_prop['accuracy']:<16.2f} | {metrics_iso['accuracy']:<17.2f} | {delta_acc:<+16.2f}")
    print(f"{'Brier Score (Calibration)':<25} | {metrics_prop['brier']:<16.4f} | {metrics_iso['brier']:<17.4f} | {metrics_prop['brier'] - metrics_iso['brier']:<+16.4f}")
    print("-" * 80)
    print(f"ΔAUC 95% Track-Clustered CI:       [{boot_res['ci_delta_auc'][0]:+.2f}%, {boot_res['ci_delta_auc'][1]:+.2f}%]")
    print(f"ΔPR-AUC 95% Track-Clustered CI:    [{boot_res['ci_delta_prauc'][0]:+.2f}%, {boot_res['ci_delta_prauc'][1]:+.2f}%]")
    print(f"Two-Sided Bootstrap p-value (H2):  p = {boot_res['p_value_auc']:.5f}")
    print(f"Pre-Registered MMES Threshold:     ΔAUC >= +0.015 (+1.50%)")
    print("-" * 80)

    # Formal Verdict
    ci_lower = boot_res['ci_delta_auc'][0]
    meets_mmes = (delta_auc >= 1.50)
    ci_positive = (ci_lower > 0.0)

    print("PRE-REGISTERED SCIENTIFIC VERDICT:")
    if meets_mmes and ci_positive:
        print("  >> [VERDICT: H2 SUPPORTED] <<")
        print(f"  The proposed relational interaction graph significantly outperforms the isolated baseline")
        print(f"  by ΔAUC = {delta_auc:+.2f}%, strictly satisfying the pre-registered MMES threshold (+1.50%)")
        print(f"  with the 95% Track-Clustered Bootstrap CI excluding zero [{ci_lower:+.2f}%, {boot_res['ci_delta_auc'][1]:+.2f}%].")
    elif ci_positive and not meets_mmes:
        print("  >> [VERDICT: STATISTICALLY DETECTABLE, BUT BELOW MMES] <<")
        print(f"  The relational model shows a statistically detectable improvement (95% CI > 0),")
        print(f"  but the effect size (ΔAUC = {delta_auc:+.2f}%) falls short of the pre-registered MMES threshold (+1.50%).")
        print(f"  Conclusion: Traffic interaction features provide marginal, sub-threshold utility.")
    else:
        print("  >> [VERDICT: H2 NULL / UNSUPPORTED] <<")
        print(f"  The 95% Track-Clustered Bootstrap CI spans zero [{ci_lower:+.2f}%, {boot_res['ci_delta_auc'][1]:+.2f}%].")
        print(f"  Empirical evidence fails to reject the null hypothesis of no interaction utility.")
        print(f"  Conclusion: The relational graph transformer does not reliably outperform the isolated baseline.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
