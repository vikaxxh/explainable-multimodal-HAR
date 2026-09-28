"""
Evaluation CLI for Performance Benchmarking, Ablations, and Robustness Stress-Testing.
Usage:
  # 1. Run full progressive ablation matrix (A1 to A9)
  python evaluate.py --ablation

  # 2. Run robustness stress-testing
  python evaluate.py --model_type proposed --robustness

  # 3. Evaluate specific checkpoint
  python evaluate.py --checkpoint experiments/checkpoints/checkpoint_best.pt
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

from configs.config_loader import load_config
from datasets.synthetic_dataset import SyntheticMultimodalDataset
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel
from evaluation.classification_metrics import compute_classification_metrics, compute_crossing_intention_metrics
from evaluation.robustness import RobustnessEvaluator
from evaluation.ablation import run_ablation_study
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate X-MIST / EMIT-HAR Research Models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--model_type", type=str, default="proposed", help="Model type to evaluate")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to evaluation data")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Data split to evaluate")
    parser.add_argument("--ablation", action="store_true", help="Execute complete ablation matrix (A1 to A9)")
    parser.add_argument("--robustness", action="store_true", help="Execute robustness and sensor stress-testing")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for dry runs")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size = config["data"]["window_size"]

    print(f"[Eval] Running evaluation on device: {device}")

    # Prepare Data
    split_dir = os.path.join(args.data_dir, args.split)
    if args.synthetic or not os.path.exists(split_dir):
        test_ds = SyntheticMultimodalDataset(num_samples=64, window_size=window_size, seed=999)
    else:
        test_ds = MultimodalSequenceDataset(data_dir=args.data_dir, split=args.split, window_size=window_size)

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_multimodal_batch
    )

    # 1. Ablation Matrix
    if args.ablation:
        run_ablation_study(test_loader, device=device, output_dir=config["project"]["output_dir"])
        return

    # 2. Build or Load Target Model
    model = ProposedXMISTModel(feature_dim=config["model"]["feature_dim"])
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Eval] Loaded weights from {args.checkpoint}")
    else:
        print("[Eval] Evaluating model architecture (uninitialized/default weights)...")

    model.to(device)
    model.eval()

    ped_classes = config.get("taxonomy", {}).get("pedestrian_classes", None)

    # 3. Robustness Suite
    if args.robustness:
        print("\n" + "="*80)
        print("RUNNING ROBUSTNESS AND MISSING MODALITY STRESS TESTS (Phase 27)")
        print("="*80)
        evaluator = RobustnessEvaluator(model, device=device, class_names=ped_classes)
        rob_results = evaluator.run_all_robustness_tests(test_loader)
        print(f"{'Condition':<22} | {'Accuracy':>10} | {'Weighted-F1':>12} | {'Macro-F1':>10} | {'Crossing-F1':>12}")
        print("-" * 80)
        for condition, res in rob_results.items():
            cross_f1 = res.get("binary_crossing", {}).get("f1", 0.0)
            w_f1 = res.get("weighted_f1", res["macro_f1"])
            print(f"{condition:<22} | {res['accuracy']:>9.2f}% | {w_f1:>11.2f}% | {res['macro_f1']:>9.2f}% | {cross_f1:>11.2f}%")
        print("="*80 + "\n")
        return

    # 4. Standard Classification Evaluation
    y_true_cross, y_probs_cross, y_pred_cross = [], [], []
    y_true_action, y_pred_action = [], []
    y_true_ped, y_pred_ped = [], []
    neighbor_strata = []

    total_batches = len(test_loader)
    print(f"[Eval] Starting evaluation across {len(test_ds):,} sequences ({total_batches} batches) on split '{args.split}'...")

    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)

            # 1. Primary Intention Benchmark: Dedicated crossing head
            if "crossing_logits" in outputs:
                cross_logits = outputs["crossing_logits"]
                cross_probs = F.softmax(cross_logits, dim=-1)[:, 1].cpu().numpy()
                cross_preds = torch.argmax(cross_logits, dim=-1).cpu().numpy()
            elif "ped_logits" in outputs:
                # Fallback for legacy checkpoints
                ped_probs = F.softmax(outputs["ped_logits"], dim=-1)
                cross_probs = ped_probs[:, 5].cpu().numpy() if ped_probs.shape[-1] > 5 else ped_probs[:, 0].cpu().numpy()
                cross_preds = (torch.argmax(outputs["ped_logits"], dim=-1) == 5).long().cpu().numpy()
            else:
                cross_probs = np.zeros(len(batch["trajectory"]))
                cross_preds = np.zeros(len(batch["trajectory"]), dtype=int)

            if "cross" in batch:
                cross_targets = batch["cross"].cpu().numpy()
            elif "ped_label" in batch:
                cross_targets = (batch["ped_label"] == 5).long().cpu().numpy()
            else:
                cross_targets = np.zeros(len(cross_preds), dtype=int)

            y_true_cross.extend(cross_targets.tolist())
            y_probs_cross.extend(cross_probs.tolist())
            y_pred_cross.extend(cross_preds.tolist())

            # Neighbor count stratification
            if "neighbor_mask" in batch:
                n_active = batch["neighbor_mask"][:, -1, :].sum(dim=-1).cpu().numpy()
                neighbor_strata.extend(n_active.tolist())
            else:
                neighbor_strata.extend([0] * len(cross_targets))

            # Secondary Action state (Walking vs Standing)
            if "action_logits" in outputs and "action" in batch:
                action_preds = torch.argmax(outputs["action_logits"], dim=-1).cpu().numpy()
                action_targets = batch["action"].cpu().numpy()
                y_true_action.extend(action_targets.tolist())
                y_pred_action.extend(action_preds.tolist())

            # Multi-class ped behaviors
            if "ped_logits" in outputs and "ped_label" in batch:
                preds = torch.argmax(outputs["ped_logits"], dim=-1).cpu().numpy()
                targets = batch["ped_label"].cpu().numpy()
                y_pred_ped.extend(preds.tolist())
                y_true_ped.extend(targets.tolist())

            if (b_idx + 1) % 10 == 0 or b_idx == total_batches - 1:
                pct = ((b_idx + 1) / total_batches) * 100.0
                print(f"[Eval] Progress: {b_idx + 1}/{total_batches} batches ({pct:.1f}%)...", end="\r", flush=True)

    print()  # newline after progress bar

    # 1. Primary Benchmark Metrics: Crossing Intention
    cross_metrics = compute_crossing_intention_metrics(y_true_cross, y_probs_cross, y_pred_cross)

    print("\n" + "="*85)
    print(f"STANDARDIZED BENCHMARK EVALUATION RESULTS ({args.split.upper()} SPLIT: {len(y_true_cross):,} SEQUENCES)")
    print("="*85)

    print("\n-- 1. Primary Benchmark: Crossing Intention Prediction (Human Ground Truth) --")
    print(f"Accuracy:         {cross_metrics['accuracy']:.2f}%")
    print(f"ROC-AUC:          {cross_metrics['auc']:.2f}%")
    print(f"F1-Score:         {cross_metrics['f1']:.2f}%")
    print(f"Precision:        {cross_metrics['precision']:.2f}%")
    print(f"Recall:           {cross_metrics['recall']:.2f}%")
    print(f"Sample Support:   {cross_metrics['crossing_count']:,} Crossing | {cross_metrics['non_crossing_count']:,} Non-Crossing")

    # 2. Stratified Evaluation by Neighbor Count
    if len(neighbor_strata) == len(y_true_cross):
        print("\n-- 2. Interaction Stratification Breakdown (by Non-Ego Neighbor Count K) --")
        print("-" * 85)
        print(f"{'Strata':<22} | {'Samples':>9} | {'Accuracy':>10} | {'ROC-AUC':>10} | {'F1-Score':>10} | {'Recall':>10}")
        print("-" * 85)
        k_arr = np.array(neighbor_strata)
        y_t_arr = np.array(y_true_cross)
        y_pr_arr = np.array(y_probs_cross)
        y_p_arr = np.array(y_pred_cross)

        for s_label, mask in [
            ("K = 0 (Ego-only)", k_arr == 0),
            ("K = 1 (1 Neighbor)", k_arr == 1),
            ("K >= 2 (Dense Multi-Agent)", k_arr >= 2),
            ("All Sequences", np.ones(len(k_arr), dtype=bool))
        ]:
            if np.sum(mask) > 0:
                s_met = compute_crossing_intention_metrics(y_t_arr[mask], y_pr_arr[mask], y_p_arr[mask])
                print(f"{s_label:<22} | {np.sum(mask):>9,} | {s_met['accuracy']:>9.2f}% | {s_met['auc']:>9.2f}% | {s_met['f1']:>9.2f}% | {s_met['recall']:>9.2f}%")
            else:
                print(f"{s_label:<22} | {0:>9} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
        print("-" * 85)
        print("Note: On PIE/JAAD, ego-vehicle is the primary interaction partner (majority K=0 non-ego).")

    # 3. Secondary Action State Benchmark (Standing vs Walking)
    if y_true_action and len(y_true_action) > 0:
        act_met = compute_crossing_intention_metrics(y_true_action, y_pred_action, y_pred_action)
        print("\n-- 3. Secondary Benchmark: Action State Recognition (Standing vs. Walking) --")
        print(f"Action Accuracy:  {act_met['accuracy']:.2f}% | F1: {act_met['f1']:.2f}% | Prec: {act_met['precision']:.2f}% | Rec: {act_met['recall']:.2f}%")

    # 4. Multi-Class Behavioral Overview
    if y_true_ped and len(y_true_ped) > 0:
        metrics = compute_classification_metrics(y_true_ped, y_pred_ped, class_names=ped_classes)
        print("\n-- 4. Multi-Class Behavioral Breakdown (Pedestrian Head) --")
        print(f"Overall Accuracy: {metrics['accuracy']:.2f}% | Weighted-F1: {metrics['weighted_f1']:.2f}% | Macro-F1: {metrics['macro_f1']:.2f}%")

    print("="*85 + "\n")


if __name__ == "__main__":
    main()
