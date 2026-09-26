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
from evaluation.classification_metrics import compute_classification_metrics
from evaluation.robustness import RobustnessEvaluator
from evaluation.ablation import run_ablation_study


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate X-MIST / EMIT-HAR Research Models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--model_type", type=str, default="proposed", help="Model type to evaluate")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to evaluation data")
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

    # Prepare Test Data
    if args.synthetic or not os.path.exists(os.path.join(args.data_dir, "test")):
        test_ds = SyntheticMultimodalDataset(num_samples=64, window_size=window_size, seed=999)
    else:
        test_ds = MultimodalSequenceDataset(data_dir=args.data_dir, split="test", window_size=window_size)

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
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            preds = torch.argmax(outputs["ped_logits"], dim=-1).cpu().numpy()
            targets = batch["ped_label"].cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(targets.tolist())

    metrics = compute_classification_metrics(y_true, y_pred, class_names=ped_classes)

    print("\n" + "="*80)
    print(f"EVALUATION RESULTS (Total Test Sequences: {len(y_true):,})")
    print("="*80)

    print("\n-- 1. Multi-Class Performance Overview (Task A: 8 Behaviors) --")
    print(f"Overall Accuracy:        {metrics['accuracy']:.2f}%")
    print(f"Weighted-Precision:      {metrics['weighted_precision']:.2f}%")
    print(f"Weighted-Recall:         {metrics['weighted_recall']:.2f}%")
    print(f"Weighted-F1 (Support):   {metrics['weighted_f1']:.2f}%  (Reflects true test distribution)")
    print(f"Active-Class Macro-F1:   {metrics['active_macro_f1']:.2f}%  (Averaged over classes with test samples)")
    print(f"All-Class Macro-F1:      {metrics['macro_f1']:.2f}%  (Arithmetic mean over all 8 classes)")

    if "per_class" in metrics:
        print("\n-- 2. Per-Class Performance Breakdown Table --")
        print("-" * 80)
        print(f"{'Class Name':<16} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support (Count)':>16}")
        print("-" * 80)
        for c_name, c_data in metrics["per_class"].items():
            sup_str = f"{c_data['support']:,}" if c_data['support'] > 0 else "0 (No test data)"
            print(f"{c_name:<16} | {c_data['precision']:>9.2f}% | {c_data['recall']:>9.2f}% | {c_data['f1']:>9.2f}% | {sup_str:>16}")
        print("-" * 80)

    if "binary_crossing" in metrics:
        bc = metrics["binary_crossing"]
        print("\n-- 3. Binary Crossing Intention Benchmark (Crossing vs. Non-Crossing) --")
        print(f"Binary Accuracy:         {bc['accuracy']:.2f}%")
        print(f"Binary Precision:        {bc['precision']:.2f}%")
        print(f"Binary Recall:           {bc['recall']:.2f}%")
        print(f"Binary F1-Score:         {bc['f1']:.2f}%")
        print(f"Crossing Samples:        {bc['crossing_count']:,} | Non-Crossing Samples: {bc['non_crossing_count']:,}")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
