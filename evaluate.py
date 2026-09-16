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

    # 3. Robustness Suite
    if args.robustness:
        print("\n" + "="*70)
        print("RUNNING ROBUSTNESS AND MISSING MODALITY STRESS TESTS (Phase 27)")
        print("="*70)
        evaluator = RobustnessEvaluator(model, device=device)
        rob_results = evaluator.run_all_robustness_tests(test_loader)
        for condition, res in rob_results.items():
            print(f"Condition [{condition:20s}] -> Acc: {res['accuracy']:5.2f}% | Macro-F1: {res['macro_f1']:5.2f}%")
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

    metrics = compute_classification_metrics(y_true, y_pred)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:        {metrics['accuracy']:.2f}%")
    print(f"Macro-Precision: {metrics['macro_precision']:.2f}%")
    print(f"Macro-Recall:    {metrics['macro_recall']:.2f}%")
    print(f"Macro-F1:        {metrics['macro_f1']:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
