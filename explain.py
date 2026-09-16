"""
XAI Explanation Generation and Faithfulness Validation CLI.
Phases 20, 21, 22, and 26:
Usage:
  # Generate explanation and run faithfulness verification on a sample
  python explain.py --sample_idx 0 --faithfulness --visualize
"""

import os
import argparse
import torch

from configs.config_loader import load_config
from datasets.synthetic_dataset import SyntheticMultimodalDataset
from datasets.multimodal_dataset import collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel
from explainability.explanation_generator import HumanReadableExplanationGenerator
from explainability.faithfulness import XAIFaithfulnessEvaluator
from visualization.attention_maps import plot_temporal_attention, plot_modality_breakdown
from visualization.trajectories import plot_agent_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="Generate X-MIST Multimodal Explanations")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="experiments/results/xai")
    parser.add_argument("--faithfulness", action="store_true", help="Run quantitative XAI faithfulness tests")
    parser.add_argument("--visualize", action="store_true", help="Save visualization figures")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load sample sequence
    dataset = SyntheticMultimodalDataset(num_samples=16, window_size=config["data"]["window_size"])
    batch = collate_multimodal_batch([dataset[args.sample_idx]])
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # 2. Load Model
    model = ProposedXMISTModel(feature_dim=config["model"]["feature_dim"])
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Model Inference
    with torch.no_grad():
        outputs = model(batch)

    # 4. Generate Human-Readable Explanation Report (Phase 22)
    generator = HumanReadableExplanationGenerator(fps=config["data"]["fps"])
    report = generator.generate_report(outputs, batch, sample_idx=0)
    print(report["formatted_text"])

    # Save explanation text to file
    report_file = os.path.join(args.output_dir, f"explanation_sample_{args.sample_idx}.txt")
    with open(report_file, "w") as f:
        f.write(report["formatted_text"])
    print(f"[XAI] Explanation report saved to {report_file}")

    # 5. Faithfulness Verification (Phase 26)
    if args.faithfulness:
        print("\n" + "="*70)
        print("RUNNING QUANTITATIVE XAI FAITHFULNESS VERIFICATION (Phase 26)")
        print("="*70)
        faith_evaluator = XAIFaithfulnessEvaluator(model, device=device)
        faith_suite = faith_evaluator.run_full_suite(batch)

        print("\n[Modality Faithfulness (Drop when zeroed)]:")
        for k, v in faith_suite["modality_faithfulness"].items():
            print(f"  • {k}: {v}")

        print("\n[Temporal Faithfulness (Critical vs Random)]:")
        for k, v in faith_suite["temporal_faithfulness"].items():
            print(f"  • {k}: {v}")

        print("\n[Interaction Faithfulness (Removing Interacting Agents)]:")
        for k, v in faith_suite["interaction_faithfulness"].items():
            print(f"  • {k}: {v}")
        print("="*70 + "\n")

    # 6. Save Visualizations
    if args.visualize:
        print("[XAI] Generating and saving publication figures...")
        # Temporal Attention
        temp_attn = outputs["temporal_attn"][0].detach().cpu().numpy()
        attn_fig = os.path.join(args.output_dir, f"temporal_attention_{args.sample_idx}.png")
        plot_temporal_attention(temp_attn, save_path=attn_fig)

        # Modality Donut/Bar
        mod_fig = os.path.join(args.output_dir, f"modality_breakdown_{args.sample_idx}.png")
        plot_modality_breakdown(report["modality_breakdown"], save_path=mod_fig)

        # Spatial Trajectories
        traj_fig = os.path.join(args.output_dir, f"trajectories_{args.sample_idx}.png")
        plot_agent_trajectories(
            batch["trajectory"][0].detach().cpu().numpy(),
            batch["neighbor_agents"][0].detach().cpu().numpy(),
            save_path=traj_fig
        )
        print(f"[XAI] Figures successfully saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
