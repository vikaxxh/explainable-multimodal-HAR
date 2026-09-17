"""
Ablation Study Engine and Benchmark Comparison Table Generator.
Phases 24 & 25 of implementation plan:
Executes the progressive ablation suite (A1 to A9) and generates the primary thesis table:
Model | RGB | Pose | Trajectory | Scene | Interaction | XAI | Macro-F1
"""

import os
import torch
import pandas as pd
from typing import Dict, Any, List
from torch.utils.data import DataLoader

from models.baselines import (
    RGBBaselineModel,
    TrajectoryBaselineModel,
    PoseBaselineModel,
    IntentFormerReproductionModel
)
from models.proposed_model import ProposedXMISTModel
from evaluation.classification_metrics import compute_classification_metrics
from datasets.synthetic_dataset import SyntheticMultimodalDataset
from datasets.multimodal_dataset import collate_multimodal_batch


def build_ablation_model(variant_id: str, feature_dim: int = 256) -> torch.nn.Module:
    """
    Constructs model configuration matching Phase 24 ablations.
    """
    if variant_id == "A1_rgb_only":
        return RGBBaselineModel(feature_dim=feature_dim)
    elif variant_id == "A_traj_only":
        return TrajectoryBaselineModel(feature_dim=feature_dim)
    elif variant_id == "A_pose_only":
        return PoseBaselineModel(feature_dim=feature_dim)
    elif variant_id == "A2_rgb_pose":
        return ProposedXMISTModel(
            feature_dim=feature_dim,
            use_rgb=True, use_pose=True, use_traj=False, use_scene=False,
            use_cross_modal=False, use_interaction=False, use_modality_gating=False
        )
    elif variant_id == "A3_rgb_traj":
        return ProposedXMISTModel(
            feature_dim=feature_dim,
            use_rgb=True, use_pose=False, use_traj=True, use_scene=False,
            use_cross_modal=False, use_interaction=False, use_modality_gating=False
        )
    elif variant_id == "A4_rgb_pose_traj":
        return ProposedXMISTModel(
            feature_dim=feature_dim,
            use_rgb=True, use_pose=True, use_traj=True, use_scene=False,
            use_cross_modal=False, use_interaction=False, use_modality_gating=False
        )
    elif variant_id == "A5_plus_scene":
        return ProposedXMISTModel(
            feature_dim=feature_dim,
            use_rgb=True, use_pose=True, use_traj=True, use_scene=True,
            use_cross_modal=False, use_interaction=False, use_modality_gating=False
        )
    elif variant_id == "A6_plus_cross_modal":
        return ProposedXMISTModel(
            feature_dim=feature_dim,
            use_rgb=True, use_pose=True, use_traj=True, use_scene=True,
            use_cross_modal=True, use_interaction=False, use_modality_gating=True
        )
    elif variant_id == "A7_intentformer_base":
        return IntentFormerReproductionModel(feature_dim=feature_dim)
    elif variant_id == "A8_plus_interaction":
        return ProposedXMISTModel(
            feature_dim=feature_dim,
            use_rgb=True, use_pose=True, use_traj=True, use_scene=True,
            use_cross_modal=True, use_interaction=True, use_modality_gating=True
        )
    elif variant_id == "A9_proposed_full":
        return ProposedXMISTModel(
            feature_dim=feature_dim,
            use_rgb=True, use_pose=True, use_traj=True, use_scene=True,
            use_cross_modal=True, use_interaction=True, use_modality_gating=True
        )
    else:
        raise ValueError(f"Unknown ablation variant: {variant_id}")


@torch.no_grad()
def evaluate_model_on_loader(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> Dict[str, Any]:
    model.eval()
    model.to(device)
    y_true = []
    y_pred = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch)
        preds = torch.argmax(outputs["ped_logits"], dim=-1).cpu().numpy()
        targets = batch["ped_label"].cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(targets.tolist())

    return compute_classification_metrics(y_true, y_pred)


def run_ablation_study(
    test_loader: DataLoader,
    device: str = "cpu",
    output_dir: str = "experiments/results"
) -> pd.DataFrame:
    """
    Evaluates all configurations and generates comparison table (Phase 25).
    """
    os.makedirs(output_dir, exist_ok=True)

    benchmark_matrix = [
        {"name": "RGB baseline", "id": "A1_rgb_only", "rgb": "✓", "pose": "", "traj": "", "scene": "", "inter": "", "xai": ""},
        {"name": "Pose baseline", "id": "A_pose_only", "rgb": "", "pose": "✓", "traj": "", "scene": "", "inter": "", "xai": ""},
        {"name": "Trajectory baseline", "id": "A_traj_only", "rgb": "", "pose": "", "traj": "✓", "scene": "", "inter": "", "xai": ""},
        {"name": "IntentFormer-style", "id": "A7_intentformer_base", "rgb": "✓", "pose": "", "traj": "✓", "scene": "✓", "inter": "", "xai": ""},
        {"name": "Multimodal Transformer", "id": "A6_plus_cross_modal", "rgb": "✓", "pose": "✓", "traj": "✓", "scene": "✓", "inter": "", "xai": ""},
        {"name": "+ Interaction", "id": "A8_plus_interaction", "rgb": "✓", "pose": "✓", "traj": "✓", "scene": "✓", "inter": "✓", "xai": ""},
        {"name": "Proposed (X-MIST)", "id": "A9_proposed_full", "rgb": "✓", "pose": "✓", "traj": "✓", "scene": "✓", "inter": "✓", "xai": "✓"}
    ]

    records = []
    print("\n" + "="*80)
    print("RUNNING ABLATION STUDY AND BASELINE COMPARISONS (Phases 24 & 25)")
    print("="*80)

    for item in benchmark_matrix:
        print(f"Evaluating {item['name']}...")
        model = build_ablation_model(item["id"])
        metrics = evaluate_model_on_loader(model, test_loader, device=device)

        records.append({
            "Model": item["name"],
            "RGB": item["rgb"],
            "Pose": item["pose"],
            "Trajectory": item["traj"],
            "Scene": item["scene"],
            "Interaction": item["inter"],
            "XAI": item["xai"],
            "Accuracy (%)": metrics["accuracy"],
            "Macro-F1 (%)": metrics["macro_f1"]
        })

    df = pd.DataFrame(records)

    # Save to CSV and Markdown
    csv_path = os.path.join(output_dir, "ablation_comparison_table.csv")
    md_path = os.path.join(output_dir, "ablation_comparison_table.md")
    df.to_csv(csv_path, index=False)
    
    try:
        table_str = df.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        table_str = df.to_string(index=False)

    with open(md_path, "w") as f:
        f.write("# Thesis Table: Progressive Benchmark & Ablation Comparison\n\n")
        f.write(table_str)

    print("\n" + table_str + "\n")
    print(f"[Ablation] Results saved to {csv_path} and {md_path}")
    return df
