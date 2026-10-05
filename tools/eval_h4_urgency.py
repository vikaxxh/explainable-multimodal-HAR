#!/usr/bin/env python3
"""
tools/eval_h4_urgency.py: Hypothesis H4 Kinematic Approach Urgency Evaluation Engine.
Evaluates whether crossing intention models respond monotonically to physical approach urgency:
- Scales closing neighbor approach velocity by alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
- Measures Spearman rank correlation rho(alpha, p_cross)
- Tests against placebo perturbations via McNemar's test
- Clustered at the independent pedestrian track level
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
from typing import Dict, List, Tuple, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel


ALPHA_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def evaluate_h4(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    speed_floor: float = 0.002,
    seed: int = 42
) -> Dict[str, Any]:
    model.eval()
    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("RUNNING HYPOTHESIS H4 KINEMATIC APPROACH URGENCY AUDIT")
    print(f"Alpha Scale Grid: {ALPHA_GRID}")
    print(f"Speed Floor for Dynamic Agents: {speed_floor}")
    print("=" * 80)

    total_windows = 0
    closing_windows = 0
    separating_windows = 0
    stationary_windows = 0

    track_rhos_closing: Dict[str, List[float]] = {}
    track_rhos_separating: Dict[str, List[float]] = {}
    track_rhos_placebo: Dict[str, List[float]] = {}

    all_rhos_closing = []
    all_rhos_separating = []
    all_rhos_placebo = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if "track_id" not in batch:
                raise KeyError("Batch is missing 'track_id'. Ensure datasets/multimodal_dataset.py forwards 'track_id'.")

            B = batch["trajectory"].shape[0]
            T = batch["trajectory"].shape[1]
            total_windows += B
            t_eval = T - 1

            p_pos = batch["trajectory"][:, t_eval, :2].numpy()     # (B, 2)
            p_vel = batch["trajectory"][:, t_eval, 2:4].numpy()    # (B, 2)
            n_pos = batch["neighbor_agents"][:, t_eval, :, :2].numpy() # (B, K, 2)
            n_vel = batch["neighbor_agents"][:, t_eval, :, 2:4].numpy() # (B, K, 2)
            n_val = batch["neighbor_mask"][:, t_eval].numpy().astype(bool) # (B, K)

            for b in range(B):
                valid_k = np.where(n_val[b])[0]
                if len(valid_k) == 0:
                    continue

                # Compute line-of-sight closing speeds for all valid neighbors
                closing_speeds = []
                for k in valid_k:
                    rel_p = p_pos[b] - n_pos[b, k]
                    dist = float(np.linalg.norm(rel_p))
                    rel_v = p_vel[b] - n_vel[b, k]
                    v_mag = float(np.linalg.norm(n_vel[b, k]))

                    if dist < 1e-4 or v_mag < speed_floor:
                        closing_speeds.append((0.0, k, "stationary"))
                    else:
                        unit_r = rel_p / dist
                        v_closing = float(-np.dot(rel_v, unit_r))
                        cat = "closing" if v_closing > 0.001 else "separating"
                        closing_speeds.append((v_closing, k, cat))

                # Pick the most urgent dynamic neighbor (highest closing velocity)
                closing_speeds.sort(key=lambda x: x[0], reverse=True)
                top_v_closing, top_k, category = closing_speeds[0]

                if category == "stationary":
                    stationary_windows += 1
                    continue
                elif category == "closing":
                    closing_windows += 1
                else:
                    separating_windows += 1

                # Construct scaled perturbation inputs for alpha grid
                # Shape: [len(ALPHA_GRID), T, ...]
                probs_alpha = []
                for alpha in ALPHA_GRID:
                    batch_pert = {k: (v[b:b+1].clone().to(device) if torch.is_tensor(v) else v[b:b+1]) for k, v in batch.items()}
                    
                    # Co-rescale neighbor trajectory kinematically:
                    # p(t) = p(T) - alpha * (p(T) - p(t))
                    # v(t) = alpha * v(t)
                    ref_p = batch_pert["neighbor_agents"][:, t_eval:t_eval+1, top_k, :2].clone()
                    disp = ref_p - batch_pert["neighbor_agents"][:, :, top_k, :2]
                    batch_pert["neighbor_agents"][:, :, top_k, :2] = ref_p - alpha * disp
                    batch_pert["neighbor_agents"][:, :, top_k, 2:4] = batch_pert["neighbor_agents"][:, :, top_k, 2:4] * alpha

                    out = model(batch_pert)
                    if getattr(model, "_use_ped_head_fallback", False):
                        p_cr = float(torch.softmax(out["ped_logits"], dim=-1)[0, 5].cpu().item())
                    else:
                        p_cr = float(torch.softmax(out["crossing_logits"], dim=-1)[0, 1].cpu().item())
                    probs_alpha.append(p_cr)

                # Compute Spearman rank correlation rho(alpha, p_cross)
                # If all probabilities are identical, rho = 0
                if len(set(probs_alpha)) > 1:
                    rho, _ = stats.spearmanr(ALPHA_GRID, probs_alpha)
                    rho = float(rho) if not np.isnan(rho) else 0.0
                else:
                    rho = 0.0

                # Run Placebo Control: permute neighbor coordinates across time
                batch_placebo = {k: (v[b:b+1].clone().to(device) if torch.is_tensor(v) else v[b:b+1]) for k, v in batch.items()}
                perm_idx = rng.permutation(T)
                batch_placebo["neighbor_agents"][:, :, top_k] = batch_placebo["neighbor_agents"][:, perm_idx, top_k]
                probs_placebo = []
                for alpha in ALPHA_GRID:
                    batch_p_alpha = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_placebo.items()}
                    batch_p_alpha["neighbor_agents"][:, :, top_k, 2:4] *= alpha
                    out_p = model(batch_p_alpha)
                    if getattr(model, "_use_ped_head_fallback", False):
                        p_cr_p = float(torch.softmax(out_p["ped_logits"], dim=-1)[0, 5].cpu().item())
                    else:
                        p_cr_p = float(torch.softmax(out_p["crossing_logits"], dim=-1)[0, 1].cpu().item())
                    probs_placebo.append(p_cr_p)

                if len(set(probs_placebo)) > 1:
                    rho_p, _ = stats.spearmanr(ALPHA_GRID, probs_placebo)
                    rho_p = float(rho_p) if not np.isnan(rho_p) else 0.0
                else:
                    rho_p = 0.0

                tid = str(batch["track_id"][b])
                if category == "closing":
                    track_rhos_closing.setdefault(tid, []).append(rho)
                    all_rhos_closing.append(rho)
                    track_rhos_placebo.setdefault(tid, []).append(rho_p)
                    all_rhos_placebo.append(rho_p)
                else:
                    track_rhos_separating.setdefault(tid, []).append(rho)
                    all_rhos_separating.append(rho)

            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {total_windows:,} windows | Closing: {closing_windows:,} | Separating: {separating_windows:,} | Stationary: {stationary_windows:,}")

    # Track-clustered median correlations
    track_med_closing = [float(np.median(v)) for v in track_rhos_closing.values()]
    track_med_placebo = [float(np.median(v)) for v in track_rhos_placebo.values()]
    track_med_separating = [float(np.median(v)) for v in track_rhos_separating.values()]

    # Shares with expected negative monotonic response (rho < -0.50)
    neg_closing = sum(1 for r in track_med_closing if r < -0.50)
    neg_placebo = sum(1 for r in track_med_placebo if r < -0.50)
    neg_sep = sum(1 for r in track_med_separating if r < -0.50)
    n_cl = max(1, len(track_med_closing))
    n_sep = max(1, len(track_med_separating))

    # McNemar's paired test comparing paired tracks between closing vs placebo
    # Table: [[both neg, closing neg only], [placebo neg only, neither neg]]
    n_01 = 0 # closing < -0.50 and placebo >= -0.50
    n_10 = 0 # placebo < -0.50 and closing >= -0.50
    for r_c, r_p in zip(track_med_closing, track_med_placebo):
        if r_c < -0.50 and r_p >= -0.50:
            n_01 += 1
        elif r_p < -0.50 and r_c >= -0.50:
            n_10 += 1

    # McNemar test statistic with continuity correction
    b_c = n_01 + n_10
    if b_c > 0:
        mc_stat = (abs(n_01 - n_10) - 1.0)**2 / b_c
        p_mcnemar = float(stats.chi2.sf(mc_stat, df=1))
    else:
        mc_stat, p_mcnemar = 0.0, 1.0

    print("\n" + "=" * 80)
    print("HYPOTHESIS 4 (H4) KINEMATIC URGENCY AUDIT REPORT")
    print("=" * 80)
    print(f"Total Windows Evaluated:              {total_windows:,}")
    print(f"Closing Dynamic Windows (v_close > 0): {closing_windows:,}")
    print(f"Separating Windows (v_close <= 0):     {separating_windows:,}")
    print(f"Stationary Windows:                   {stationary_windows:,}")
    print("-" * 80)
    print(f"Independent Closing Track Clusters:    {len(track_med_closing)} tracks")
    print(f"Median Spearman rho (Closing):         {float(np.median(track_med_closing)):.4f}")
    print(f"Median Spearman rho (Placebo):         {float(np.median(track_med_placebo)):.4f}")
    print(f"Median Spearman rho (Separating):      {float(np.median(track_med_separating)):.4f}")
    print("-" * 80)
    print(f"Monotonic Response Share (rho < -0.50):")
    print(f"  Closing Condition:                  {neg_closing}/{n_cl} ({neg_closing/n_cl*100:.1f}%)")
    print(f"  Placebo Condition:                  {neg_placebo}/{n_cl} ({neg_placebo/n_cl*100:.1f}%)")
    print(f"  Separating Condition:               {neg_sep}/{n_sep} ({neg_sep/n_sep*100:.1f}%)")
    print("-" * 80)
    print(f"Paired McNemar Test (Closing vs Placebo): chi2 = {mc_stat:.2f}, p = {p_mcnemar:.5e}")
    print("=" * 80 + "\n")

    return {
        "closing_tracks": len(track_med_closing),
        "median_rho_closing": float(np.median(track_med_closing)),
        "median_rho_placebo": float(np.median(track_med_placebo)),
        "share_negative_closing": neg_closing / n_cl,
        "share_negative_placebo": neg_placebo / n_cl,
        "mcnemar_chi2": float(mc_stat),
        "mcnemar_p_val": float(p_mcnemar)
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Hypothesis H4: Response to Normalized Kinematic Approach Urgency")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    p.add_argument("--data_dir", type=str, default="data/processed_pie", help="Path to processed dataset")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    p.add_argument("--speed_floor", type=float, default=0.002, help="Speed floor for stationary agents")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--frozen", action="store_true", help="Required flag when evaluating on held-out test split")
    args = p.parse_args()

    # Pre-registration test guard
    if args.split == "test" and not args.frozen:
        sys.exit(
            "Refusing to execute on held-out test split without --frozen flag. "
            "The test split must only be evaluated once under pre-registered conditions."
        )

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

    evaluate_h4(
        model=model,
        loader=loader,
        device=device,
        speed_floor=args.speed_floor,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
