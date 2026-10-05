#!/usr/bin/env python3
"""
Hypothesis H3: Vectorized Causal Ablation Faithfulness Engine.
Evaluates whether dynamic interaction attention reflects genuine behavioral sensitivity
via counterfactual neighbor masking compared against distance/speed/heading-matched controls.
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Generator

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel


def make_batches_and_predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    def batch_generator() -> Generator[Dict[str, Any], None, None]:
        for batch in loader:
            if "track_id" not in batch:
                raise KeyError(
                    "Fatal: batch is missing 'track_id'. Ensure datasets/multimodal_dataset.py "
                    "reads 'ped_id' or 'metadata' and forwards 'track_id' in collate_multimodal_batch."
                )

            T = batch["trajectory"].shape[1]
            t_eval = T - 1

            # Bounding-box centers in normalized coords: [x, y, vx, vy, heading]
            nb = batch["neighbor_agents"][:, t_eval].cpu().numpy()  # (B, K, 5)
            pr = batch["trajectory"][:, t_eval].cpu().numpy()        # (B, 5)

            # Positions are in absolute normalized image coordinates [0, 1]
            # Compute relative spatial and kinematic properties
            rel_pos = nb[..., :2] - pr[:, None, :2]
            dists = np.linalg.norm(rel_pos, axis=-1)                # (B, K)
            speeds = np.linalg.norm(nb[..., 2:4], axis=-1)          # (B, K)
            headings = nb[..., 4]                                   # (B, K)

            yield {
                "neighbor_valid": batch["neighbor_mask"][:, t_eval].cpu().numpy().astype(bool), # (B, K)
                "neighbor_dist": dists,
                "neighbor_speed": speeds,
                "neighbor_heading": headings,
                "track_id": np.asarray(batch["track_id"]),
                "_raw": batch
            }

    @torch.no_grad()
    def predict_batch_with_mask(batch_dict: Dict[str, Any], slot_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs model inference with selective neighbor slots masked out.
        slot_mask: (B, K) boolean mask. True = keep neighbor, False = mask neighbor out.
        """
        raw = {k: (v.to(device).clone() if torch.is_tensor(v) else v) for k, v in batch_dict["_raw"].items()}
        T = raw["trajectory"].shape[1]

        if slot_mask is not None:
            # slot_mask is (B, K) -> expand to (B, 1, K) across time
            m = torch.as_tensor(slot_mask, device=device)[:, None, :]
            # Zeroing features and setting mask=False matches absent padding convention
            raw["neighbor_mask"] = raw["neighbor_mask"] & m
            raw["neighbor_agents"] = raw["neighbor_agents"] * m[..., None]

        out = model(raw)
        if getattr(model, "_use_ped_head_fallback", False):
            # Checkpoint was trained on 8-class pedestrian behavior (Class 5 = Crossing)
            probs = torch.softmax(out["ped_logits"], dim=-1)[:, 5].cpu().numpy()
        else:
            lg = out["crossing_logits"]  # (B, 2)
            probs = torch.softmax(lg, dim=-1)[:, 1].cpu().numpy()
        
        attn_w = out.get("interaction_weights")
        if attn_w is None:
            raise RuntimeError(
                "Fatal: Model forward pass did not return 'interaction_weights'. "
                "Ensure self.use_interaction is True and the interaction transformer is active."
            )

        attn_weights = attn_w[:, T - 1].cpu().numpy()
        return probs, attn_weights

    return batch_generator(), predict_batch_with_mask


def find_matched_control_slot(
    target_k: int,
    valid_mask: np.ndarray,      # (K,)
    dists: np.ndarray,           # (K,)
    speeds: np.ndarray,          # (K,)
    headings: np.ndarray,        # (K,)
    dist_tol: float,
    speed_tol: float,
    heading_tol_rad: float,
    speed_floor: float
) -> Optional[int]:
    """
    Finds best matched control slot k_ctrl for target_k within tolerances.
    """
    d_star = dists[target_k]
    v_star = speeds[target_k]
    theta_star = headings[target_k]

    best_k = None
    best_error = float("inf")

    K = len(valid_mask)
    for k in range(K):
        if k == target_k or not valid_mask[k]:
            continue

        d_k = dists[k]
        v_k = speeds[k]
        theta_k = headings[k]

        # 1. Distance relative tolerance
        dist_err = abs(d_k - d_star) / max(d_star, 1e-4)
        if dist_err > dist_tol:
            continue

        # 2. Speed tolerance with physical floor for stationary/parked agents
        v_denom = max(v_star, speed_floor)
        speed_err = abs(v_k - v_star) / v_denom
        if speed_err > speed_tol:
            continue

        # 3. Signed angular difference wrapped to [-pi, pi] (evaluated only when both agents are moving)
        if v_star >= speed_floor and v_k >= speed_floor:
            diff_h = abs((theta_k - theta_star + math.pi) % (2 * math.pi) - math.pi)
            if diff_h > heading_tol_rad:
                continue
            heading_err = diff_h / heading_tol_rad
        else:
            # Stationary / parked agents have undefined heading from camera jitter; do not penalize
            heading_err = 0.0

        total_err = dist_err + speed_err + heading_err
        if total_err < best_error:
            best_error = total_err
            best_k = k

    return best_k


def run_h3_experiment(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    selection_mode: str = "attention", # "attention" or "random" (placebo)
    dist_tol: float = 0.15,
    speed_tol: float = 0.20,
    heading_tol_deg: float = 30.0,
    speed_floor: float = 0.002,
    seed: int = 42
) -> Dict[str, Any]:
    
    rng = np.random.default_rng(seed)
    gen, predict_fn = make_batches_and_predict(model, loader, device)
    heading_tol_rad = heading_tol_deg * (math.pi / 180.0)

    total_windows = 0
    windows_with_neighbors = 0
    windows_k_single = 0         # Single neighbor present (no candidate possible)
    windows_tolerance_failed = 0 # Multiple neighbors present but none within tolerance
    matched_evaluations = 0

    track_deltas: Dict[str, List[float]] = {}
    delta_target_all = []
    delta_ctrl_all = []
    delta_faith_all = []
    all_valid_speeds = []

    print("\n" + "=" * 80)
    print(f"RUNNING HYPOTHESIS H3 CAUSAL ABLATION AUDIT (Mode: {selection_mode.upper()})")
    print(f"Tolerance: Dist +/-{dist_tol*100:.0f}%, Speed +/-{speed_tol*100:.0f}%, Heading +/-{heading_tol_deg:.0f}° | Speed Floor: {speed_floor}")
    print("=" * 80)

    for batch_idx, batch_dict in enumerate(gen):
        B, K = batch_dict["neighbor_valid"].shape
        total_windows += B

        # Collect speed distribution for calibration diagnostic
        valid_speeds = batch_dict["neighbor_speed"][batch_dict["neighbor_valid"]]
        if len(valid_speeds) > 0:
            all_valid_speeds.extend(valid_speeds.tolist())

        # 1. Unablated Baseline Pass
        probs_0, attn_w = predict_fn(batch_dict, slot_mask=None)

        # 2. Batch-Level Ablation Passes per Slot (K forward passes for entire batch)
        probs_ablated = np.zeros((B, K), dtype=np.float32)
        for k in range(K):
            mask_k = np.ones((B, K), dtype=bool)
            mask_k[:, k] = False
            probs_ablated[:, k], _ = predict_fn(batch_dict, slot_mask=mask_k)

        # 3. Match Pairs and Compute Differentials
        for b in range(B):
            valid_k = np.where(batch_dict["neighbor_valid"][b])[0]
            num_valid = len(valid_k)

            if num_valid == 0:
                continue
            windows_with_neighbors += 1

            if num_valid < 2:
                windows_k_single += 1
                continue

            # Target Selection
            if selection_mode == "attention":
                # Select top-attended active neighbor
                attns = attn_w[b, valid_k]
                target_k = valid_k[int(np.argmax(attns))]
            elif selection_mode == "random":
                # Seeded placebo negative control: pick uniform random active neighbor
                target_k = int(rng.choice(valid_k))
            else:
                raise ValueError(f"Unknown selection mode: {selection_mode}")

            # Match Control
            ctrl_k = find_matched_control_slot(
                target_k=target_k,
                valid_mask=batch_dict["neighbor_valid"][b],
                dists=batch_dict["neighbor_dist"][b],
                speeds=batch_dict["neighbor_speed"][b],
                headings=batch_dict["neighbor_heading"][b],
                dist_tol=dist_tol,
                speed_tol=speed_tol,
                heading_tol_rad=heading_tol_rad,
                speed_floor=speed_floor
            )

            if ctrl_k is None:
                windows_tolerance_failed += 1
                continue

            matched_evaluations += 1

            # Causal output displacements
            p_orig = probs_0[b]
            delta_target = abs(p_orig - probs_ablated[b, target_k])
            delta_ctrl = abs(p_orig - probs_ablated[b, ctrl_k])
            delta_faith = delta_target - delta_ctrl

            delta_target_all.append(delta_target)
            delta_ctrl_all.append(delta_ctrl)
            delta_faith_all.append(delta_faith)

            tid = str(batch_dict["track_id"][b])
            track_deltas.setdefault(tid, []).append(delta_faith)

        if (batch_idx + 1) % 50 == 0:
            print(f"Processed {total_windows:,} windows | Matched: {matched_evaluations:,} | Tol Failures: {windows_tolerance_failed:,}")

    # Speed Distribution Diagnostic
    if len(all_valid_speeds) > 0:
        sp_p10, sp_p25, sp_p50, sp_p75, sp_p90 = np.percentile(all_valid_speeds, [10, 25, 50, 75, 90])
        print(f"\n[Speed Calibration Diagnostic] Valid Neighbor Speeds (Normalized):")
        print(f"  P10={sp_p10:.5f} | P25={sp_p25:.5f} | P50={sp_p50:.5f} | P75={sp_p75:.5f} | P90={sp_p90:.5f}")
        print(f"  Configured speed_floor: {speed_floor:.5f}")

    print("\n" + "=" * 80)
    print("HYPOTHESIS 3 (H3) EMPIRICAL AUDIT REPORT")
    print("=" * 80)
    print(f"Selection Mode:                    {selection_mode.upper()}")
    print(f"Total Windows Processed:           {total_windows:,}")
    print(f"Windows with Neighbors (K >= 1):   {windows_with_neighbors:,} ({windows_with_neighbors/max(total_windows,1)*100:.1f}%)")
    print(f"Windows with Single Neighbor (K=1):{windows_k_single:,} (no control possible)")
    print(f"Windows with Multiple (K >= 2):    {windows_with_neighbors - windows_k_single:,}")
    print(f"Matching Tolerance Failures:       {windows_tolerance_failed:,} (no match within tolerances)")
    print(f"Matched Control Pairs Evaluated:   {matched_evaluations:,}")
    print("-" * 80)

    if matched_evaluations < 10:
        print("[Error] Insufficient matched pairs found. Check tolerances and speed floor.")
        return {}

    # Track-Level Clustered Aggregation
    track_medians = [float(np.median(vals)) for vals in track_deltas.values() if len(vals) > 0]
    num_tracks = len(track_medians)

    # Two-Sided Paired Wilcoxon Signed-Rank Test across Track Medians
    w_stat, p_val = stats.wilcoxon(track_medians, alternative="two-sided")

    # 2,000 Track-Clustered Bootstrap Resamples for 95% Confidence Interval
    boot_rng = np.random.default_rng(seed)
    boot_medians = [np.median(boot_rng.choice(track_medians, size=num_tracks, replace=True)) for _ in range(2000)]
    ci_lower = float(np.percentile(boot_medians, 2.5))
    ci_upper = float(np.percentile(boot_medians, 97.5))

    print(f"Track Clusters (Independent Units):{num_tracks} tracks")
    print(f"Median Target Displacement (ΔP_tgt):{float(np.median(delta_target_all)):.5f}")
    print(f"Median Control Displacement (ΔP_ctrl):{float(np.median(delta_ctrl_all)):.5f}")
    print(f"Track Median Differential (δ):     {float(np.median(track_medians)):.5f}")
    print(f"95% Track-Clustered Bootstrap CI:  [{ci_lower:.5f}, {ci_upper:.5f}]")
    print(f"Paired Wilcoxon Signed-Rank Test:  W = {w_stat:.1f}, two-sided p = {p_val:.5e}")
    print("=" * 80 + "\n")

    return {
        "selection_mode": selection_mode,
        "total_windows": total_windows,
        "matched_pairs": matched_evaluations,
        "track_clusters": num_tracks,
        "track_median_delta": float(np.median(track_medians)),
        "ci_95": (ci_lower, ci_upper),
        "wilcoxon_stat": float(w_stat),
        "p_val_two_sided": float(p_val)
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Hypothesis H3: Attribution Faithfulness via Causal Ablation")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    p.add_argument("--data_dir", type=str, default="data/processed_pie", help="Path to processed dataset")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    p.add_argument("--selection", type=str, default="attention", choices=["attention", "random"],
                   help="Target selection mode: 'attention' (test) or 'random' (placebo check)")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for vectorized ablation")
    p.add_argument("--dist_tol", type=float, default=0.15, help="Relative distance tolerance (default: 0.15)")
    p.add_argument("--speed_tol", type=float, default=0.20, help="Relative speed tolerance (default: 0.20)")
    p.add_argument("--heading_tol", type=float, default=30.0, help="Heading tolerance in degrees (default: 30.0)")
    p.add_argument("--speed_floor", type=float, default=0.002, help="Speed floor for stationary agents (calibrated on val)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for placebo reproducibility")
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

    run_h3_experiment(
        model=model,
        loader=loader,
        device=device,
        selection_mode=args.selection,
        dist_tol=args.dist_tol,
        speed_tol=args.speed_tol,
        heading_tol_deg=args.heading_tol,
        speed_floor=args.speed_floor,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
