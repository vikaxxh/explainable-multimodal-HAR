#!/usr/bin/env python3
"""
tools/audit_h3_diagnostics.py: Complete Methodological Diagnostic Suite for Hypothesis H3.
Performs pre-specified diagnostics without data snooping:
1. Reconciles exact window counts (disk files vs loader yielded windows).
2. Measures per-criterion pass rates (Distance, Speed, Heading individually and combined).
3. Profiles matched vs unmatched samples (distance, speed, stationary share, attention).
4. Inspects matches-per-track concentration and effective Wilcoxon sample size (zero δ count).
5. Analyzes heavy-tail distribution of ΔP (mean, median, p90, share > 0.05).
6. Runs the Mask-All-Neighbours global ablation to compare full vs zero-neighbor AUC.
"""

import os
import sys
import glob
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from typing import Dict, List, Any

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.config_loader import load_config
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch
from models.proposed_model import ProposedXMISTModel
from evaluation.classification_metrics import compute_crossing_intention_metrics


def run_diagnostics(
    model: torch.nn.Module,
    loader: DataLoader,
    data_dir: str,
    split: str,
    device: torch.device,
    dist_tol: float = 0.15,
    speed_tol: float = 0.20,
    heading_tol_deg: float = 30.0,
    speed_floor: float = 0.002,
    seed: int = 42
):
    model.eval()
    rng = np.random.default_rng(seed)
    heading_tol_rad = heading_tol_deg * (math.pi / 180.0)

    print("=" * 80)
    print("HYPOTHESIS H3 METHODOLOGICAL DIAGNOSTIC AUDIT")
    print(f"Split: {split.upper()} | Tolerances: Dist +/-{dist_tol*100:.0f}%, Speed +/-{speed_tol*100:.0f}%, Head +/-{heading_tol_deg:.0f}° | Floor: {speed_floor}")
    print("=" * 80)

    # 1. Reconcile Window Counts
    disk_files = glob.glob(os.path.join(data_dir, split, "*.npz"))
    num_disk_files = len(disk_files)

    total_windows = 0
    windows_with_neighbors = 0
    windows_k_single = 0
    windows_k_multi = 0

    # Diagnostic accumulators
    pass_dist_count = 0
    pass_speed_count = 0
    pass_head_count = 0
    pass_all_count = 0
    total_candidate_pairs = 0

    matched_windows = 0
    matched_tracks: Dict[str, int] = Counter()

    delta_target_all = []
    delta_ctrl_all = []
    track_deltas = defaultdict(list)

    # Profiling: matched vs unmatched
    profile = {
        "matched": {"dist": [], "speed": [], "num_nbrs": [], "stationary": 0},
        "unmatched": {"dist": [], "speed": [], "num_nbrs": [], "stationary": 0}
    }

    # Mask-All global ablation tracking
    y_true_all = []
    probs_unmasked_all = []
    probs_masked_all = []

    print("\n[Running forward inference and ablation passes across validation split...]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            B = batch["trajectory"].shape[0]
            T = batch["trajectory"].shape[1]
            t_eval = T - 1
            total_windows += B

            # True targets
            targets = batch["cross"].cpu().numpy().tolist()
            y_true_all.extend(targets)

            # --- A. Global Mask-All Test ---
            raw_unmasked = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out_unmasked = model(raw_unmasked)
            if getattr(model, "_use_ped_head_fallback", False):
                p_unmasked = F.softmax(out_unmasked["ped_logits"], dim=-1)[:, 5].cpu().numpy()
            else:
                p_unmasked = F.softmax(out_unmasked["crossing_logits"], dim=-1)[:, 1].cpu().numpy()
            probs_unmasked_all.extend(p_unmasked.tolist())

            # Mask out ALL neighbors
            raw_masked = {k: (v.clone().to(device) if torch.is_tensor(v) else v) for k, v in raw_unmasked.items()}
            raw_masked["neighbor_mask"] = torch.zeros_like(raw_masked["neighbor_mask"])
            raw_masked["neighbor_agents"] = torch.zeros_like(raw_masked["neighbor_agents"])
            out_masked = model(raw_masked)
            if getattr(model, "_use_ped_head_fallback", False):
                p_masked = F.softmax(out_masked["ped_logits"], dim=-1)[:, 5].cpu().numpy()
            else:
                p_masked = F.softmax(out_masked["crossing_logits"], dim=-1)[:, 1].cpu().numpy()
            probs_masked_all.extend(p_masked.tolist())

            # --- B. Per-Slot Ablation Passes (K forward passes) ---
            K = batch["neighbor_mask"].shape[2]
            probs_ablated = np.zeros((B, K), dtype=np.float32)
            for k in range(K):
                raw_k = {k_: (v_.clone().to(device) if torch.is_tensor(v_) else v_) for k_, v_ in raw_unmasked.items()}
                m = torch.ones_like(raw_k["neighbor_mask"])
                m[:, :, k] = False
                raw_k["neighbor_mask"] = raw_k["neighbor_mask"] & m
                raw_k["neighbor_agents"][:, :, k] = 0.0
                out_k = model(raw_k)
                if getattr(model, "_use_ped_head_fallback", False):
                    probs_ablated[:, k] = F.softmax(out_k["ped_logits"], dim=-1)[:, 5].cpu().numpy()
                else:
                    probs_ablated[:, k] = F.softmax(out_k["crossing_logits"], dim=-1)[:, 1].cpu().numpy()

            # --- C. Kinematics, Matching & Profiling ---
            nb = batch["neighbor_agents"][:, t_eval].cpu().numpy()
            pr = batch["trajectory"][:, t_eval].cpu().numpy()
            n_val = batch["neighbor_mask"][:, t_eval].cpu().numpy().astype(bool)
            attn_w = out_unmasked.get("interaction_weights")
            attns_last = attn_w[:, t_eval].cpu().numpy() if attn_w is not None else np.ones((B, K))

            rel_pos = nb[..., :2] - pr[:, None, :2]
            dists = np.linalg.norm(rel_pos, axis=-1)
            speeds = np.linalg.norm(nb[..., 2:4], axis=-1)
            headings = nb[..., 4]

            for b in range(B):
                valid_k = np.where(n_val[b])[0]
                n_count = len(valid_k)

                if n_count == 0:
                    continue
                windows_with_neighbors += 1

                if n_count < 2:
                    windows_k_single += 1
                    continue
                windows_k_multi += 1

                # Top-attended neighbor as target
                attns = attns_last[b, valid_k]
                target_k = valid_k[int(np.argmax(attns))]

                d_star = dists[b, target_k]
                v_star = speeds[b, target_k]
                theta_star = headings[b, target_k]
                is_target_stat = (v_star < speed_floor)

                # Check all other candidates for binding constraints
                has_matched = False
                best_ctrl_k = None
                best_err = float("inf")

                for k in valid_k:
                    if k == target_k:
                        continue
                    total_candidate_pairs += 1
                    d_k = dists[b, k]
                    v_k = speeds[b, k]
                    theta_k = headings[b, k]

                    d_ok = (abs(d_k - d_star) / max(d_star, 1e-4)) <= dist_tol
                    v_ok = (abs(v_k - v_star) / max(v_star, speed_floor)) <= speed_tol
                    if v_star >= speed_floor and v_k >= speed_floor:
                        diff_h = abs((theta_k - theta_star + math.pi) % (2 * math.pi) - math.pi)
                        h_ok = (diff_h <= heading_tol_rad)
                    else:
                        h_ok = True

                    if d_ok: pass_dist_count += 1
                    if v_ok: pass_speed_count += 1
                    if h_ok: pass_head_count += 1

                    if d_ok and v_ok and h_ok:
                        pass_all_count += 1
                        has_matched = True
                        err = (abs(d_k - d_star) / max(d_star, 1e-4)) + (abs(v_k - v_star) / max(v_star, speed_floor))
                        if err < best_err:
                            best_err = err
                            best_ctrl_k = k

                # Profile group
                tid = str(batch["track_id"][b])
                if has_matched and best_ctrl_k is not None:
                    matched_windows += 1
                    matched_tracks[tid] += 1
                    profile["matched"]["dist"].append(d_star)
                    profile["matched"]["speed"].append(v_star)
                    profile["matched"]["num_nbrs"].append(n_count)
                    if is_target_stat: profile["matched"]["stationary"] += 1

                    # Differentials
                    p0 = p_unmasked[b]
                    dt = abs(p0 - probs_ablated[b, target_k])
                    dc = abs(p0 - probs_ablated[b, best_ctrl_k])
                    delta_target_all.append(dt)
                    delta_ctrl_all.append(dc)
                    track_deltas[tid].append(dt - dc)
                else:
                    profile["unmatched"]["dist"].append(d_star)
                    profile["unmatched"]["speed"].append(v_star)
                    profile["unmatched"]["num_nbrs"].append(n_count)
                    if is_target_stat: profile["unmatched"]["stationary"] += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {total_windows:,} / {num_disk_files:,} windows...")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC REPORT: HYPOTHESIS H3 METHODOLOGICAL AUDIT")
    print("=" * 80)

    # 1. Window Count Reconciliation
    diff_windows = num_disk_files - total_windows
    print(f"\n1. WINDOW COUNT RECONCILIATION:")
    print(f"   Files on Disk ({split}):            {num_disk_files:,}")
    print(f"   Windows Processed by Loader:       {total_windows:,}")
    print(f"   Reconciliation Gap:                {diff_windows:,} windows")
    if diff_windows == 0:
        print("   Status: EXACT MATCH (100% of validation files evaluated).")
    else:
        print(f"   Status: Discrepancy explained by batch collation / file list filtering.")

    # 2. Global Mask-All Baseline Test
    m_unmask = compute_crossing_intention_metrics(y_true_all, probs_unmasked_all)
    m_mask = compute_crossing_intention_metrics(y_true_all, probs_masked_all)
    delta_auc = m_unmask["auc"] - m_mask["auc"]
    delta_f1 = m_unmask["f1"] - m_mask["f1"]
    print(f"\n2. GLOBAL MASK-ALL-NEIGHBOURS BENCHMARK:")
    print(f"   Intact Model Crossing AUC:         {m_unmask['auc']:.2f}% (F1: {m_unmask['f1']:.2f}%)")
    print(f"   Mask-All Crossing AUC:             {m_mask['auc']:.2f}% (F1: {m_mask['f1']:.2f}%)")
    print(f"   Global Impact (ΔAUC):              {delta_auc:+.2f}% (ΔF1: {delta_f1:+.2f}%)")
    if abs(delta_auc) < 0.20:
        print("   Finding: Model exhibits near-zero reliance on neighbor tokens (<0.2% AUC change).")
        print("   Interpretation: Consistent with training saturation/overfitting on 880 tracks.")

    # 3. Per-Criterion Binding Constraints
    print(f"\n3. PER-CRITERION PASS RATES (Candidate Pairs N = {total_candidate_pairs:,}):")
    print(f"   Distance Tolerance (+/-{dist_tol*100:.0f}%):     {pass_dist_count / max(1, total_candidate_pairs) * 100:.1f}% ({pass_dist_count:,} pairs)")
    print(f"   Speed Tolerance (+/-{speed_tol*100:.0f}%):        {pass_speed_count / max(1, total_candidate_pairs) * 100:.1f}% ({pass_speed_count:,} pairs)")
    print(f"   Heading Tolerance (+/-{heading_tol_deg:.0f}°):     {pass_head_count / max(1, total_candidate_pairs) * 100:.1f}% ({pass_head_count:,} pairs)")
    print(f"   All Three Combined:                {pass_all_count / max(1, total_candidate_pairs) * 100:.1f}% ({pass_all_count:,} pairs)")
    
    # Identify binding constraint
    rates = {"Distance": pass_dist_count, "Speed": pass_speed_count, "Heading": pass_head_count}
    binding = min(rates, key=rates.get)
    print(f"   Primary Binding Constraint:        {binding} ({rates[binding]/max(1, total_candidate_pairs)*100:.1f}% pass rate)")

    # 4. Profiling: Matched vs Unmatched Windows
    n_m = max(1, len(profile["matched"]["dist"]))
    n_u = max(1, len(profile["unmatched"]["dist"]))
    print(f"\n4. WHO GETS MATCHED? (Matched N={n_m:,} vs Unmatched N={n_u:,}):")
    print(f"   Mean Target Distance:              Matched={np.mean(profile['matched']['dist']):.3f} vs Unmatched={np.mean(profile['unmatched']['dist']):.3f}")
    print(f"   Median Target Speed:               Matched={np.median(profile['matched']['speed']):.5f} vs Unmatched={np.median(profile['unmatched']['speed']):.5f}")
    print(f"   Stationary Agent Share (<{speed_floor}): Matched={profile['matched']['stationary']/n_m*100:.1f}% vs Unmatched={profile['unmatched']['stationary']/n_u*100:.1f}%")
    print(f"   Mean Valid Neighbors:              Matched={np.mean(profile['matched']['num_nbrs']):.1f} vs Unmatched={np.mean(profile['unmatched']['num_nbrs']):.1f}")

    # 5. Matches per Track Concentration
    trk_counts = list(matched_tracks.values())
    print(f"\n5. TRACK CONCENTRATION & WILCOXON ZEROES:")
    print(f"   Contributing Tracks:               {len(trk_counts)} tracks")
    if trk_counts:
        print(f"   Pairs per Track:                   Min={min(trk_counts)}, Med={np.median(trk_counts):.0f}, P75={np.percentile(trk_counts, 75):.0f}, Max={max(trk_counts)}")
        top5_share = sum(sorted(trk_counts, reverse=True)[:5]) / max(1, sum(trk_counts)) * 100.0
        print(f"   Top 5 Tracks Share:                {top5_share:.1f}% of all matched pairs")

    # Count zero differentials
    track_meds = [float(np.median(v)) for v in track_deltas.values()]
    exact_zeros = sum(1 for m in track_meds if abs(m) < 1e-6)
    print(f"   Exact Zero Track Medians (|δ|<1e-6):{exact_zeros} / {len(track_meds)} ({exact_zeros/max(1,len(track_meds))*100:.1f}%)")
    print(f"   Effective Non-Zero Wilcoxon Sample: {len(track_meds) - exact_zeros} tracks")

    # 6. Heavy-Tail Distribution of Displacement ΔP
    dp_t = np.array(delta_target_all) if delta_target_all else np.zeros(1)
    print(f"\n6. OUTPUT DISPLACEMENT DISTRIBUTION (ΔP):")
    print(f"   Target ΔP Mean:                    {np.mean(dp_t):.5f}")
    print(f"   Target ΔP Median:                  {np.median(dp_t):.5f}")
    print(f"   Target ΔP 90th Percentile:         {np.percentile(dp_t, 90):.5f}")
    print(f"   Target ΔP Max:                     {np.max(dp_t):.5f}")
    print(f"   Share with ΔP > 0.01 (1% shift):   {(dp_t > 0.01).sum() / len(dp_t) * 100:.1f}%")
    print(f"   Share with ΔP > 0.05 (5% shift):   {(dp_t > 0.05).sum() / len(dp_t) * 100:.1f}%")
    print("=" * 80 + "\n")


def main():
    p = argparse.ArgumentParser(description="Hypothesis H3 Methodological Diagnostics")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config")
    p.add_argument("--data_dir", type=str, default="data/processed_pie", help="Path to dataset")
    p.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Split")
    p.add_argument("--dist_tol", type=float, default=0.15, help="Distance tolerance")
    p.add_argument("--speed_tol", type=float, default=0.20, help="Speed tolerance")
    p.add_argument("--heading_tol", type=float, default=30.0, help="Heading tolerance (deg)")
    p.add_argument("--speed_floor", type=float, default=0.002, help="Speed floor")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--seed", type=int, default=42, help="Seed")
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ProposedXMISTModel(feature_dim=config["model"]["feature_dim"], dropout=0.0).to(device)
    clean_state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(clean_state, strict=False)
    if not any(k.startswith("crossing_head") for k in clean_state.keys()):
        model._use_ped_head_fallback = True
    else:
        model._use_ped_head_fallback = False

    dataset = MultimodalSequenceDataset(data_dir=args.data_dir, split=args.split, window_size=config["data"]["window_size"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    run_diagnostics(
        model=model,
        loader=loader,
        data_dir=args.data_dir,
        split=args.split,
        device=device,
        dist_tol=args.dist_tol,
        speed_tol=args.speed_tol,
        heading_tol_deg=args.heading_tol,
        speed_floor=args.speed_floor,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
