#!/usr/bin/env python3
"""
H1-H4 of docs/preregistered_evaluation_protocol.md (28 Sep 2026) for the CVMI trajectory models.

H1  Neighbour availability: share of windows/tracks with K=0, K=1, K>=2 (K = valid neighbours at the
    last observed frame). Prediction: K>=1 in a minority (<50%) of windows.
H2  Stratified utility, completing the CVMI H2 run: ΔAUC and ΔF1 (threshold 0.5) per stratum for
    full vs isolated and full vs neighbours-removed, seed-ensemble probabilities, track bootstrap.
H3  Targeted vs matched-control removal (windows with >=2 neighbours). Target = neighbour with the
    highest seed-averaged interaction attention (mean over the 16 frames). Control = random pick among
    other neighbours with |Δdist| <= max(0.15*dist, 0.05), |Δspeed| <= 0.20*speed, |Δheading| <= 30°
    (window means of distance to the pedestrian, neighbour speed, heading of mean neighbour velocity).
    δ = |Δp(remove target)| - |Δp(remove control)|. Oracle = max |Δp| over single-neighbour removals.
    Test: Wilcoxon signed-rank on per-track mean δ; 95% track bootstrap CI of the median δ.
H4  Urgency: each (window, neighbour) unit, neighbour trajectory co-rescaled around its last observed
    position, p'(t) = p(T) - α (p(T) - p(t)), v' = α v, α in {0.5, 0.75, 1, 1.25, 1.5, 2}.
    Units are closing / separating by v_closing at the last frame. Per unit Spearman ρ(α, p_cross);
    expected sign = ρ < -0.5. Placebo: identical rescaling after rotating the neighbour's displacement
    and velocity by a fixed random angle (scrambles the approach direction, keeps the motion
    magnitude). McNemar exact test on closing units: expected sign (real) vs expected sign (placebo).
OOD Range violations and Mahalanobis distance of perturbed neighbour features vs training neighbours.
Holm-Bonferroni over the H1-H4 p-values. Strata with < 25 tracks are flagged descriptive-only.

Exploratory relative to the CVMI lock (docs/cvmi_protocol_lock.md), which pre-registered only H2;
H1/H3/H4 follow the earlier 28 Sep protocol, written before these models were trained.

Usage:
  python tools/eval_mechanism.py --data_dir data/processed_pie_tte --split test \
      --full_ckpts C/traj_full_seed{42,100,2024}/checkpoint_best.pt \
      --iso_ckpts  C/traj_isolated_seed{42,100,2024}/checkpoint_best.pt \
      --output_dir experiments/results/cvmi_mechanism_test
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon, binomtest
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_paper import load_split, load_model, predict, paired_bootstrap, git_state  # noqa: E402
from evaluation.hypothesis_testing import holm_bonferroni, compute_mahalanobis_ood  # noqa: E402

ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
N_MIN = 25


def ensemble_p(models, batch, device, na, nm):
    return np.mean([predict(m, batch, device, neighbor_override=(na, nm)) for m in models], axis=0)


@torch.no_grad()
def ensemble_attention(models, batch, device, bs=256):
    """Seed-averaged interaction attention per neighbour slot, mean over observed frames -> [n, N]."""
    out = []
    for m in models:
        parts = []
        for s in range(0, batch["trajectory"].shape[0], bs):
            b = {k: (v[s:s + bs].to(device) if torch.is_tensor(v) else v[s:s + bs]) for k, v in batch.items()}
            w = m(b)["interaction_weights"]
            if w is None:
                raise RuntimeError("Model returned no interaction_weights (is this a traj_full checkpoint?)")
            parts.append(w.float().mean(1).cpu().numpy())
        out.append(np.concatenate(parts))
    return np.mean(out, axis=0)


def track_bootstrap(stat_fn, idx_by_track, n_boot, rng):
    groups = list(idx_by_track.values())
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        v = stat_fn(ii)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.array(vals)


def h1(k_last, tracks, n_boot, rng):
    uniq = np.unique(tracks)
    idx_by = {t: np.where(tracks == t)[0] for t in uniq}
    share = float(np.mean(k_last >= 1))
    boots = track_bootstrap(lambda ii: float(np.mean(k_last[ii] >= 1)), idx_by, n_boot, rng)
    trk_any = np.array([np.any(k_last[idx_by[t]] >= 1) for t in uniq])
    return {
        "windows": int(len(k_last)), "tracks": int(len(uniq)),
        "window_share": {"K0": float(np.mean(k_last == 0)), "K1": float(np.mean(k_last == 1)),
                         "K2plus": float(np.mean(k_last >= 2))},
        "window_share_K1plus": share,
        "window_share_K1plus_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "track_share_any_K1plus": float(trk_any.mean()),
        # One-sided bootstrap p for the prediction "K>=1 share < 0.5"
        "p_one_sided": float(max(np.mean(boots >= 0.5), 1.0 / len(boots))),
        "prediction_supported": bool(np.percentile(boots, 97.5) < 0.5),
    }


def paired_f1_bootstrap(y, pa, pb, tracks, n_boot, rng):
    uniq = np.unique(tracks)
    idx_by = {t: np.where(tracks == t)[0] for t in uniq}
    f = lambda ii: f1_score(y[ii], pa[ii] >= 0.5, zero_division=0) - f1_score(y[ii], pb[ii] >= 0.5, zero_division=0)
    point = f(np.arange(len(y)))
    boots = track_bootstrap(lambda ii: f(ii) if len(np.unique(y[ii])) == 2 else None, idx_by, n_boot, rng)
    if len(boots) < 100:
        return {"delta_f1": float(point), "ci": [float("nan")] * 2, "n_tracks": int(len(uniq))}
    return {"delta_f1": float(point), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_tracks": int(len(uniq))}


def h2(y, k_last, tracks, p_full, p_iso, p_removed, n_boot, seed, rng):
    out = {}
    for name, pb in [("full_vs_isolated", p_iso), ("full_vs_neighbours_removed", p_removed)]:
        if pb is None:
            continue
        for stratum, m in [("all", np.ones_like(y, bool)), ("K=0", k_last == 0), ("K>=1", k_last >= 1)]:
            n_tr = len(np.unique(tracks[m]))
            row = {"n_tracks": int(n_tr), "descriptive_only": bool(n_tr < N_MIN)}
            if len(np.unique(y[m])) == 2:
                row["auc"] = paired_bootstrap(y[m], p_full[m], pb[m], tracks[m], n_boot, seed)
                row["f1"] = paired_f1_bootstrap(y[m], p_full[m], pb[m], tracks[m], n_boot, rng)
            out[f"{name} [{stratum}]"] = row
    k1 = out.get("full_vs_isolated [K>=1]", {}).get("auc", {})
    k0 = out.get("full_vs_isolated [K=0]", {}).get("auc", {})
    supported = bool(k1.get("ci", [np.nan])[0] > 0 and (np.isnan(k0.get("ci", [np.nan])[0]) or k0["ci"][0] <= 0 <= k0["ci"][1]))
    return {"strata": out, "p_K1plus_full_vs_isolated": k1.get("p_two_sided", float("nan")), "supported": supported}


def neighbour_stats(traj, na, nm):
    """Window-mean distance to the pedestrian, neighbour speed, heading of mean neighbour velocity."""
    dist = np.linalg.norm(traj[:, :, None, :2] - na[..., :2], axis=-1).mean(1)
    speed = np.linalg.norm(na[..., 2:4], axis=-1).mean(1)
    mv = na[..., 2:4].mean(1)
    heading = np.arctan2(mv[..., 1], mv[..., 0])
    return dist, speed, heading


def remove_slot(na, nm, rows, slots):
    na2, nm2 = na.clone(), nm.clone()
    na2[rows, :, slots, :] = 0.0
    nm2[rows, :, slots] = False
    return na2, nm2


def h3(models, batch, device, y, tracks, n_boot, rng):
    traj = batch["trajectory"].numpy()
    na, nm = batch["neighbor_agents"], batch["neighbor_mask"]
    valid = nm[:, -1].numpy()
    rows = np.where(valid.sum(1) >= 2)[0]
    if len(rows) == 0:
        return {"n_windows": 0, "note": "no windows with >= 2 neighbours"}
    attn = ensemble_attention(models, batch, device)
    attn = np.where(valid, attn, -np.inf)
    dist, speed, heading = neighbour_stats(traj, na.numpy(), nm.numpy())
    p0 = ensemble_p(models, batch, device, na, nm)

    # |Δp| for removing each slot on its own (oracle and lookup for target/control)
    N = valid.shape[1]
    dp = np.full((len(y), N), np.nan)
    for k in range(N):
        r = np.where(valid[:, k])[0]
        r = r[np.isin(r, rows)]
        if len(r) == 0:
            continue
        na2, nm2 = remove_slot(na, nm, torch.as_tensor(r), k)
        dp[r, k] = np.abs(ensemble_p(models, batch, device, na2, nm2)[r] - p0[r])

    delta, d_tgt, d_ctl, d_orc, tgt_is_orc, kept, n_no_ctrl = [], [], [], [], [], [], 0
    for i in rows:
        t = int(np.argmax(attn[i]))
        ang = np.abs(np.angle(np.exp(1j * (heading[i] - heading[i, t]))))
        cand = [k for k in range(N) if valid[i, k] and k != t
                and abs(dist[i, k] - dist[i, t]) <= max(0.15 * dist[i, t], 0.05)
                and abs(speed[i, k] - speed[i, t]) <= 0.20 * speed[i, t]
                and ang[k] <= np.pi / 6]
        if not cand:
            n_no_ctrl += 1
            continue
        c = int(rng.choice(cand))
        delta.append(dp[i, t] - dp[i, c])
        d_tgt.append(dp[i, t]); d_ctl.append(dp[i, c]); d_orc.append(np.nanmax(dp[i]))
        tgt_is_orc.append(t == int(np.nanargmax(dp[i])))
        kept.append(i)
    res = {"n_windows_ge2": int(len(rows)), "n_matched": int(len(kept)), "n_no_control": int(n_no_ctrl)}
    if len(kept) < 10:
        res["note"] = "too few matched windows for inference"
        return res
    delta, kept = np.array(delta), np.array(kept)
    tr = tracks[kept]
    uniq = np.unique(tr)
    per_track = np.array([delta[tr == u].mean() for u in uniq])
    idx_by = {u: np.where(tr == u)[0] for u in uniq}
    boots = track_bootstrap(lambda ii: float(np.median(delta[ii])), idx_by, n_boot, rng)
    nz = per_track[per_track != 0]
    w_p = float(wilcoxon(nz).pvalue) if len(nz) >= 10 else float("nan")
    res.update({
        "n_tracks": int(len(uniq)), "descriptive_only": bool(len(uniq) < N_MIN),
        "median_delta": float(np.median(delta)),
        "median_delta_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "mean_abs_dp_target": float(np.mean(d_tgt)), "mean_abs_dp_control": float(np.mean(d_ctl)),
        "mean_abs_dp_oracle": float(np.mean(d_orc)), "target_is_oracle_share": float(np.mean(tgt_is_orc)),
        "wilcoxon_p_track_means": w_p,
    })
    lo = res["median_delta_ci"][0]
    res["supported"] = bool(res["median_delta"] > 0 and lo > 0 and np.isfinite(w_p) and w_p < 0.05)
    return res


def rescale(na, rows, k, alpha, theta=None):
    """Co-rescale neighbour slot k of the given rows around its last observed position."""
    na2 = na.clone()
    seg = na2[rows, :, k, :]                       # [r, T, 5]
    pT = seg[:, -1:, :2]
    disp = seg[:, :, :2] - pT
    vel = seg[:, :, 2:4]
    if theta is not None:
        c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
        rot = lambda v: torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)
        disp, vel = rot(disp), rot(vel)
        seg[:, :, 4] = torch.atan2(torch.sin(seg[:, :, 4] + theta[:, None]), torch.cos(seg[:, :, 4] + theta[:, None]))
    seg[:, :, :2] = pT + alpha * disp
    seg[:, :, 2:4] = alpha * vel
    na2[rows, :, k, :] = seg
    return na2


def h4(models, batch, device, train_feats, rng):
    traj = batch["trajectory"]
    na, nm = batch["neighbor_agents"], batch["neighbor_mask"]
    valid = nm[:, -1].numpy()
    N = valid.shape[1]
    units = {"closing": [], "separating": []}
    rho_real, rho_plac, ood = {}, {}, []
    for k in range(N):
        rows = np.where(valid[:, k])[0]
        if len(rows) == 0:
            continue
        r_t = torch.as_tensor(rows)
        rel = traj[rows, -1, :2] - na[rows, -1, k, :2]
        vrel = traj[rows, -1, 2:4] - na[rows, -1, k, 2:4]
        v_close = (-(rel * vrel).sum(-1) / (rel.norm(dim=-1) + 1e-5)).numpy()
        theta = torch.as_tensor(rng.uniform(np.pi / 2, 3 * np.pi / 2, len(rows)), dtype=na.dtype)
        P_real = np.stack([ensemble_p(models, batch, device, rescale(na, r_t, k, a), nm)[rows] for a in ALPHAS], 1)
        P_plac = np.stack([ensemble_p(models, batch, device, rescale(na, r_t, k, a, theta), nm)[rows] for a in ALPHAS], 1)
        for a in (0.5, 2.0):
            pert = rescale(na, r_t, k, a)[rows, :, k, :].reshape(-1, 5).numpy()
            ood.append((a, pert))
        for j, i in enumerate(rows):
            key = (int(i), k)
            rho_real[key] = spearmanr(ALPHAS, P_real[j]).correlation if np.ptp(P_real[j]) > 0 else 0.0
            rho_plac[key] = spearmanr(ALPHAS, P_plac[j]).correlation if np.ptp(P_plac[j]) > 0 else 0.0
            units["closing" if v_close[j] > 0 else "separating"].append(key)

    def summary(keys):
        r = np.array([rho_real[u] for u in keys])
        p = np.array([rho_plac[u] for u in keys])
        if len(r) == 0:
            return {"n_units": 0}
        return {"n_units": int(len(r)), "median_rho": float(np.median(r)), "share_expected_sign": float(np.mean(r < -0.5)),
                "placebo_median_rho": float(np.median(p)), "placebo_share_expected_sign": float(np.mean(p < -0.5)),
                "share_any_response": float(np.mean(r != 0))}
    res = {g: summary(u) for g, u in units.items()}
    cl = units["closing"]
    if cl:
        real = np.array([rho_real[u] < -0.5 for u in cl])
        plac = np.array([rho_plac[u] < -0.5 for u in cl])
        b, c = int(np.sum(real & ~plac)), int(np.sum(~real & plac))
        p = float(binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c > 0 else 1.0
        res["mcnemar_closing"] = {"real_only": b, "placebo_only": c, "p_one_sided": p}
        res["supported"] = bool(p < 0.05 and res["closing"]["share_expected_sign"] > res["closing"]["placebo_share_expected_sign"])
    base = na[nm].reshape(-1, 5).numpy()
    res["ood"] = {"unperturbed_test": compute_mahalanobis_ood(train_feats, base)}
    for a in (0.5, 2.0):
        res["ood"][f"alpha_{a}"] = compute_mahalanobis_ood(train_feats, np.concatenate([p for aa, p in ood if aa == a]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--full_ckpts", nargs="+", required=True)
    ap.add_argument("--iso_ckpts", nargs="*", default=[])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    batch = load_split(args.data_dir, args.split)
    y = batch["cross"].numpy()
    tracks = np.asarray(batch["track_id"])
    k_last = batch["neighbor_mask"][:, -1].sum(1).numpy()
    tr = load_split(args.data_dir, "train")
    train_feats = tr["neighbor_agents"][tr["neighbor_mask"]].reshape(-1, 5).numpy()

    full = []
    for p in args.full_ckpts:
        m, variant, *_ = load_model(p, device)
        if variant != "traj_full":
            sys.exit(f"{p} is {variant}, expected traj_full")
        full.append(m)
    iso = []
    for p in args.iso_ckpts:
        m, variant, *_ = load_model(p, device)
        if variant != "traj_isolated":
            sys.exit(f"{p} is {variant}, expected traj_isolated")
        iso.append(m)

    na, nm = batch["neighbor_agents"], batch["neighbor_mask"]
    p_full = ensemble_p(full, batch, device, na, nm)
    p_removed = ensemble_p(full, batch, device, torch.zeros_like(na), torch.zeros_like(nm))
    p_iso = ensemble_p(iso, batch, device, na, nm) if iso else None

    print("H1 ..."); r1 = h1(k_last, tracks, args.n_boot, rng)
    print("H2 ..."); r2 = h2(y, k_last, tracks, p_full, p_iso, p_removed, args.n_boot, args.seed, rng)
    print("H3 ..."); r3 = h3(full, batch, device, y, tracks, args.n_boot, rng)
    print("H4 ..."); r4 = h4(full, batch, device, train_feats, rng)

    names = ["H1", "H2", "H3", "H4"]
    pvals = [r1["p_one_sided"], r2["p_K1plus_full_vs_isolated"], r3.get("wilcoxon_p_track_means", float("nan")),
             r4.get("mcnemar_closing", {}).get("p_one_sided", float("nan"))]
    finite = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    holm = dict(zip([names[i] for i in finite], holm_bonferroni([pvals[i] for i in finite])))

    summary = {"note": "H1/H3/H4 per the 28 Sep 2026 protocol; exploratory relative to the CVMI lock (H2 only)",
               "split": args.split, "data_dir": args.data_dir, "git": git_state(),
               "full_ckpts": args.full_ckpts, "iso_ckpts": args.iso_ckpts,
               "H1": r1, "H2": r2, "H3": r3, "H4": r4,
               "holm": {k: {"p_raw": pvals[names.index(k)], "p_adj": v[0], "reject": v[1]} for k, v in holm.items()}}
    with open(os.path.join(args.output_dir, f"mechanism_{args.split}.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    f = lambda v: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.4f}"
    ci = lambda c: f"[{f(c[0])}, {f(c[1])}]"
    L = [f"# H1-H4 ({args.split}, {len(y)} windows, {len(np.unique(tracks))} tracks)", "",
         "## H1 neighbour availability",
         f"K=0 {r1['window_share']['K0']:.3f} | K=1 {r1['window_share']['K1']:.3f} | K>=2 {r1['window_share']['K2plus']:.3f} of windows; "
         f"K>=1 share {r1['window_share_K1plus']:.3f} {ci(r1['window_share_K1plus_ci'])}; "
         f"tracks with any K>=1 window {r1['track_share_any_K1plus']:.3f}. "
         f"Prediction (K>=1 < 50%): {'supported' if r1['prediction_supported'] else 'not supported'}", "",
         "## H2 stratified utility (seed ensemble)", "",
         "| Comparison | ΔAUC [95% CI] | ΔF1 [95% CI] | tracks |", "|---|---|---|---|"]
    for k, row in r2["strata"].items():
        a, fl = row.get("auc", {}), row.get("f1", {})
        L.append(f"| {k} | {f(a.get('delta_auc'))} {ci(a.get('ci', [None, None]))} | "
                 f"{f(fl.get('delta_f1'))} {ci(fl.get('ci', [None, None]))} | {row['n_tracks']}{' (descriptive)' if row['descriptive_only'] else ''} |")
    L += [f"H2 (K>=1 CI > 0 and K=0 CI contains 0): {'supported' if r2['supported'] else 'not supported'}", "",
          "## H3 targeted vs matched-control removal", "```json", json.dumps(r3, indent=2, default=float), "```",
          "## H4 urgency response", "```json", json.dumps(r4, indent=2, default=float), "```",
          "## Holm-Bonferroni (alpha 0.05)", "```json", json.dumps(summary["holm"], indent=2, default=float), "```"]
    open(os.path.join(args.output_dir, f"mechanism_{args.split}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"Saved {args.output_dir}/mechanism_{args.split}.md")


if __name__ == "__main__":
    main()
