#!/usr/bin/env python3
"""
Ego-motion analysis for the PIE TTE benchmark (exploratory, post protocol lock; specified 2026-11-05
before being run).

Uses PIE's on-board vehicle data (annotations_vehicle/setXX/video_XXXX_obd.xml: per-frame OBD_speed in
km/h and gyroZ yaw rate) over each window's 16 observed frames.

Pre-specified analyses (fixed before running):
  A. Ego speed as a predictor: logistic regression on [mean, last-frame] OBD speed, fit on train,
     test AUC with 95% track-bootstrap CI; compared with pedestrian apparent speed [mean, last] and both.
  B. Apparent-speed confound: Spearman correlation between pedestrian apparent speed and ego speed;
     AUC of the apparent-speed model within ego-speed strata.
  C. Interaction where the shortcut is weak: seed-ensemble ΔAUC (traj_full - traj_isolated) and both
     models' AUC within ego-speed strata, from the locked H2 run's saved test predictions.
  Strata (window-mean OBD speed): stopped < 5 km/h, slow 5-20 km/h, moving >= 20 km/h.
  Strata with < 25 tracks or a single class are reported descriptively only.

Usage:
  python tools/ego_motion_analysis.py --data_dir data/processed_pie_tte --vehicle_dir <.../annotations_vehicle> \
      --predictions experiments/results/cvmi_test/predictions_test.npz --output_dir experiments/results/cvmi_ego
"""
import os
import sys
import glob
import json
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

STRATA = [("stopped (<5 km/h)", 0.0, 5.0), ("slow (5-20 km/h)", 5.0, 20.0), ("moving (>=20 km/h)", 20.0, np.inf)]
N_MIN = 25


def find_vehicle_dir(root):
    hits = sorted(glob.glob(os.path.join(root, "**", "annotations_vehicle"), recursive=True))
    if not hits:
        sys.exit(f"No annotations_vehicle folder under {root}; pass --vehicle_dir")
    return hits[0]


def load_obd(vehicle_dir):
    """{(set_id, video_id): (speed[frame], |yaw rate|[frame])}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(vehicle_dir, "set*", "*_obd.xml"))):
        set_id = os.path.basename(os.path.dirname(f))
        vid = os.path.basename(f).replace("_obd.xml", "")
        frames = ET.parse(f).getroot().findall("frame")
        n = max(int(fr.get("id")) for fr in frames) + 1
        spd, yaw = np.full(n, np.nan), np.full(n, np.nan)
        for fr in frames:
            i = int(fr.get("id"))
            spd[i] = float(fr.get("OBD_speed", "nan"))
            yaw[i] = abs(float(fr.get("gyroZ", "nan")))
        out[(set_id, vid)] = (spd, yaw)
    if not out:
        sys.exit(f"No *_obd.xml files in {vehicle_dir}")
    return out


def load_split(data_dir, split, obd, obs_len=16):
    files = sorted(glob.glob(os.path.join(data_dir, split, "*.npz")))
    if not files:
        sys.exit(f"No .npz files in {os.path.join(data_dir, split)}")
    rows = {k: [] for k in ["y", "track", "ego_mean", "ego_last", "yaw_mean", "ped_mean", "ped_last"]}
    missing = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        md = d["metadata"].item()
        key = (md["set_id"], md["video_id"])
        end = int(md["obs_end_frame"])
        fr = np.arange(end - obs_len + 1, end + 1)
        spd, yaw = obd.get(key, (None, None))
        if spd is None or fr[-1] >= len(spd) or np.isnan(spd[fr]).any():
            missing += 1
            ego = np.full(obs_len, np.nan); yw = np.full(obs_len, np.nan)
        else:
            ego, yw = spd[fr], yaw[fr]
        v = np.linalg.norm(d["trajectory"][:, 2:4], axis=-1)
        rows["y"].append(int(d["cross"]))
        rows["track"].append(f'{md["set_id"]}_{md["video_id"]}_{md["ped_id"]}')
        rows["ego_mean"].append(np.mean(ego)); rows["ego_last"].append(ego[-1]); rows["yaw_mean"].append(np.mean(yw))
        rows["ped_mean"].append(v.mean()); rows["ped_last"].append(v[-1])
    out = {k: np.array(v) for k, v in rows.items()}
    out["missing_obd"] = missing
    return out


def boot_auc(y, p, tracks, n_boot, rng):
    uniq = np.unique(tracks)
    groups = [np.where(tracks == u)[0] for u in uniq]
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], p[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 100 else [np.nan, np.nan]


def boot_delta(y, pa, pb, tracks, n_boot, rng):
    uniq = np.unique(tracks)
    groups = [np.where(tracks == u)[0] for u in uniq]
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 100 else [np.nan, np.nan]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_pie_tte")
    ap.add_argument("--vehicle_dir", default=None)
    ap.add_argument("--pie_root", default="data/raw/PIE")
    ap.add_argument("--predictions", default="experiments/results/cvmi_test/predictions_test.npz")
    ap.add_argument("--output_dir", default="experiments/results/cvmi_ego")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    obd = load_obd(args.vehicle_dir or find_vehicle_dir(args.pie_root))
    tr = load_split(args.data_dir, "train", obd)
    te = load_split(args.data_dir, "test", obd)
    print(f"OBD missing windows: train {tr['missing_obd']}, test {te['missing_obd']}")
    ok_tr = np.isfinite(tr["ego_mean"]); ok_te = np.isfinite(te["ego_mean"])
    res = {"note": "exploratory, post-lock; analyses and strata specified before running",
           "missing_obd": {"train": int(tr["missing_obd"]), "test": int(te["missing_obd"])}}

    # Descriptives by class
    res["ego_speed_by_class_test"] = {
        name: {"windows": int(((te["y"] == c) & ok_te).sum()),
               "ego_speed_p10_p50_p90_kmh": [round(float(v), 1) for v in np.percentile(te["ego_mean"][(te["y"] == c) & ok_te], [10, 50, 90])],
               "share_stopped": round(float(np.mean(te["ego_mean"][(te["y"] == c) & ok_te] < 5.0)), 3)}
        for c, name in [(1, "crossing"), (0, "not_crossing")]}

    # A. Ego speed vs pedestrian apparent speed as predictors
    feats = {"ego_speed": ["ego_mean", "ego_last"], "ped_apparent_speed": ["ped_mean", "ped_last"],
             "ego_plus_ped_speed": ["ego_mean", "ego_last", "ped_mean", "ped_last"],
             "ego_speed_and_yaw": ["ego_mean", "ego_last", "yaw_mean"]}
    res["A_auc_test"] = {}
    preds_lr = {}
    for name, cols in feats.items():
        Xtr = np.column_stack([tr[c][ok_tr] for c in cols])
        Xte = np.column_stack([te[c][ok_te] for c in cols])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, tr["y"][ok_tr])
        p = clf.predict_proba(Xte)[:, 1]
        preds_lr[name] = p
        res["A_auc_test"][name] = {"auc": float(roc_auc_score(te["y"][ok_te], p)),
                                   "ci": boot_auc(te["y"][ok_te], p, te["track"][ok_te], args.n_boot, rng)}
        print(f"A {name:22s} AUC {res['A_auc_test'][name]['auc']:.3f} {res['A_auc_test'][name]['ci']}")

    # B. Confound: correlation and within-stratum AUC of apparent speed
    rho = spearmanr(te["ped_mean"][ok_te], te["ego_mean"][ok_te])
    res["B_spearman_ped_vs_ego_speed"] = {"rho": float(rho.correlation), "p": float(rho.pvalue)}
    y_ok, t_ok, e_ok = te["y"][ok_te], te["track"][ok_te], te["ego_mean"][ok_te]
    res["B_ped_speed_auc_by_stratum"] = {}
    for sname, lo, hi in STRATA:
        m = (e_ok >= lo) & (e_ok < hi)
        n_tr = len(np.unique(t_ok[m]))
        row = {"windows": int(m.sum()), "tracks": int(n_tr), "crossing_share": float(y_ok[m].mean()) if m.any() else None,
               "descriptive_only": bool(n_tr < N_MIN)}
        if m.sum() and len(np.unique(y_ok[m])) == 2:
            p = preds_lr["ped_apparent_speed"][m]
            row["auc"] = float(roc_auc_score(y_ok[m], p))
            row["ci"] = boot_auc(y_ok[m], p, t_ok[m], args.n_boot, rng)
        res["B_ped_speed_auc_by_stratum"][sname] = row

    # C. Deep models within ego-speed strata (saved predictions of the locked run)
    if os.path.exists(args.predictions):
        P = np.load(args.predictions, allow_pickle=True)
        if not np.array_equal(np.asarray(P["track_ids"]), te["track"]) or not np.array_equal(P["y"], te["y"]):
            sys.exit("Saved predictions are not aligned with the test files (order or labels differ).")
        full = np.mean([P[k] for k in P.files if k.startswith("full_seed") and not k.endswith(("_removed", "_shuffled"))], axis=0)
        iso = np.mean([P[k] for k in P.files if k.startswith("iso_seed")], axis=0)
        res["C_deep_by_stratum"] = {}
        for sname, lo, hi in STRATA:
            m = ok_te & (te["ego_mean"] >= lo) & (te["ego_mean"] < hi)
            n_tr = len(np.unique(te["track"][m]))
            row = {"windows": int(m.sum()), "tracks": int(n_tr), "descriptive_only": bool(n_tr < N_MIN)}
            if m.sum() and len(np.unique(te["y"][m])) == 2:
                yy, tt = te["y"][m], te["track"][m]
                row.update({"auc_isolated": float(roc_auc_score(yy, iso[m])), "auc_full": float(roc_auc_score(yy, full[m])),
                            "delta_auc": float(roc_auc_score(yy, full[m]) - roc_auc_score(yy, iso[m])),
                            "delta_ci": boot_delta(yy, full[m], iso[m], tt, args.n_boot, rng)})
            res["C_deep_by_stratum"][sname] = row
            print(f"C {sname:20s} {row}")
    else:
        print(f"No predictions at {args.predictions}; skipping C")

    with open(os.path.join(args.output_dir, "ego_motion.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    md = ["# Ego-motion analysis (PIE test, exploratory)", "", "```json", json.dumps(res, indent=2, default=float), "```"]
    open(os.path.join(args.output_dir, "ego_motion.md"), "w").write("\n".join(md) + "\n")
    print(f"Saved {args.output_dir}/ego_motion.md")


if __name__ == "__main__":
    main()
