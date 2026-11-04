#!/usr/bin/env python3
"""
Shortcut / simple-feature baselines for the PIE TTE benchmark (exploratory, post protocol lock).

Explains how much of the ~0.93 AUC of the trajectory models is already available from simple
box/position cues. Non-crossing windows end 1-2 s before the LAST annotated frame of a track, which
is often where the pedestrian leaves the image, so position / box size near the image edge may be a
strong cue. Each feature set is fitted on train and scored on the requested splits with logistic
regression and gradient boosting; AUC comes with a 95% track-clustered bootstrap CI.

Usage:
  python tools/shortcut_baselines.py --data_dir data/processed_pie_tte --splits val test \
      --output_dir experiments/results/cvmi_baselines
"""
import os
import sys
import glob
import json
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def load_split(data_dir, split):
    files = sorted(glob.glob(os.path.join(data_dir, split, "*.npz")))
    if not files:
        sys.exit(f"No .npz files in {os.path.join(data_dir, split)}")
    traj, bbox, y, tracks, k_last = [], [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        md = d["metadata"].item()
        traj.append(d["trajectory"])
        bbox.append(d["bbox"])
        y.append(int(d["cross"]))
        tracks.append(f'{md["set_id"]}_{md["video_id"]}_{md["ped_id"]}')
        k_last.append(int(d["neighbor_mask"][-1].sum()))
    return {"traj": np.stack(traj), "bbox": np.stack(bbox), "y": np.array(y),
            "tracks": np.array(tracks), "k_last": np.array(k_last)}


def features(d):
    """Named feature sets, all derived from the observed 16 frames only."""
    t, b = d["traj"], d["bbox"]  # t: [N,16,5] x,y,vx,vy,heading | b: [N,16,4] x,y,w,h (normalised)
    x, y = t[:, -1, 0], t[:, -1, 1]
    w, h = b[:, -1, 2], b[:, -1, 3]
    edge = np.minimum(x, 1.0 - x)
    scale_rate = (b[:, -1, 3] - b[:, 0, 3]) / np.maximum(b[:, 0, 3], 1e-6)  # box growth ~ ego approach
    mean_v = t[:, :, 2:4].mean(1)
    std_v = t[:, :, 2:4].std(1)
    disp = t[:, -1, :2] - t[:, 0, :2]
    speed = np.linalg.norm(t[:, :, 2:4], axis=-1)
    return {
        "speed_only": np.c_[speed.mean(1), speed[:, -1]],
        "motion_only": np.c_[scale_rate, mean_v, std_v, disp, speed.mean(1), speed[:, -1]],
        "box_size_only": np.c_[w, h],
        "position_only": np.c_[x, y, edge],
        "position_box": np.c_[x, y, edge, w, h],
        "position_box_motion": np.c_[x, y, edge, w, h, scale_rate, mean_v, std_v, disp],
        "full_window_flat": np.c_[t.reshape(len(t), -1), b.reshape(len(b), -1)],
    }


def bootstrap_auc(y, p, tracks, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(tracks)
    idx_by = {u: np.where(tracks == u)[0] for u in uniq}
    vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx_by[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(np.unique(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], p[ii]))
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"), float("nan"))
    return float(lo), float(hi)


def describe(d):
    """Per-class distribution of the suspected shortcut cues at the last observed frame."""
    t, b, y = d["traj"], d["bbox"], d["y"]
    edge = np.minimum(t[:, -1, 0], 1.0 - t[:, -1, 0])
    out = {}
    for c, name in [(1, "crossing"), (0, "not_crossing")]:
        m = y == c
        out[name] = {
            "windows": int(m.sum()),
            "edge_dist_p10_p50_p90": [round(float(v), 3) for v in np.percentile(edge[m], [10, 50, 90])],
            "box_h_p10_p50_p90": [round(float(v), 3) for v in np.percentile(b[m, -1, 3], [10, 50, 90])],
            "speed_p50": round(float(np.median(np.linalg.norm(t[m, :, 2:4], axis=-1))), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_pie_tte")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--output_dir", default="experiments/results/cvmi_baselines")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train = load_split(args.data_dir, "train")
    evals = {s: load_split(args.data_dir, s) for s in args.splits}
    ftr = features(train)
    fev = {s: features(d) for s, d in evals.items()}
    models = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gboost": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=args.seed),
    }

    results = {"note": "exploratory, run after protocol lock; not part of the pre-registered H2 test",
               "cue_distributions": {s: describe(d) for s, d in [("train", train), *evals.items()]},
               "auc": {}}
    for fname in ftr:
        for mname, mk in models.items():
            clf = mk().fit(ftr[fname], train["y"])
            row = {}
            for s, d in evals.items():
                p = clf.predict_proba(fev[s][fname])[:, 1]
                lo, hi = bootstrap_auc(d["y"], p, d["tracks"], args.n_boot, args.seed)
                row[s] = {"auc": float(roc_auc_score(d["y"], p)), "ci": [lo, hi]}
            results["auc"][f"{fname}/{mname}"] = row
            print(f"{fname:22s} {mname:7s} " + "  ".join(
                f"{s}: {r['auc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" for s, r in row.items()))

    with open(os.path.join(args.output_dir, "baselines.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    L = ["# Simple-feature baselines (exploratory, post-lock)", "",
         "| Features | Model | " + " | ".join(f"{s} AUC [95% CI]" for s in args.splits) + " |",
         "|---|---|" + "---|" * len(args.splits)]
    for key, row in results["auc"].items():
        f, m = key.split("/")
        L.append(f"| {f} | {m} | " + " | ".join(
            f"{row[s]['auc']:.3f} [{row[s]['ci'][0]:.3f}, {row[s]['ci'][1]:.3f}]" for s in args.splits) + " |")
    L += ["", "## Cue distributions at last observed frame", "", "```json",
          json.dumps(results["cue_distributions"], indent=2), "```"]
    open(os.path.join(args.output_dir, "baselines.md"), "w").write("\n".join(L) + "\n")
    print(f"Saved {args.output_dir}/baselines.md")


if __name__ == "__main__":
    main()
