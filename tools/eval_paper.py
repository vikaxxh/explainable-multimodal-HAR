#!/usr/bin/env python3
"""
CVMI 2027 paper evaluation: interaction (traj_full) vs isolated (traj_isolated) on PIE TTE.

For every checkpoint: crossing probabilities on the split. For interaction checkpoints also two
controls: (a) all neighbours removed, (b) neighbours shuffled across windows of OTHER tracks.
Reports per-seed metrics (mean ± std), seed-ensemble metrics, and paired track-clustered bootstrap
CIs for ΔAUC (full - isolated), overall and by neighbour count K (last observed frame).

Usage:
  python tools/eval_paper.py --data_dir data/processed_pie_tte --split val \
      --full_ckpts experiments/checkpoints/traj_full_seed{42,100,2024}/checkpoint_best.pt \
      --iso_ckpts  experiments/checkpoints/traj_isolated_seed{42,100,2024}/checkpoint_best.pt \
      --output_dir experiments/results/cvmi_val
  (test split additionally needs --frozen)
"""
import os
import sys
import json
import argparse
import re
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.multimodal_dataset import MultimodalSequenceDataset, collate_multimodal_batch  # noqa: E402
from train import build_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTOCOL_DOC = os.path.join(REPO, "docs", "cvmi_protocol_lock.md")
EXPECTED_VARIANT = {"full": "traj_full", "iso": "traj_isolated"}
# Config sections that must be identical across all compared checkpoints (everything except the variant/seed)
SHARED_CFG = [("project", "data_dir"), ("data",), ("training",), ("model", "feature_dim"), ("model", "dropout"),
              ("model", "interaction"), ("model", "temporal"), ("model", "trajectory_encoder")]


def git_state():
    def run(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": (run("status", "--porcelain") or "").splitlines()}


def check_protocol_lock(git, doc=PROTOCOL_DOC):
    """The test split may only be evaluated once the protocol note is committed with its lock commit."""
    if not os.path.exists(doc):
        sys.exit(f"Protocol note missing: {doc}")
    text = open(doc).read()
    m = re.search(r"Commit:\s*([0-9a-fA-F]{7,40})", text)
    if not m or "____" in text.split("\n## ")[0]:
        sys.exit(f"{os.path.relpath(doc, REPO)} is not locked (fill in 'Date locked' and 'Commit', then commit it).")
    lock = m.group(1)
    if git["commit"] is None:
        sys.exit("Cannot verify protocol lock: not a git checkout.")
    try:
        subprocess.check_call(["git", "merge-base", "--is-ancestor", lock, "HEAD"], cwd=REPO,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        sys.exit(f"Protocol lock commit {lock} is not an ancestor of HEAD {git['commit'][:10]}.")
    if any(doc.endswith(l.split()[-1]) for l in git["dirty_files"]):
        sys.exit(f"{os.path.relpath(doc, REPO)} has uncommitted changes.")
    return lock


def cfg_get(cfg, path):
    for k in path:
        cfg = (cfg or {}).get(k)
    return cfg


def load_split(data_dir, split):
    ds = MultimodalSequenceDataset(data_dir=data_dir, split=split)
    items = [ds[i] for i in range(len(ds))]
    batch = collate_multimodal_batch(items)
    return batch


def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location=device)
    variant = ck["config"]["project"]["model_variant"]
    model = build_model(variant, ck["config"], device)
    state = {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{path}: missing={missing} unexpected={unexpected}")
    model.eval()
    return model, variant, int(ck.get("epoch", -1)) + 1, ck["config"]["project"].get("seed"), ck["config"]


@torch.no_grad()
def predict(model, batch, device, bs=256, neighbor_override=None):
    n = batch["trajectory"].shape[0]
    out = []
    for s in range(0, n, bs):
        b = {k: (v[s:s + bs].to(device) if torch.is_tensor(v) else v[s:s + bs]) for k, v in batch.items()}
        if neighbor_override is not None:
            na, nm = neighbor_override
            b["neighbor_agents"] = na[s:s + bs].to(device)
            b["neighbor_mask"] = nm[s:s + bs].to(device)
        lg = model(b)["crossing_logits"]
        out.append(F.softmax(lg.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
        "f1": f1_score(y, yhat, zero_division=0), "acc": accuracy_score(y, yhat),
        "bal_acc": balanced_accuracy_score(y, yhat), "brier": brier_score_loss(y, p),
    }


def cross_track_derangement(track_ids, rng):
    """Permutation with perm[i] from a different track than i (Sattolo cycle + repair)."""
    n = len(track_ids)
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):  # Sattolo: single cycle, no fixed points
        j = rng.integers(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    tid = np.asarray(track_ids)
    for _ in range(20):
        bad = np.where(tid[perm] == tid)[0]
        if len(bad) == 0:
            break
        for i in bad:
            j = rng.integers(0, n)
            if tid[perm[j]] != tid[i] and tid[perm[i]] != tid[j]:
                perm[i], perm[j] = perm[j], perm[i]
    same = float(np.mean(tid[perm] == tid))
    return perm, same


def paired_bootstrap(y, pa, pb, tracks, n_boot, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(tracks)
    idx_by = {t: np.where(tracks == t)[0] for t in uniq}
    point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[t] for t in samp])
        if len(np.unique(y[ii])) < 2:
            continue
        deltas.append(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
    deltas = np.array(deltas)
    if len(deltas) < 0.9 * n_boot:
        return {"delta_auc": point, "ci": [float("nan"), float("nan")], "p_two_sided": float("nan"),
                "n_tracks": int(len(uniq)), "warning": "too few valid resamples"}
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    return {"delta_auc": float(point), "ci": [float(lo), float(hi)], "p_two_sided": float(max(p, 1.0 / len(deltas))),
            "n_tracks": int(len(uniq)), "n_windows": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed_pie_tte")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--full_ckpts", nargs="+", required=True)
    ap.add_argument("--iso_ckpts", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mmes", type=float, default=0.015, help="Minimum meaningful ΔAUC")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--protocol_doc", default=PROTOCOL_DOC, help="Locked protocol note required for --split test")
    args = ap.parse_args()
    if args.split == "test" and not args.frozen:
        sys.exit("Refusing to touch the test split without --frozen.")
    git = git_state()
    lock_commit = check_protocol_lock(git, os.path.abspath(args.protocol_doc)) if args.split == "test" else None
    if git["dirty_files"]:
        print(f"WARNING: {len(git['dirty_files'])} uncommitted file(s); results will record a dirty tree.")
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = load_split(args.data_dir, args.split)
    y = batch["cross"].numpy()
    tracks = np.asarray(batch["track_id"])
    k_last = batch["neighbor_mask"][:, -1].sum(-1).numpy()
    rng = np.random.default_rng(args.seed)
    perm, same_frac = cross_track_derangement(tracks, rng)
    shuffled = (batch["neighbor_agents"][perm], batch["neighbor_mask"][perm])
    removed = (torch.zeros_like(batch["neighbor_agents"]), torch.zeros_like(batch["neighbor_mask"]))
    print(f"[{args.split}] windows={len(y)} tracks={len(np.unique(tracks))} crossing={y.mean():.3f} "
          f"K0={np.mean(k_last == 0):.3f} | shuffle same-track fraction={same_frac:.4f}")

    preds, per_seed = {}, {"full": [], "iso": [], "full_removed": [], "full_shuffled": []}
    ref_cfg, ckpt_info = None, []
    for group, paths in [("full", args.full_ckpts), ("iso", args.iso_ckpts)]:
        for path in paths:
            model, variant, epoch, seed, cfg = load_model(path, device)
            if variant != EXPECTED_VARIANT[group]:
                sys.exit(f"{path}: variant '{variant}' passed as --{group}_ckpts (expected '{EXPECTED_VARIANT[group]}').")
            if ref_cfg is None:
                ref_cfg = cfg
            for key in SHARED_CFG:
                if cfg_get(cfg, key) != cfg_get(ref_cfg, key):
                    sys.exit(f"{path}: config '{'.'.join(key)}' differs from the first checkpoint; "
                             "models are not comparable.")
            tag = f"{group}_seed{seed}"
            if seed is None or tag in preds:
                sys.exit(f"{path}: missing or duplicate seed ({seed}) within --{group}_ckpts.")
            ckpt_info.append({"group": group, "path": path, "variant": variant, "seed": seed, "best_epoch": epoch})
            preds[tag] = predict(model, batch, device)
            per_seed[group].append(metrics(y, preds[tag]))
            print(f"  {tag:<22} ({variant}, best epoch {epoch}): AUC={per_seed[group][-1]['auc']:.4f}")
            if group == "full":
                preds[tag + "_removed"] = predict(model, batch, device, neighbor_override=removed)
                preds[tag + "_shuffled"] = predict(model, batch, device, neighbor_override=shuffled)
                per_seed["full_removed"].append(metrics(y, preds[tag + "_removed"]))
                per_seed["full_shuffled"].append(metrics(y, preds[tag + "_shuffled"]))

    def ens(prefix, suffix=""):
        keys = [k for k in preds if k.startswith(prefix + "_seed") and k.endswith(suffix) and
                (suffix or not (k.endswith("_removed") or k.endswith("_shuffled")))]
        return np.mean([preds[k] for k in keys], axis=0)

    p_full, p_iso = ens("full"), ens("iso")
    p_rem, p_shuf = ens("full", "_removed"), ens("full", "_shuffled")

    summary = {"split": args.split, "n_windows": int(len(y)), "n_tracks": int(len(np.unique(tracks))),
               "crossing_share": float(y.mean()), "shuffle_same_track_fraction": same_frac,
               "git_commit": git["commit"], "git_dirty_files": git["dirty_files"], "protocol_lock_commit": lock_commit,
               "checkpoints": ckpt_info, "n_boot": args.n_boot, "bootstrap_seed": args.seed, "mmes": args.mmes,
               "per_seed_mean_std": {}, "ensemble": {}, "comparisons": {}}
    for g, lst in per_seed.items():
        summary["per_seed_mean_std"][g] = {m: [float(np.mean([d[m] for d in lst])), float(np.std([d[m] for d in lst]))]
                                          for m in lst[0]} if lst else {}
    for name, p in [("full", p_full), ("iso", p_iso), ("full_removed", p_rem), ("full_shuffled", p_shuf)]:
        summary["ensemble"][name] = {k: float(v) for k, v in metrics(y, p).items()}

    comps = {"full_vs_isolated": (p_full, p_iso), "full_vs_neighbours_removed": (p_full, p_rem),
             "full_vs_neighbours_shuffled": (p_full, p_shuf)}
    strata = {"all": np.ones_like(y, bool), "K=0": k_last == 0, "K>=1": k_last >= 1}
    for cname, (pa, pb) in comps.items():
        for sname, mask in strata.items():
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            summary["comparisons"][f"{cname} [{sname}]"] = paired_bootstrap(
                y[mask], pa[mask], pb[mask], tracks[mask], args.n_boot, args.seed)

    h2 = summary["comparisons"].get("full_vs_isolated [all]", {})
    lo = h2.get("ci", [np.nan])[0]
    summary["h2_verdict"] = (
        "undetermined (no valid bootstrap CI)" if not np.isfinite(lo) else
        "supported (CI>0 and ΔAUC>=MMES)" if lo > 0 and h2["delta_auc"] >= args.mmes else
        "detectable but below MMES" if lo > 0 else
        "not supported (CI includes 0 or negative)")

    np.savez_compressed(os.path.join(args.output_dir, f"predictions_{args.split}.npz"),
                        y=y, track_ids=tracks, k_last=k_last, perm=perm, **preds)
    with open(os.path.join(args.output_dir, f"results_{args.split}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # Markdown table for the paper
    L = [f"# Results ({args.split}, {len(y)} windows, {summary['n_tracks']} tracks)", "",
         "| Model | AUC | PR-AUC | F1 | Bal. Acc | Brier |", "|---|---|---|---|---|---|"]
    names = {"iso": "Isolated (trajectory only)", "full": "Interaction (trajectory + neighbours)",
             "full_removed": "Interaction, neighbours removed", "full_shuffled": "Interaction, neighbours shuffled"}
    for g in ["iso", "full", "full_removed", "full_shuffled"]:
        ms = summary["per_seed_mean_std"][g]
        L.append(f"| {names[g]} | " + " | ".join(f"{ms[m][0]:.3f} ± {ms[m][1]:.3f}" for m in ["auc", "pr_auc", "f1", "bal_acc", "brier"]) + " |")
    L += ["", "Per-seed mean ± std across seeds.", "", "| Comparison | ΔAUC | 95% CI (track bootstrap) | p | tracks |", "|---|---|---|---|---|"]
    for k, v in summary["comparisons"].items():
        L.append(f"| {k} | {v['delta_auc']:+.4f} | [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] | {v['p_two_sided']:.4f} | {v['n_tracks']} |")
    L += ["", f"H2 verdict: **{summary['h2_verdict']}**"]
    open(os.path.join(args.output_dir, f"results_{args.split}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
