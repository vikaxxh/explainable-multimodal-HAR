# CVMI 2027 Evaluation Protocol (fill in date + commit hash, commit BEFORE any Set 03 run)

Date locked: 2026-11-04  Commit: 36bd0c5

## Disclosure
Earlier pipeline versions (before this protocol) ran trained models on PIE Set 03
(`pbs_eval_pie.pbs`, `run_check2_shuffled.py`, `pbs_eval_h3.pbs`). Those models used a different
labelling (crossing anywhere inside a 32-frame window), a wrong time step (1/10 s instead of 1/30 s)
and placeholder RGB/pose/scene inputs. All of those results are discarded. The benchmark below is a new
configuration; Set 03 under this configuration has not been evaluated.

Validation results (Set 05/06) were inspected before locking: ΔAUC full vs isolated = +0.0072,
95% CI [-0.0063, +0.0222] (227 tracks). Nothing was changed after seeing them.

## Data
- PIE, official splits: train set01/02/04, val set05/06, test set03.
- Observation 16 frames (30 fps); observation ends 30-60 frames (1-2 s) before the crossing event
  (first 'crossing' frame) or before the last frame for non-crossing pedestrians; overlap 0.6.
- Pedestrians annotated only 'crossing-irrelevant' excluded.
- Inputs: pedestrian box-centre kinematics [x, y, vx, vy, heading] (normalised image coords) and up to
  4 neighbours (pedestrians/vehicles annotated in all 16 frames, mean distance <= 0.20).

## Models (identical except the neighbour branch)
- `traj_isolated`: trajectory encoder -> temporal transformer -> crossing head.
- `traj_full`: same + neighbour interaction transformer (learned token when no neighbour).
- Config `configs/cvmi_pie_tte.yaml`; seeds 42, 100, 2024; checkpoint = max validation ROC-AUC,
  early stopping patience 10. No hyper-parameter tuning on Set 03.
- All six runs trained on CPU (10 cores, PBS `long`/`short` queues) because no GPU was free.

## Primary hypothesis (H2)
ΔAUC = AUC(traj_full seed-ensemble) - AUC(traj_isolated seed-ensemble) on Set 03.
Supported iff the 95% track-clustered bootstrap CI (2000 resamples over pedestrian tracks) excludes 0
AND ΔAUC >= 0.015. Otherwise reported as a null / sub-threshold result.

## Secondary (descriptive)
- ΔAUC stratified by K (neighbours at last observed frame): K=0, K>=1.
- Controls on traj_full: neighbours removed; neighbours shuffled across other tracks.
- Per-seed mean ± std of AUC, PR-AUC, F1, balanced accuracy, Brier.

## Execution
`tools/eval_paper.py --split test --frozen` run exactly once (`scripts_hpc/pbs_cvmi_eval_test.pbs`).
