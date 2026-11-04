# CVMI 2027 JAAD Replication Protocol (fill in date + commit hash, commit BEFORE any JAAD test run)

Date locked: ____  Commit: ____

## Status
Written 2026-11-04, after the PIE protocol was locked (`docs/cvmi_protocol_lock.md`, commit 36bd0c5) and
before any JAAD data was preprocessed. The JAAD replication is secondary to the PIE H2 test.

## Data
- JAAD_beh (pedestrians with behavioural tags), official video-disjoint splits in
  `data/splits/jaad_beh/{train,val,test}.txt`.
- Same protocol as PIE: observation 16 frames (30 fps), ending 30-60 frames (1-2 s) before the crossing
  event (first 'crossing' frame) or before the last annotated frame for non-crossing pedestrians;
  overlap 0.6; 'crossing-irrelevant' pedestrians excluded.
- Inputs: box-centre kinematics [x, y, vx, vy, heading], normalised by each clip's own resolution.
  Neighbours: up to 4 other pedestrians ('pedestrian', 'ped', 'people' tracks) annotated in all 16
  frames, mean distance <= 0.20. JAAD has no vehicle boxes, so neighbours are pedestrians only.
- Built by `preprocess_jaad_tte.py`; `tools/check_tte_data.py` must pass before training.

## Models
- Identical to PIE: `traj_full` vs `traj_isolated`, config `configs/cvmi_pie_tte.yaml` unchanged,
  seeds 42, 100, 2024, checkpoint = max validation ROC-AUC, early stopping patience 10, CPU training.
- No hyper-parameter changes for JAAD, and no tuning on the JAAD test split.

## Hypothesis (same rule as PIE H2)
ΔAUC = AUC(traj_full seed-ensemble) - AUC(traj_isolated seed-ensemble) on the JAAD_beh test split.
Replicates iff the 95% track-clustered bootstrap CI (2000 resamples) excludes 0 AND ΔAUC >= 0.015.
Secondary: K-stratified ΔAUC, neighbours-removed and neighbours-shuffled controls.

## Execution
`tools/eval_paper.py --split test --frozen --protocol_doc docs/cvmi_jaad_protocol_lock.md`
run exactly once via `DATA=data/processed_jaad_tte TAG=cvmi_jaad DOC=docs/cvmi_jaad_protocol_lock.md`
with `scripts_hpc/pbs_cvmi_eval_test.pbs`.
