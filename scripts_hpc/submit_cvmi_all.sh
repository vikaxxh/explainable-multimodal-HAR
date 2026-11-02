#!/bin/bash
# Submits 2 models x 3 seeds = 6 independent training jobs (run AFTER cvmi_prep has finished).
cd "$HOME/explainable-multimodal-HAR" || exit 1
for MODEL in traj_full traj_isolated; do
  for SEED in 42 100 2024; do
    qsub -N "cvmi_${MODEL}_${SEED}" -o "cvmi_${MODEL}_${SEED}.pbslog" -v MODEL=$MODEL,SEED=$SEED scripts_hpc/pbs_cvmi_train.pbs
  done
done
