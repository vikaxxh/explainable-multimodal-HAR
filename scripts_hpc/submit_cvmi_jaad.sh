#!/bin/bash
# JAAD replication, CPU only. PBS job dependencies are unreliable on this cluster, so stages are
# submitted by hand:
#   1. qsub scripts_hpc/pbs_cvmi_jaad_preprocess.pbs      (wait for "No warnings." in cvmi_jaad_preprocess.log)
#   2. bash scripts_hpc/submit_cvmi_jaad.sh               (this script: 2 models x 3 seeds)
#   3. when all 6 train.log files end in "Finished":
#      qsub -q short -l select=1:ncpus=10:mem=32gb -l walltime=02:00:00 \
#           -v DATA=data/processed_jaad_tte,TAG=cvmi_jaad,OMP_NUM_THREADS=10 scripts_hpc/pbs_cvmi_eval_val.pbs
# Test eval stays manual and needs docs/cvmi_jaad_protocol_lock.md locked first.
set -euo pipefail
cd "$HOME/explainable-multimodal-HAR"
[ -f data/processed_jaad_tte/sanity_report.json ] || { echo "Run JAAD preprocessing first"; exit 1; }
Q=${Q:-short}
for MODEL in traj_full traj_isolated; do
  for SEED in 42 100 2024; do
    J=$(qsub -q $Q -l select=1:ncpus=10:mem=32gb -l walltime=06:00:00 -N "jaad_${MODEL}_${SEED}" \
        -v MODEL=$MODEL,SEED=$SEED,DATA=data/processed_jaad_tte,TAG=cvmi_jaad,OMP_NUM_THREADS=10 \
        scripts_hpc/pbs_cvmi_train.pbs)
    echo "train $MODEL/$SEED: $J"
  done
done
