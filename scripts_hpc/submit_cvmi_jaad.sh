#!/bin/bash
# JAAD replication, CPU only: preprocess -> 2 models x 3 seeds -> validation eval, chained with
# PBS dependencies (do not qmove these jobs: that can drop the dependency).
# Run from the repo root after scripts_hpc/setup_jaad_annotations.sh. Test eval stays manual.
set -euo pipefail
cd "$HOME/explainable-multimodal-HAR"
Q=${Q:-short}
RES="select=1:ncpus=10:mem=32gb"
PREP=$(qsub -q $Q -l $RES -o "$PWD/cvmi_jaad_preprocess.log" scripts_hpc/pbs_cvmi_jaad_preprocess.pbs)
echo "prep: $PREP"
DEPS=""
for MODEL in traj_full traj_isolated; do
  for SEED in 42 100 2024; do
    J=$(qsub -W depend=afterok:$PREP -q $Q -l $RES -l walltime=06:00:00 -N "jaad_${MODEL}_${SEED}" \
        -o "$PWD/cvmi_jaad_${MODEL}_${SEED}.pbslog" \
        -v MODEL=$MODEL,SEED=$SEED,DATA=data/processed_jaad_tte,TAG=cvmi_jaad,OMP_NUM_THREADS=10 \
        scripts_hpc/pbs_cvmi_train.pbs)
    echo "train $MODEL/$SEED: $J"; DEPS="$DEPS:$J"
  done
done
VAL=$(qsub -W depend=afterok$DEPS -q $Q -l $RES -l walltime=02:00:00 -o "$PWD/cvmi_jaad_eval_val.log" \
      -v DATA=data/processed_jaad_tte,TAG=cvmi_jaad,OMP_NUM_THREADS=10 scripts_hpc/pbs_cvmi_eval_val.pbs)
echo "val eval: $VAL"
