#!/bin/bash
# ==============================================================================
# Slurm Job Script: Dataset Preprocessing (X-MIST / EMIT-HAR)
# Preprocesses PIE and JAAD into official leakage-free benchmark partitions.
# ==============================================================================
#SBATCH --job-name=emit_prep
#SBATCH --output=experiments/logs/prep_%j.out
#SBATCH --error=experiments/logs/prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

echo "=========================================================="
echo "Starting Preprocessing: Job $SLURM_JOB_ID on $(hostname)"
echo "Time: $(date)"
echo "=========================================================="

# 1. Activate Environment
source ~/.bashrc
conda activate emit-har || source activate emit-har

mkdir -p experiments/logs data/processed_pie_clean data/processed_jaad_clean

# 2. Preprocess PIE (Official Benchmark: Train=01,02,04 | Val=05,06 | Test=03)
echo "=========================================================="
echo "Step 1: Extracting official PIE sequences..."
echo "=========================================================="
python preprocess_multidataset.py \
    --max_jaad_samples -1 \
    --max_pie_samples 0 \
    --output_dir data/processed_pie_clean \
    --stride 4 \
    --clean

# 3. Preprocess JAAD (Official Video-Disjoint Split: jaad_beh)
echo "=========================================================="
echo "Step 2: Extracting official JAAD sequences..."
echo "=========================================================="
python preprocess_multidataset.py \
    --max_pie_samples -1 \
    --max_jaad_samples 0 \
    --output_dir data/processed_jaad_clean \
    --stride 4 \
    --clean

echo "=========================================================="
echo "All preprocessing completed successfully at $(date)"
echo "=========================================================="
