#!/bin/bash
# ==============================================================================
# Slurm Job Script: Automated Ablation Study & Comparison Matrix (Phases 24 & 25)
# ==============================================================================
#SBATCH --job-name=emit_ablation
#SBATCH --output=experiments/logs/ablation_%j.out
#SBATCH --error=experiments/logs/ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

source ~/.bashrc
conda activate emit-har || source activate emit-har

echo "Running Full Ablation Matrix (A1 to A9)..."
python evaluate.py --config configs/config.yaml --ablation
echo "Ablation study completed."
