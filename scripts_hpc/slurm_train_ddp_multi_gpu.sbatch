#!/bin/bash
# ==============================================================================
# Slurm Job Script: Multi-GPU Distributed Data Parallel (DDP) via torchrun
# ==============================================================================
#SBATCH --job-name=emit_train_ddp
#SBATCH --output=experiments/logs/train_ddp_%j.out
#SBATCH --error=experiments/logs/train_ddp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

echo "=========================================================="
echo "Starting Distributed DDP Training: $SLURM_JOB_ID on $(hostname)"
echo "Number of GPUs allocated: 4"
echo "=========================================================="

# 1. Load Modules
module purge 2>/dev/null || true
module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true

# 2. Activate Environment
source ~/.bashrc
conda activate emit-har || source activate emit-har

# 3. Environment variables for PyTorch Distributed and NCCL
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

mkdir -p experiments/logs experiments/checkpoints experiments/results

# 4. Launch Distributed Multi-GPU Training via torchrun
# 4 processes per node, automatically binds local GPUs 0, 1, 2, 3
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    train.py \
    --config configs/config.yaml \
    --model proposed \
    --distributed \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0002

echo "Distributed training completed at $(date)"
