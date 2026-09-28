#!/bin/bash
# ==============================================================================
# Slurm Job Script: Single-GPU Training (X-MIST / EMIT-HAR)
# ==============================================================================
#SBATCH --job-name=emit_train_1gpu
#SBATCH --output=experiments/logs/train_1gpu_%j.out
#SBATCH --error=experiments/logs/train_1gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

echo "=========================================================="
echo "Starting Job: $SLURM_JOB_ID on $(hostname) at $(date)"
echo "CUDA Devices Available: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

# 1. Load Modules (Modify for your cluster's module naming, e.g. module load cuda/11.8)
module purge 2>/dev/null || true
module load cuda/11.8 2>/dev/null || module load cuda 2>/dev/null || true

# 2. Activate Conda Environment
source ~/.bashrc
conda activate emit-har || source activate emit-har

# 3. Parameters (Configurable via command-line arguments)
DATA_DIR=${1:-"data/processed_pie_clean"}
SEED=${2:-42}
CHECKPOINT_DIR="experiments/checkpoints/pie_seed${SEED}"

mkdir -p experiments/logs "$CHECKPOINT_DIR" experiments/results

echo "=========================================================="
echo "Configuration: Data=$DATA_DIR | Seed=$SEED | Checkpoints=$CHECKPOINT_DIR"
echo "=========================================================="

# 4. Launch Training
python train.py \
    --config configs/config.yaml \
    --data_dir "$DATA_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --seed "$SEED" \
    --model proposed \
    --epochs 50 \
    --batch_size 16 \
    --lr 0.0001

echo "Training completed at $(date)"

# 5. Evaluate Best Checkpoint on Test Split
echo "=========================================================="
echo "Running Benchmark Evaluation on Test Split..."
echo "=========================================================="
python evaluate.py \
    --checkpoint "$CHECKPOINT_DIR/checkpoint_best.pt" \
    --data_dir "$DATA_DIR" \
    --split test

# 6. Quantitative XAI Faithfulness Verification
echo "=========================================================="
echo "Running Distance-Matched XAI Faithfulness Verification..."
echo "=========================================================="
python explain.py \
    --checkpoint "$CHECKPOINT_DIR/checkpoint_best.pt" \
    --data_dir "$DATA_DIR" \
    --faithfulness

echo "Job $SLURM_JOB_ID fully completed at $(date)"
