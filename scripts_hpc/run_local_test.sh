#!/bin/bash
# ==============================================================================
# Fast Local Sanity & Pipeline Verification Script
# Runs 1 epoch of training, evaluation, XAI report, and ablation matrix
# ==============================================================================
set -e

echo "=== [1/4] Running Quick Training Smoke Test on Synthetic Data ==="
python train.py --model proposed --synthetic --epochs 1 --batch_size 4

echo "=== [2/4] Running Evaluation Test ==="
python evaluate.py --synthetic

echo "=== [3/4] Running XAI Explanation & Faithfulness Validation ==="
python explain.py --sample_idx 0 --faithfulness --visualize

echo "=== [4/4] Running Full Ablation Matrix ==="
python evaluate.py --ablation

echo "=========================================================="
echo "ALL TESTS PASSED! CODEBASE IS VERIFIED AND HPC-READY."
echo "=========================================================="
