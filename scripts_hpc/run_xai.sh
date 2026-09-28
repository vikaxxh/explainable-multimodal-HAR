#!/bin/bash
# ==============================================================================
# Helper Script: Generate XAI Explanations, Faithfulness, and Figures
# ==============================================================================
PYTHON_BIN="$HOME/.conda/envs/emit-har/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python)"
fi

echo "Running XAI explanation generation using: $PYTHON_BIN"
$PYTHON_BIN explain.py \
    --config configs/config.yaml \
    --checkpoint experiments/checkpoints/titan/checkpoint_best.pt \
    --data_dir data/processed_titan \
    --sample_idx 0 \
    --faithfulness \
    --visualize
