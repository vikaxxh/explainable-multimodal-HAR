#!/bin/bash
# Preprocess scaled Honda TITAN dataset into data/processed_titan
PYTHON_BIN="$HOME/.conda/envs/emit-har/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python)"
fi

echo "Running scaled TITAN preprocessing using: $PYTHON_BIN"
$PYTHON_BIN preprocess_multidataset.py \
    --max_jaad_samples -1 \
    --max_pie_samples -1 \
    --max_titan_samples 0 \
    --max_micro_samples -1 \
    --output_dir data/processed_titan \
    --stride 4 \
    --clean
