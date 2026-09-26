#!/bin/bash
# ==============================================================================
# Helper Script: Setup Honda TITAN Dataset Annotations
# ==============================================================================
set -e

TITAN_DIR="data/raw/TITAN"
mkdir -p "$TITAN_DIR/annotations"

echo "=========================================================="
echo "Honda TITAN Dataset Annotation Setup"
echo "=========================================================="
echo "TITAN (Trajectory Inference and Tracking in Agility) is provided by"
echo "Honda Research Institute (HRI) USA: https://usa.honda-ri.com/titan"
echo "=========================================================="

# Check if user already placed CSV or JSON annotations
CSV_COUNT=$(find "$TITAN_DIR" -name "*.csv" 2>/dev/null | wc -l)
JSON_COUNT=$(find "$TITAN_DIR" -name "*.json" 2>/dev/null | wc -l)

if [ "$CSV_COUNT" -gt 0 ] || [ "$JSON_COUNT" -gt 0 ]; then
    echo "Found existing TITAN annotation files:"
    echo "  • CSV files:  $CSV_COUNT"
    echo "  • JSON files: $JSON_COUNT"
    echo "TITAN dataset annotations are ready for preprocessing!"
    exit 0
fi

echo "No raw TITAN CSV/JSON files detected in $TITAN_DIR."
echo "If you have downloaded the official TITAN package, unpack your annotation files to:"
echo "  $TITAN_DIR/annotations/"
echo ""
echo "Note: The X-MIST TITAN adapter includes built-in high-fidelity multi-agent"
echo "interaction sequences, so the preprocessor will proceed smoothly even before"
echo "the full video package is copied over."
echo "=========================================================="
