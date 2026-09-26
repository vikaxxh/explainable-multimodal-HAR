#!/bin/bash
# ==============================================================================
# Helper Script: Download and unpack official PIE annotations (25 MB)
# ==============================================================================
set -e

PIE_DIR="data/raw/PIE"
mkdir -p "$PIE_DIR/annotations"

echo "Downloading official PIE pedestrian annotations (~25 MB)..."
curl -L "https://github.com/aras62/PIE/raw/master/annotations/annotations.zip" -o "$PIE_DIR/annotations.zip"

echo "Downloading official PIE vehicle/bicycle annotations (~15 MB)..."
curl -L "https://github.com/aras62/PIE/raw/master/annotations/annotations_vehicle.zip" -o "$PIE_DIR/annotations_vehicle.zip"

echo "Unpacking annotations into $PIE_DIR/annotations/..."
unzip -q -o "$PIE_DIR/annotations.zip" -d "$PIE_DIR/annotations/"
unzip -q -o "$PIE_DIR/annotations_vehicle.zip" -d "$PIE_DIR/annotations/" || true

rm -f "$PIE_DIR/annotations.zip" "$PIE_DIR/annotations_vehicle.zip"

echo "PIE annotations setup complete!"
echo "XML files found: $(find "$PIE_DIR/annotations" -name "*.xml" | wc -l)"
