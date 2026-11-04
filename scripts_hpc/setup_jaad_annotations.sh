#!/bin/bash
# Fetches the official JAAD annotations (XML only, no videos needed) into data/raw/JAAD/annotations.
# Run once on the login node: bash scripts_hpc/setup_jaad_annotations.sh
set -euo pipefail
cd "$HOME/explainable-multimodal-HAR"
REPO=data/raw/JAAD_repo
[ -d "$REPO" ] || git clone --depth 1 https://github.com/ykotseruba/JAAD.git "$REPO"
mkdir -p data/raw/JAAD
if [ -f "$REPO/annotations.zip" ] && [ ! -d "$REPO/annotations" ]; then
  unzip -q "$REPO/annotations.zip" -d "$REPO"
fi
[ -d "$REPO/annotations" ] || { echo "No annotations/ folder in $REPO"; ls "$REPO"; exit 1; }
rm -rf data/raw/JAAD/annotations
ln -s "$PWD/$REPO/annotations" data/raw/JAAD/annotations
echo "JAAD XML files: $(ls data/raw/JAAD/annotations/*.xml | wc -l)"
