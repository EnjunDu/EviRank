#!/bin/bash
# Evaluate EviRank on a benchmark dataset.
# Usage: bash scripts/evaluate.sh [split]

set -e

SPLIT="${1:-test}"

echo "=== EviRank Evaluation ==="
echo "Split: ${SPLIT}"
echo ""

python -m src.main \
    --mode evaluate \
    --config configs/default.yaml \
    --split "${SPLIT}"
