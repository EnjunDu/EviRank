#!/bin/bash
# Train student model via knowledge distillation.
# Usage: bash scripts/train_student.sh

set -e

echo "=== EviRank Student Distillation ==="
echo "Training student re-ranker..."
echo ""

python -m src.main \
    --mode train \
    --config configs/default.yaml
