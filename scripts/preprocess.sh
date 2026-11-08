#!/bin/bash
# Preprocess: build FAISS index from pre-computed embeddings.
# Usage: bash scripts/preprocess.sh

set -e

echo "=== EviRank Preprocessing ==="
echo "Building FAISS index from pre-computed embeddings..."

python -m src.main \
    --mode preprocess \
    --config configs/default.yaml

echo "Done."
