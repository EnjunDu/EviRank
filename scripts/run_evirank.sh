#!/bin/bash
# Run EviRank pipeline: retrieve + evidence mining + re-ranking.
# Usage: bash scripts/run_evirank.sh "your query text here"

set -e

QUERY="${1:-A woman in a blue dress playing guitar at an outdoor cafe}"

echo "=== EviRank Pipeline ==="
echo "Query: ${QUERY}"
echo ""

python -m src.main \
    --mode retrieve \
    --config configs/default.yaml \
    --query "${QUERY}"
