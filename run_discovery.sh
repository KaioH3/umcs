#!/bin/bash
# Run batch discovery - keep running until Ctrl+C
# Usage: ./run_discovery.sh

set -e

source .env
export GROQ_API_KEY

mkdir -p data

echo "============================================"
echo "UMCS Batch Discovery"
echo "============================================"
echo "Words: $(wc -l < data/common_words_ext.txt)"
echo "Languages: PT,EN,ES,IT,DE,FR"
echo "Output: data/inferred_words.csv"
echo "============================================"
echo "Starting... (Ctrl+C to stop)"
echo ""

./batch_discover data/common_words_ext.txt "PT,EN,ES,IT,DE,FR"

echo ""
echo "============================================"
echo "Done! Check data/inferred_words.csv"
echo "Words discovered: $(wc -l < data/inferred_words.csv)"
echo "============================================"
