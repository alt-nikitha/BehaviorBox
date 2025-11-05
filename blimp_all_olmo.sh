#!/bin/bash


# Configuration
DATA_PATH="/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl"
FEATURES_DIR="/home/nsrikant/bbox_outputs/output/blimp_full"
SAVE_DIR="results"
METRIC="seq_logprob"

# List of models to evaluate
MODEL_NAMES=(
    "OLMo2-step0-tokens0B"
    "OLMo2-step1000-tokens3B"
    "OLMo2-step2000-tokens5B"
    "OLMo2-step3000-tokens7B"
    "OLMo2-step4000-tokens9B"
    "OLMo2-step5000-tokens11B"
    "OLMo2-step6000-tokens13B"
    "OLMo2-step7000-tokens15B"
    "OLMo2-step8000-tokens17B"
    "OLMo2-step9000-tokens19B"
    "OLMo2-step10000-tokens21B"
)


# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================"
echo "Starting BLiMP Evaluation for ${#MODEL_NAMES[@]} models"
echo "================================================"
echo ""

# Counter for progress
TOTAL=${#MODEL_NAMES[@]}
CURRENT=0

# Loop through each model
for MODEL in "${MODEL_NAMES[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo -e "${BLUE}[$CURRENT/$TOTAL] Processing model: $MODEL${NC}"
    echo "----------------------------------------"
    
    # Construct paths
    

    # Run evaluation
    python blimp_evaluate.py \
        --data_path "$DATA_PATH" \
        --features_dir "$FEATURES_DIR" \
        --model_name "$MODEL" \
        --save_dir "$SAVE_DIR" \
        --metric "$METRIC"
    
    # Check if command succeeded
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully completed $MODEL${NC}"
    else
        echo -e "${RED}✗ Failed to process $MODEL${NC}"
    fi
    
    echo ""
done

echo "================================================"
echo -e "${GREEN}All evaluations complete!${NC}"
echo "================================================"
echo ""
echo "Results saved in: $SAVE_DIR"
echo ""
