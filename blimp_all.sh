#!/bin/bash


# Configuration
DATA_PATH="/data/user_data/nsrikant/bbox_data/data/blimp_full.jsonl"
FEATURES_DIR="/data/user_data/nsrikant/bbox_data/output/output/blimp_full"
SAVE_DIR="results"
METRIC="seq_logprob"

# List of models to evaluate
# MODEL_NAMES=(
#     "pythia-160m-step1"
#     "pythia-160m-step2"
#     "pythia-160m-step4"
#     "pythia-160m-step8"
#     "pythia-160m-step16"
#     "pythia-160m-step32"
#     "pythia-160m-step64"
#     "pythia-160m-step128"
#     "pythia-160m-step256"
#     "pythia-160m-step512"
#     "pythia-160m-step1000"
#     "pythia-160m-step10000"
#     "pythia-160m-step70000"
#     "pythia-160m-step100000"
#     "pythia-160m"
# )


MODEL_NAMES=(
    "pythia-6_9b-step1"
    "pythia-6_9b-step2"
    "pythia-6_9b-step4"
    "pythia-6_9b-step8"
    "pythia-6_9b-step16"
    "pythia-6_9b-step32"
    "pythia-6_9b-step64"
    "pythia-6_9b-step128"
    "pythia-6_9b-step256"
    "pythia-6_9b-step512"
    "pythia-6_9b-step1000"
    "pythia-6_9b-step10000"
    "pythia-6_9b-step70000"
    "pythia-6_9b-step100000"
    "pythia-6_9b"
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
