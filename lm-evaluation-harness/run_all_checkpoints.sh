#!/bin/bash
# =============================================================================
# Submit evaluation jobs for each checkpoint revision
# Usage: ./run_all_checkpoints.sh [checkpoints.txt]
# =============================================================================

CHECKPOINT_FILE="${1:-/home/nsrikant/BehaviorBoxNew/checkpoints_info/olmo3_7b_checkpoints.txt}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./eval_results}"

# while IFS= read -r revision || [ -n "$revision" ]; do
#     [[ -z "$revision" || "$revision" =~ ^# ]] && continue
    
#     output_dir="$BASE_OUTPUT_DIR/$revision"
#     mkdir -p "$output_dir"
    
#     echo "Submitting: $revision -> $output_dir"
#     sbatch --export=ALL,MODEL_REVISION="$revision",OUTPUT_DIR="$output_dir" run_all_tasks.sh
# done < "$CHECKPOINT_FILE" 


sbatch --export=ALL,MODEL_REVISION="stage1-step146000",OUTPUT_DIR="./eval_results/stage1-step146000" run_all_tasks.sh