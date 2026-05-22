#!/bin/bash
# =============================================================================
# Submit evaluation jobs for each checkpoint revision
# Limits concurrency to MAX_CONCURRENT jobs at a time.
# Usage: ./run_all_checkpoints.sh [checkpoints.txt]
# =============================================================================

CHECKPOINT_FILE="${1:-/home/nsrikant/BehaviorBoxNew/checkpoints_info/amber_checkpoints.txt}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./eval_results_amber}"
job_ids=()

while IFS= read -r revision || [ -n "$revision" ]; do
    [[ -z "$revision" || "$revision" =~ ^# ]] && continue

    output_dir="$BASE_OUTPUT_DIR/$revision"
    mkdir -p "$output_dir"

    echo "Submitting: $revision -> $output_dir"
    # Keep trying until sbatch succeeds; wait if QOS limit is hit
    while true; do
        jobid=$(sbatch --parsable --export=ALL,MODEL_REVISION="$revision",OUTPUT_DIR="$output_dir" run_all_tasks.sh 2>&1)
        if [[ "$jobid" =~ ^[0-9]+$ ]]; then
            echo "  Submitted job $jobid"
            job_ids+=("$jobid")
            break
        else
            echo "  QOS limit hit, waiting for a slot..."
            sleep 60
        fi
    done
done < "$CHECKPOINT_FILE"

echo "All jobs submitted. Job IDs: ${job_ids[*]}"

# sbatch --export=ALL,MODEL_REVISION="stage1-step103000",OUTPUT_DIR="eval_results_olmo3/stage1-step103000" run_all_tasks.sh 2>&1

