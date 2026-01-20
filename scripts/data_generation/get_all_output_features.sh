#!/bin/bash

DATADIR="/data/user_data/nsrikant/bbox_data/data/larger_pile.jsonl"
OUTPUTDIR="/data/user_data/nsrikant/bbox_data/output/output/larger_pile"
MODEL_ID="EleutherAI/pythia-6.9b"
prefix="pythia-6_9b"
MAX_JOBS=2  # Process sequentially to avoid GPU resource deadlock
running_jobs=0

# REVISIONS=(
# "step1"
# "step2"
# "step4"
# "step8"
# "step16"
# "step32"
# "step64"
# "step128"
# "step256"
# "step512"
# "step1000"
# "step10000"
# "step70000"
# "step100000"
# ""
# )

REVISIONS=(
"step256"
"step512"
)


# Create a tracking file for this run
RUN_ID=$(date +%s)
TRACKING_FILE="scripts/data_generation/tmp/run_${RUN_ID}_tracking.txt"
mkdir -p scripts/data_generation/tmp
touch "$TRACKING_FILE"

echo "Run ID: $RUN_ID"
echo "Tracking file: $TRACKING_FILE"
echo ""

# Cleanup function
cleanup() {
  echo ""
  echo "=========================================="
  echo "Cleaning up all vLLM servers from this run..."
  echo "=========================================="
  
  if [ -f "$TRACKING_FILE" ]; then
    while read -r model_name vllm_jobid; do
      if [ -n "$vllm_jobid" ] && [ "$vllm_jobid" != "none" ]; then
        echo "Canceling vLLM job $vllm_jobid for model $model_name"
        scancel $vllm_jobid 2>/dev/null || true
        
        # Clean up metadata
        rm -f "scripts/data_generation/tmp/${model_name}/host_port.txt"
        rm -f "scripts/data_generation/tmp/${model_name}/ready.flag"
        rm -f "scripts/data_generation/tmp/${model_name}/vllm_jobid.txt"
      fi
    done < "$TRACKING_FILE"
    
    rm -f "$TRACKING_FILE"
  fi
  
  echo "✓ Cleanup complete"
  echo ""
}

# Set up trap to cleanup on exit or interrupt
trap cleanup EXIT INT TERM

for STEP in "${REVISIONS[@]}"; do
    if [ -z "$STEP" ]; then
        NAME="${prefix}"
    else
        NAME="${prefix}-${STEP}"
    fi
   
    echo "=== Starting $NAME ==="

    # Launch get_output_features.sh and capture its output
    (
      # Run the script
      bash scripts/data_generation/get_output_features.sh \
          --model_name="${NAME}" \
          --model_id="${MODEL_ID}" \
          --data="${DATADIR}" \
          --output_dir="${OUTPUTDIR}" \
          ${STEP:+--revision="$STEP"} \
          --batch_size=1000 \
          --async_limiter=1000 \
          --slurm
      
      # After completion, record that this vLLM server should be cleaned up
      vllm_jobid_file="scripts/data_generation/tmp/${NAME}/vllm_jobid.txt"
      if [ -f "$vllm_jobid_file" ]; then
        vllm_jobid=$(cat "$vllm_jobid_file")
        echo "${NAME} ${vllm_jobid}" >> "$TRACKING_FILE"
      fi
      
      echo "=== Completed $NAME ==="
    ) &

    ((running_jobs++))

    # Wait when we hit the max
    if (( running_jobs >= MAX_JOBS )); then
        wait -n
        ((running_jobs--))
    fi
done

# Wait for all remaining jobs
echo ""
echo "=== Waiting for all remaining jobs to complete ==="
wait

echo ""
echo "=== All extraction jobs completed ==="
echo ""

# The trap will handle cleanup automatically