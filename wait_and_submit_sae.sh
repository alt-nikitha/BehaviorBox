#!/bin/bash
# Wait for job 6516520 to complete, then submit output features job with QOS retry logic

JOB_TO_WAIT=6519802
SUBMIT_SCRIPT="scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n_moreearly_amber.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=200.json"

echo "Waiting for job $JOB_TO_WAIT to complete..."

while squeue -j "$JOB_TO_WAIT" --noheader 2>/dev/null | grep -q "$JOB_TO_WAIT"; do
    sleep 60
done

echo "Job $JOB_TO_WAIT completed."
sacct -j "$JOB_TO_WAIT" --format=JobID,State,ExitCode --noheader

# Submit with retry on QOS errors (no limit)
RETRY_DELAY=120

while true; do
    echo "Submitting $SUBMIT_SCRIPT ..."
    OUTPUT=$(sbatch "$SUBMIT_SCRIPT" 2>&1)

    if echo "$OUTPUT" | grep -qi "Submitted batch job"; then
        echo "Success: $OUTPUT"
        exit 0
    else
        echo "Error: $OUTPUT"
        echo "Retrying in ${RETRY_DELAY}s..."
        sleep "$RETRY_DELAY"
    fi
done
