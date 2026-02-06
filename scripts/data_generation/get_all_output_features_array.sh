#!/bin/bash
#SBATCH --job-name=output_features
#SBATCH --array=0-10%2
#SBATCH --output=./slurm-out/output_features_parent_%A_%a.out
#SBATCH --error=./slurm-out/output_features_parent_%A_%a.err
#SBATCH --partition=cpu


DATADIR="/data/user_data/nsrikant/bbox_data/data/olmo_validation_texts.jsonl"
OUTPUTDIR="/data/user_data/nsrikant/bbox_data/output/olmo_validation_texts"
MODEL_ID="allenai/Olmo-3-1025-7B"
prefix="olmo-3-7b"

REVISIONS=(
    stage1-step1000
    stage1-step15000
    stage1-step73000
    stage1-step146000
    stage1-step365000
    stage1-step731000
    stage1-step1096000
    stage1-step1169000
    stage1-step1315000
    stage2-step18000
    main
)

# Get the revision for this array task
STEP="${REVISIONS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$STEP" ]; then
    NAME="${prefix}"
else
    NAME="${prefix}-${STEP}"
fi

echo "=== Starting $NAME (Array Task $SLURM_ARRAY_TASK_ID) ==="

# Run the script
bash scripts/data_generation/get_output_features.sh \
    --model_name="${NAME}" \
    --model_id="${MODEL_ID}" \
    --data="${DATADIR}" \
    --output_dir="${OUTPUTDIR}" \
    ${STEP:+--revision="$STEP"} \
    --batch_size=1000 \
    --async_limiter=500 \
    --slurm

echo "=== Completed $NAME ==="
