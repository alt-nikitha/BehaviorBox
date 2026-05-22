#!/bin/bash
#SBATCH --job-name=output_features
#SBATCH --array=0-1%2
#SBATCH --output=./slurm-out/output_features_parent_%A_%a.out
#SBATCH --error=./slurm-out/output_features_parent_%A_%a.err
#SBATCH --partition=cpu


# DATADIR="/data/user_data/nsrikant/bbox_data/data/amber_300_unseen.jsonl"
# OUTPUTDIR="/data/user_data/nsrikant/bbox_data/output/amber_300_unseen"
# MODEL_ID="LLM360/Amber"
# prefix="amber"


DATADIR="/data/user_data/nsrikant/bbox_data/data/olmo_256000_unseen.jsonl"
OUTPUTDIR="/data/user_data/nsrikant/bbox_data/output/olmo_256000_unseen"
MODEL_ID="allenai/Olmo-3-1025-7B"
prefix="olmo3"


REVISIONS=(
stage1-step8000
stage1-step32000
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
    --async_limiter=1000 \
    --slurm

echo "=== Completed $NAME ==="
