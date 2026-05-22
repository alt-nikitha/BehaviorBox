#!/bin/bash
#SBATCH --job-name=group_feat_cross
#SBATCH --output=./slurm-out/analysis/group_feat_cross_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=general

set -a
source scripts/env_configs/.env
set +a

source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

mkdir -p slurm-out/analysis

set -euo pipefail

THRESHOLD=${THRESHOLD:-0.7}
MIN_SHARED=${MIN_SHARED:-1}
OUT_FILE=${OUT_FILE:-analysis/group_features_by_task_cross_min${THRESHOLD}_shared${MIN_SHARED}.html}

python analysis/group_features_by_task_cross.py \
    --threshold $THRESHOLD \
    --min-shared $MIN_SHARED \
    --top-k 30 \
    --out $OUT_FILE

echo "wrote $OUT_FILE"
