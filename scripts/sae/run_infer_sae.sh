#!/bin/bash
#SBATCH --job-name=sae-infer
#SBATCH --output=./slurm-out/sae_infer_%j.out
#SBATCH --error=./slurm-out/sae_infer_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64GB
#SBATCH --time=2-00:00:00
#SBATCH --partition=general

# Usage:
#   sbatch scripts/sae/run_infer_sae.sh /path/to/documents.jsonl /path/to/output.parquet
#
# The script will:
#   1. Launch 7 vLLM servers (one per Amber checkpoint) via SLURM
#   2. Wait for all servers to come up
#   3. Run Longformer + vLLM + SAE inference in-memory
#   4. Clean up all vLLM servers on exit

DATA=${1:?Usage: sbatch run_infer_sae.sh DATA_PATH OUTPUT_PATH}
OUTPUT=${2:?Usage: sbatch run_infer_sae.sh DATA_PATH OUTPUT_PATH}

SAE_DIR="/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_amber_300/n_moreearly_amber_seed=42_ofw=0.7_N=3000_k=50_lp=None"
TRAINING_CACHE_DIR="/home/nsrikant/.cache/n_moreearly_amber/amber_300_unseen/ofw=0.7"
NORM_CACHE="/home/nsrikant/.cache/n_moreearly_amber/norm_constants.json"

set -a
source scripts/env_configs/.env
set +a

source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

mkdir -p slurm-out

cd /home/nsrikant/BehaviorBoxNew

python sae/infer_sae.py \
    --data "${DATA}" \
    --sae_dir "${SAE_DIR}" \
    --training_cache_dir "${TRAINING_CACHE_DIR}" \
    --output "${OUTPUT}" \
    --norm_cache "${NORM_CACHE}" \
    --batch_size 10 \
    --async_limiter 100
