#!/bin/bash
#SBATCH --job-name=cross_amber_olmo
#SBATCH --output=./slurm-out/analysis/cross_encode_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=general

set -a
source scripts/env_configs/.env
set +a

source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

export NCCL_P2P_DISABLE=1
module load cuda-12.4

GPU_ID=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
         awk '{print NR-1 ":" $1}' | sort -t: -k2 -nr | head -n1 | cut -d: -f1)
export CUDA_VISIBLE_DEVICES=$GPU_ID

mkdir -p slurm-out/analysis

set -euo pipefail

if [ "${SKIP_CROSS_ENCODE:-0}" != "1" ]; then
    echo "=== step 1: cross-encode AMBER↔OLMo SAEs ==="
    python analysis/cross_encode_amber_olmo.py \
        --k 50 \
        --batch-size 4096
else
    echo "=== step 1: SKIPPED (SKIP_CROSS_ENCODE=1) ==="
fi

echo "=== step 2: build same-jump cross-match report ==="
python analysis/compare_jumps_amber_olmo_cross.py \
    --min-shared 1 \
    --out analysis/compare_jumps_amber_olmo_cross_min1.html \
    --top-k-pairs 30

echo "=== done ==="
