#!/bin/bash
#SBATCH --job-name=extreme_bg_cluster
#SBATCH --output=./slurm-out/analysis/extreme_bg_cluster_%j.out
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --partition=cpu

set -a
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

echo "Running on node: $HOSTNAME"

cd analysis
python extreme_blue_green_cluster_compare.py \
    --model fam0_logreg_weights.pkl \
    --n-pool 8000 --n-extreme 1500 --k-min 2 --k-max 8 --max-show 20 \
    --out extreme_blue_green_cluster_compare.html
