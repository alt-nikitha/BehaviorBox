#!/bin/bash
#SBATCH --job-name=weighted_cluster_logreg
#SBATCH --output=./slurm-out/analysis/weighted_cluster_logreg_%j.out
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=32
#SBATCH --time=1:00:00
#SBATCH --partition=cpu

set -a
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

echo "Running on node: $HOSTNAME"

cd analysis
python weighted_cluster_logreg.py \
    --min-r 0.85 --max-l1 0.3 --per-family 548236 \
    --exclude-family 3 --target-family 0 --k 100
