#!/bin/bash
#SBATCH --job-name=logreg_dim_cluster
#SBATCH --output=./slurm-out/analysis/logreg_dim_cluster_%j.out
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
python logreg_dim_cluster_per_family.py \
    --model fam0_logreg_weights.pkl \
    --ref-class 0 --k 10 --k-min 2 --k-max 10 --n-per-family 2000 --max-show 20 \
    --out logreg_dim_cluster_per_family.html
