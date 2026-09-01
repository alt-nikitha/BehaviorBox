#!/bin/bash
#SBATCH --job-name=fam_slider_grid
#SBATCH --output=./slurm-out/analysis/fam_slider_grid_%j.out
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16
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
python dump_exclusive_slider_grid.py \
    --exclude-family 3 \
    --out slider_grid.json
