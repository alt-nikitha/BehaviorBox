#!/bin/bash
#SBATCH --job-name=fam0_two_col
#SBATCH --output=./slurm-out/analysis/fam0_two_col_%j.out
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
python fam0_vs_other_two_column.py \
    --model fam0_logreg_weights.pkl \
    --top-n 100 --out fam0_vs_other_two_column.html
