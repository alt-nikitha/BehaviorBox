#!/bin/bash
#SBATCH --job-name=fam0_easiest_tokens
#SBATCH --output=./slurm-out/analysis/fam0_easiest_tokens_%j.out
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
python logreg_easiest_fam0_tokens.py \
    --min-r 0.85 --max-l1 0.3 --per-family 548236 \
    --exclude-family 3 --target-family 0 --top-n 40 \
    --out fam0_easiest_tokens.txt
