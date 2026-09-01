#!/bin/bash
#SBATCH --job-name=fam0_structure_val
#SBATCH --output=./slurm-out/analysis/fam0_structure_val_%j.out
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
python fam0_easy_structure_validation.py \
    --model fam0_logreg_weights.pkl \
    --target-family 0 --n-easy 10000 --n-pcs 10 \
    --min-lemma-family 5 --n-perm 500
