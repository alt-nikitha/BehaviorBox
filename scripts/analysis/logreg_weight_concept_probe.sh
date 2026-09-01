#!/bin/bash
#SBATCH --job-name=logreg_concept_probe
#SBATCH --output=./slurm-out/analysis/logreg_concept_probe_%j.out
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
python logreg_weight_concept_probe.py \
    --model fam0_logreg_weights.pkl \
    --target-family 0 \
    --n-sample 20000
