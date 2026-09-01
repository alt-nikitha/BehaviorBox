#!/bin/bash
#SBATCH --job-name=fam_embed_probe
#SBATCH --output=./slurm-out/analysis/fam_embed_probe_%j.out
#SBATCH --nodes=1
#SBATCH --mem=180GB
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --partition=cpu

set -a
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

echo "Running on node: $HOSTNAME"

cd analysis
python predict_family_from_embedding.py \
    --assign-cache family_assign_4fam_c4df2b.npz \
    --per-family 3800000
