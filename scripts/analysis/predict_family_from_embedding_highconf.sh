#!/bin/bash
#SBATCH --job-name=fam_embed_probe_hc
#SBATCH --output=./slurm-out/analysis/fam_embed_probe_hc_%j.out
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=32
#SBATCH --time=2:00:00
#SBATCH --partition=cpu

set -a
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

echo "Running on node: $HOSTNAME"

cd analysis
python predict_family_from_embedding_highconf.py \
    --min-r 0.85 --max-l1 0.3 --per-family 548236 \
    --exclude-family 3 --models logreg rf
