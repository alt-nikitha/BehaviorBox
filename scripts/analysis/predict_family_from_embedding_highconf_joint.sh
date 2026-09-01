#!/bin/bash
#SBATCH --job-name=fam_embed_probe_hc_joint
#SBATCH --output=./slurm-out/analysis/fam_embed_probe_hc_joint_%j.out
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
    --assign-cache family_assign_4fam_c4df2b.npz \
    --min-r 0.6 --max-l1 0.5 --exclude-family 3
