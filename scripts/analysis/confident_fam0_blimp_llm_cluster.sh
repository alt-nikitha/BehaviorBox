#!/bin/bash
#SBATCH --job-name=fam0_blimp_llm_cluster
#SBATCH --output=./slurm-out/analysis/fam0_blimp_llm_cluster_%j.out
#SBATCH --nodes=1
#SBATCH --mem=60GB
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --partition=neulab
#SBATCH --qos=neulab_qos
#SBATCH --gres=gpu:1

set -a
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

echo "Running on node: $HOSTNAME"

cd analysis
python confident_fam0_blimp_llm_cluster.py \
    --model fam0_logreg_weights.pkl \
    --llm-model gemini/gemini-2.5-pro \
    --n-confident 2000 --k-min 2 --k-max 8 \
    --n-llm-examples 15 --max-show 20 \
    --llm-context-words 40 --display-context-words 20
