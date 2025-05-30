#!/bin/bash
#SBATCH --job-name=autolabel
#SBATCH --output=./slurm-out/analysis/autolabel_%j.out
#SBATCH --nodes=1
#SBATCH --mem=24GB
#SBATCH --time=1:00:00
#SBATCH --partition=general

set -a 
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate bbox-vllm

usage() {
  echo "Usage: $0 [--sae_dir=PATH] [--labeling_model=STRING] [--help]"
  echo
  echo "Options:"
  echo "  --sae_dir=PATH      Directory containing SAE model and outputs"
  echo "  --labeling_model=STRING Model to use for labeling"
  echo "  --help             Display this help message"
  exit 1
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --sae_dir=*)
      sae_dir="${1#*=}"
      shift
      ;;
    --labeling_model=*)
      labeling_model="${1#*=}"
      shift
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Validate required arguments
if [ -z "$sae_dir" ]; then
  echo "Error: --sae_dir is required"
  usage
fi

echo "Running on node: $HOSTNAME"
echo "SAE directory: $sae_dir"
echo "Labeling model: $labeling_model"

cd analysis/automatic_labeling

mkdir -p $sae_dir/feature_labels
python automatic_labels.py \
    --sae_dir $sae_dir \
    --labeling_model $labeling_model

python filter_and_validate_labels.py \
    --sae_dir $sae_dir