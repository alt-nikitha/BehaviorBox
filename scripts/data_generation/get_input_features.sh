#!/bin/bash
#SBATCH --job-name=input-features
#SBATCH --output=./slurm-out/input_features_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=96GB
#SBATCH --time 2-00:00:00
#SBATCH --partition=general

set -a 
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate bbox-vllm


usage() {
  echo "Usage: $0 [--data=PATH] [--output_dir=PATH] [--batch_size=NUMBER] [--help]"
  echo
  echo "Options:"
  echo "  --data=PATH       Path to input data file in jsonl format"
  echo "  --output_dir=PATH Directory to save output features"
  echo "  --batch_size=NUMBER Batch size for processing (default: 1)"
  echo "  --help           Display this help message"
  exit 1
}

# Default values
batch_size=5

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data=*)
      data="${1#*=}"
      shift
      ;;
    --output_dir=*)
      output_dir="${1#*=}"
      shift
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
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
if [ -z "$data" ] || [ -z "$output_dir" ]; then
  echo "Error: --data and --output_dir are required"
  usage
fi

echo "Input data: $data"
echo "Output directory: $output_dir"
echo "Batch size: $batch_size"

mkdir -p ${output_dir}
cd data_generation
CUDA_LAUNCH_BLOCKING=1 python input_features.py \
    --data ${data} \
    --output_dir ${output_dir}/input_features \
    --batch_size $batch_size