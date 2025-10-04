#!/bin/bash
#SBATCH --job-name=output-features
#SBATCH --output=./slurm-out/output_features_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48GB
#SBATCH --time 2-00:00:00
#SBATCH --partition=general

set -a 
source scripts/env_configs/.env
set +a

source ${MINICONDA_PATH}
conda init bash
conda activate ${ENV_NAME}

huggingface-cli login --token ${HF_TOKEN}

usage() {
  echo "Usage: $0 [--model_name=STRING] [--model_id=STRING] [--data=PATH] [--output_dir=PATH] [--batch_size=NUMBER] [--async_limiter=NUMBER] [--help]"
  echo
  echo "Options:"
  echo "  --model_name=STRING  Name for the model (e.g., olmo2-13b-sft)"
  echo "  --model_id=STRING   Huggingface model ID (e.g., allenai/OLMo-2-1124-13B-SFT)"
  echo "  --model_addr=STRING Address of vLLM server (e.g., hostname:port)"
  echo "  --data=PATH        Path to input data file in jsonl format"
  echo "  --output_dir=PATH  Directory to save output features"
  echo "  --batch_size=NUMBER Batch size for processing (default: 100)"
  echo "  --async_limiter=NUMBER Max number of async requests (default: 20)"
  echo "  --help            Display this help message"
  exit 1
}

# Default values
batch_size=100
async_limiter=20

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_name=*)
      model_name="${1#*=}"
      shift
      ;;
    --model_id=*)
      model_id="${1#*=}"
      shift
      ;;
    --model_addr=*)
      model_addr="${1#*=}"
      shift
      ;;
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
    --async_limiter=*)
      async_limiter="${1#*=}"
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
if [ -z "$model_name" ] || [ -z "$model_id" ] || [ -z "$model_addr" ] || [ -z "$data" ] || [ -z "$output_dir" ]; then
  echo "Error: --model_name, --model_id, --model_addr, --data and --output_dir are required"
  usage
fi

echo "Model name: $model_name"
echo "Model ID: $model_id"
echo "Model address: $model_addr"
echo "Input data: $data"
echo "Output directory: $output_dir"
echo "Batch size: $batch_size"
echo "Async limiter: $async_limiter"


echo $model_name $model_addr
cd data_generation
python output_features.py \
    --data ${data} \
    --output_dir ${output_dir}/output_features/${model_name} \
    --model_addr ${model_addr} \
    --model_id ${model_id} \
    --model_name ${model_name} \
    --batch_size ${batch_size} \
    --async_limiter ${async_limiter}
