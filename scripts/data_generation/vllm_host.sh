#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --output=slurm-out/vllm/host-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=48GB
#SBATCH --time 2-00:00:00
#SBATCH --partition=general

set -a 
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate bbox-vllm

# These may need to be adjusted based on your setup
PORT=8084
tensor_parallel_size=2
gpu_memory_utilization=0.8

usage() {
  echo "Usage: $0 [--model_id=STRING] [--help]"
  echo
  echo "Options:"
  echo "  --model_id=STRING  Huggingface model ID (e.g., allenai/OLMo-2-1124-13B)"
  echo "  --help            Display this help message"
  exit 1
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_id=*)
      model_id="${1#*=}"
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
if [ -z "$model_id" ]; then
  echo "Error: --model_id is required"
  usage
fi

echo "Model ID: $model_id"
echo "Port: $PORT"
echo "Tensor parallel size: $tensor_parallel_size"
echo "GPU memory utilization: $gpu_memory_utilization"

huggingface-cli login --token ${HF_TOKEN}

export VLLM_LOGGING_LEVEL=ERROR
export NCCL_P2P_DISABLE=1

mkdir -p scripts/data_generation/tmp/${model_id}

echo "$HOSTNAME:$PORT" > scripts/data_generation/tmp/${model_id}/host_port.txt

# lowering GPU utilizaition to test
if ss -tulwn | grep -q ":$PORT "; then
    echo "Port $PORT is already in use. Exiting..."
    exit 1
else
    python -m vllm.entrypoints.openai.api_server \
        --gpu_memory_utilization $gpu_memory_utilization \
        --model $model_id \
        --port $PORT \
        --tensor-parallel-size $tensor_parallel_size \
        --device cuda \
        --enable-chunked-prefill \
        --disable-log-requests \
        --download-dir ${HF_HOME} # Either shared model cache on babel or your own directory
fi
echo $PORT