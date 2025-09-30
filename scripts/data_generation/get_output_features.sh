usage() {
  echo "Usage: $0 [--model_name=STRING] [--model_id=STRING] [--revision=STRING] [--data=PATH] [--output_dir=PATH] [--batch_size=NUMBER] [--async_limiter=NUMBER] [--slurm] [--help]"
  echo
  echo "Options:"
  echo "  --model_name=STRING  Name for the model (e.g., olmo2-13b-sft)"
  echo "  --model_id=STRING   Huggingface model ID (e.g., allenai/OLMo-2-1124-13B-SFT)"
  echo "  --revision=STRING   Optional HF branch/tag/revision to load"
  echo "  --data=PATH        Path to input data file in jsonl format"
  echo "  --output_dir=PATH  Directory to save output features"
  echo "  --batch_size=NUMBER Batch size for processing (default: 100)"
  echo "  --async_limiter=NUMBER Max number of async requests (default: 20)"
  echo "  --slurm            Run scripts with SLURM job scheduler"
  echo "  --help            Display this help message"
  exit 1
}

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
    --revision=*)
      revision="${1#*=}"
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
    --slurm)
      slurm=true
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
if [ -z "$model_name" ] || [ -z "$model_id" ] || [ -z "$data" ] || [ -z "$output_dir" ]; then
  echo "Error: --model_name, --model_id, --data and --output_dir are required"
  usage
fi

# First launch vLLM server
echo "Model name: $model_name"
echo "Model ID: $model_id"
if [ -z "$slurm" ]; then
  if [ -n "$revision" ]; then
    bash scripts/data_generation/vllm_host.sh --model_id="$model_id" --revision="$revision"
  else
    bash scripts/data_generation/vllm_host.sh --model_id="$model_id"
  fi
else
  if [ -n "$revision" ]; then
    sbatch scripts/data_generation/vllm_host.sh --model_id="$model_id" --revision="$revision"
  else
    sbatch scripts/data_generation/vllm_host.sh --model_id="$model_id"
  fi
fi

# Wait for the server to start
echo "Waiting for the vLLM server to start..."
# Poll for address
# Address is written to scripts/data_generation/tmp/${model_id}/host_port.txt
model_addr=""
while [ -z "$model_addr" ]; do
  if [ -f "scripts/data_generation/tmp/${model_id}/host_port.txt" ]; then
    model_addr=$(cat "scripts/data_generation/tmp/${model_id}/host_port.txt")
  else
    echo "Waiting for scripts/data_generation/tmp/${model_id}/host_port.txt to be created..."
    sleep 5
  fi
done

# Poll for the server to be ready
echo "Model address found: $model_addr"
while ! curl -f -s http://${model_addr}/health > /dev/null 2>&1; do
    echo "Waiting for vLLM server to be ready..."
    sleep 10
done
echo "vLLM server is ready!"

echo "Model address: $model_addr"
if [ -z "$slurm" ]; then
  echo "Running output_features.sh..."
  bash scripts/data_generation/output_features.sh \
      --data="${data}" \
      --output_dir="${output_dir}" \
      --model_addr="${model_addr}" \
      --model_id="${model_id}" \
      --model_name="${model_name}" \
      --batch_size="${batch_size}" \
      --async_limiter="${async_limiter}"
else
  sbatch scripts/data_generation/output_features.sh \
      --data="${data}" \
      --output_dir="${output_dir}" \
      --model_addr="${model_addr}" \
      --model_id="${model_id}" \
      --model_name="${model_name}" \
      --batch_size="${batch_size}" \
      --async_limiter="${async_limiter}"
fi