#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --output=slurm-out/vllm/host-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=48GB
#SBATCH --time 2-00:00:00
#SBATCH --partition=general

set -a 
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

# These may need to be adjusted based on your setup
PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "Selected port: $PORT"
tensor_parallel_size=1
gpu_memory_utilization=0.9  # Increase to use more GPU memory

usage() {
  echo "Usage: $0 [--model_id=STRING] [--model_name=STRING] [--revision=STRING] [--help]"
  echo
  echo "Options:"
  echo "  --model_id=STRING  Huggingface model ID (e.g., allenai/OLMo-2-1124-13B)"
  echo "  --model_name=STRING Name for the model (e.g., olmo2-13b-sft)"
  echo "  --revision=STRING  Optional HF branch/tag/revision to load"
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
    --model_name=*)
      model_name="${1#*=}"
      shift
      ;;
    --revision=*)
      REVISION="${1#*=}"
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
if [ -z "$model_id" ] || [ -z "$model_name" ]; then
  echo "Error: --model_id and --model_name are required"
  usage
fi

echo "Model ID: $model_id"
echo "Port: $PORT"
echo "Tensor parallel size: $tensor_parallel_size"
echo "GPU memory utilization: $gpu_memory_utilization"
if [ -n "$REVISION" ]; then
  echo "Revision: $REVISION"
fi

huggingface-cli login --token ${HF_TOKEN}

export VLLM_LOGGING_LEVEL=ERROR
export NCCL_P2P_DISABLE=1

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (set by SLURM)"
echo "Using GPU allocated by SLURM"
mkdir -p scripts/data_generation/tmp/${model_name}

# Get hostname for cluster-wide connectivity (hostnames are resolvable, IPs may not be)
HOST=$(hostname)
echo "Server will be accessible at: ${HOST}:${PORT}"

# Check if the port is already in use before attempting to start
if ss -tulwn 2>/dev/null | grep -q ":$PORT " || netstat -tulwn 2>/dev/null | grep -q ":$PORT "; then
    echo "Port $PORT is already in use. Exiting..."
    exit 1
fi

# Launch the server in background and write logs to SLURM output folder
mkdir -p slurm-out/vllm
SERVER_LOG=slurm-out/vllm/server-${PORT}.log
echo "Server logs will be written to: ${SERVER_LOG}"

python -m vllm.entrypoints.openai.api_server \
  --gpu_memory_utilization $gpu_memory_utilization \
  --model $model_id \
  --served-model-name $model_name \
  --port $PORT \
  --tensor-parallel-size $tensor_parallel_size \
  ${REVISION:+ --revision ${REVISION}} \
  --disable-log-requests \
  --download-dir ${HF_HOME} \
  &> "$SERVER_LOG" &

VLLM_PID=$!
echo "Started vLLM (pid=$VLLM_PID), waiting for server to be ready..."

# Function to check if server is ready
check_server_ready() {
  # Method 1: Check if port is bound using ss (try multiple patterns)
  if command -v ss &> /dev/null; then
    # Try exact port match with word boundary
    if ss -tulwn 2>/dev/null | grep -E ":${PORT}\b" > /dev/null; then
      return 0
    fi
    # Try looser match
    if ss -tulwn 2>/dev/null | grep ":${PORT}" > /dev/null; then
      return 0
    fi
  fi
  
  # Method 2: Fallback to netstat
  if command -v netstat &> /dev/null; then
    if netstat -tulwn 2>/dev/null | grep -E ":${PORT}\b" > /dev/null; then
      return 0
    fi
    if netstat -tulwn 2>/dev/null | grep ":${PORT}" > /dev/null; then
      return 0
    fi
  fi
  
  # Method 3: Try to connect via HTTP if curl is available
  if command -v curl &> /dev/null; then
    if curl -f -s --connect-timeout 1 --max-time 2 "http://localhost:${PORT}/health" > /dev/null 2>&1; then
      return 0
    fi
  fi
  
  # Method 4: Try using /dev/tcp if available (bash built-in)
  if timeout 2 bash -c "echo > /dev/tcp/localhost/${PORT}" 2>/dev/null; then
    return 0
  fi
  
  return 1
}

# Wait for the server to bind to the port (timeout after 10 minutes)
timeout_secs=600
elapsed=0
last_log_time=0
check_count=0

echo "Debug: Starting port check loop for port ${PORT}"
echo "Debug: Available tools - curl: $(command -v curl), ss: $(command -v ss), netstat: $(command -v netstat)"

while ! check_server_ready; do
  check_count=$((check_count+1))
  
  # Check if process is still alive
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM process died unexpectedly" >&2
    echo "Check logs at: ${SERVER_LOG}" >&2
    echo "Last 50 lines:" >&2
    tail -n 50 "${SERVER_LOG}" >&2
    exit 1
  fi
  
  sleep 2
  elapsed=$((elapsed+2))
  
  # Print progress every 30 seconds with debug info
  if [ $((elapsed - last_log_time)) -ge 30 ]; then
    echo "Waiting for server to be ready... (${elapsed}s elapsed, PID=${VLLM_PID}, checks=${check_count})"
    if [ -f "${SERVER_LOG}" ]; then
      echo "Latest from log: $(tail -n 1 ${SERVER_LOG})"
    fi
    
    # Show what ports are actually listening every 60 seconds
    if [ $((elapsed % 60)) -eq 0 ]; then
      echo "Debug: Currently listening ports:"
      if command -v ss &> /dev/null; then
        ss -tulwn 2>/dev/null | grep LISTEN | head -5
      elif command -v netstat &> /dev/null; then
        netstat -tulwn 2>/dev/null | grep LISTEN | head -5
      fi
    fi
    
    last_log_time=$elapsed
  fi
  
  if [ $elapsed -ge $timeout_secs ]; then
    echo "ERROR: Server failed to become ready after ${timeout_secs}s" >&2
    echo "Debug info:"
    echo "  Port to check: ${PORT}"
    echo "  Process alive: $(kill -0 $VLLM_PID 2>/dev/null && echo yes || echo no)"
    echo "  All listening ports:"
    ss -tulwn 2>/dev/null || netstat -tulwn 2>/dev/null
    echo ""
    echo "Last 50 lines of server log:" >&2
    tail -n 50 "${SERVER_LOG}" >&2
    kill $VLLM_PID 2>/dev/null || true
    exit 1
  fi
done

echo "✓ Server is ready and responding!"

# Write the host:port file using hostname
echo "${HOST}:${PORT}" > scripts/data_generation/tmp/${model_name}/host_port.txt
echo "Wrote host_port.txt -> ${HOST}:${PORT}"

# Wait for model to be fully registered in /v1/models
echo "Waiting for model '${model_name}' to be registered..."
model_ready_timeout=300
model_elapsed=0

while ! curl -f -s --connect-timeout 2 --max-time 5 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "${model_name}"; do
  sleep 5
  model_elapsed=$((model_elapsed+5))
  
  if [ $model_elapsed -ge $model_ready_timeout ]; then
    echo "WARNING: Model not showing in /v1/models after ${model_ready_timeout}s"
    echo "Available models:"
    curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null || echo "(could not fetch models list)"
    echo "Proceeding anyway..."
    break
  fi
  
  if [ $((model_elapsed % 30)) -eq 0 ]; then
    echo "Waiting for model registration... (${model_elapsed}s elapsed)"
  fi
done

echo "✓ Model '${model_name}' is registered and ready!"

# Create a ready flag file to signal completion
touch scripts/data_generation/tmp/${model_name}/ready.flag
echo "Created ready flag at: scripts/data_generation/tmp/${model_name}/ready.flag"

echo ""
echo "=========================================="
echo "vLLM server is fully ready!"
echo "Address: ${HOST}:${PORT}"
echo "Model: ${model_name}"
echo "PID: ${VLLM_PID}"
echo "=========================================="
echo ""

# Keep the script alive by waiting on the server process so SLURM job remains
wait $VLLM_PID