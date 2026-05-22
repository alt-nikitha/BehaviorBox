#!/bin/bash

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

echo "Model name: $model_name"
echo "Model ID: $model_id"

# Check if vLLM server is already running for this model
model_addr=""
server_already_running=false

if [ -f "scripts/data_generation/tmp/${model_name}/ready.flag" ] && \
   [ -f "scripts/data_generation/tmp/${model_name}/host_port.txt" ]; then
  model_addr=$(cat "scripts/data_generation/tmp/${model_name}/host_port.txt")
  echo "✓ Found existing vLLM server for ${model_name} at ${model_addr}"
  echo "  (Trusting ready.flag - cannot verify from login node)"
  server_already_running=true
fi

# Launch vLLM server only if not already running
if [ "$server_already_running" = false ]; then
  echo ""
  echo "=========================================="
  echo "Launching new vLLM server for ${model_name}..."
  echo "=========================================="
  echo ""
  
  # Clean up any stale files from previous runs
  rm -f scripts/data_generation/tmp/${model_name}/host_port.txt
  rm -f scripts/data_generation/tmp/${model_name}/ready.flag
  rm -f scripts/data_generation/tmp/${model_name}/vllm_jobid.txt
  
  # Create directory for this model
  mkdir -p scripts/data_generation/tmp/${model_name}
  
  if [ -z "$slurm" ]; then
    # Direct execution (not recommended for production)
    if [ -n "$revision" ]; then
      bash scripts/data_generation/vllm_host.sh --model_id="$model_id" --model_name="$model_name" --revision="$revision" &
    else
      bash scripts/data_generation/vllm_host.sh --model_id="$model_id" --model_name="$model_name" &
    fi
    VLLM_PID=$!
    echo "Launched vLLM directly with PID: $VLLM_PID"
  else
    # SLURM execution
    if [ -n "$revision" ]; then
      VLLM_JOBID=$(sbatch --parsable scripts/data_generation/vllm_host.sh --model_id="$model_id" --model_name="$model_name" --revision="$revision")
    else
      VLLM_JOBID=$(sbatch --parsable scripts/data_generation/vllm_host.sh --model_id="$model_id" --model_name="$model_name")
    fi
    echo "✓ Launched vLLM via SLURM job: $VLLM_JOBID"
    echo "$VLLM_JOBID" > scripts/data_generation/tmp/${model_name}/vllm_jobid.txt
    echo ""
  fi

  # Wait for the server to signal it's ready
  echo "Waiting for vLLM server to become ready..."
  echo "This may take several minutes for model loading and initialization..."
  echo ""
  
  timeout_secs=1200  # 20 minutes
  elapsed=0
  last_status_time=0

  while [ ! -f "scripts/data_generation/tmp/${model_name}/ready.flag" ]; do
    # Print status every 30 seconds
    if [ $((elapsed - last_status_time)) -ge 30 ]; then
      echo "[${elapsed}s] Waiting for ready signal..."
      
      # Check if host_port.txt exists yet
      if [ -f "scripts/data_generation/tmp/${model_name}/host_port.txt" ]; then
        echo "  → host_port.txt found: $(cat scripts/data_generation/tmp/${model_name}/host_port.txt)"
      else
        echo "  → host_port.txt not yet created (server still starting up)"
      fi
      
      # Check if vLLM job is still running (if using SLURM)
      if [ -n "$VLLM_JOBID" ]; then
        if squeue -j $VLLM_JOBID &> /dev/null; then
          job_state=$(squeue -j $VLLM_JOBID -h -o "%T")
          echo "  → SLURM job $VLLM_JOBID status: $job_state"
        else
          echo ""
          echo "ERROR: vLLM SLURM job $VLLM_JOBID disappeared from queue!"
          echo "This usually means the job failed or was cancelled."
          echo "Check the SLURM output: slurm-out/vllm/host-${VLLM_JOBID}.out"
          echo ""
          exit 1
        fi
      fi
      
      last_status_time=$elapsed
    fi
    
    sleep 10
    elapsed=$((elapsed+10))
    
    if [ $elapsed -ge $timeout_secs ]; then
      echo ""
      echo "ERROR: Timeout waiting for vLLM server to become ready (${timeout_secs}s)"
      echo ""
      if [ -n "$VLLM_JOBID" ]; then
        echo "Check SLURM output: slurm-out/vllm/host-${VLLM_JOBID}.out"
        echo "Cancelling SLURM job..."
        scancel $VLLM_JOBID
      fi
      exit 1
    fi
  done

  echo ""
  echo "✓ vLLM server is ready!"
  model_addr=$(cat "scripts/data_generation/tmp/${model_name}/host_port.txt")
  echo "✓ Server address: $model_addr"
  echo ""
else
  echo ""
  echo "Using existing vLLM server"
  echo ""
fi

# Note: We trust the ready.flag from vLLM since it checks itself locally on the compute node
# We cannot verify from the login node due to network isolation on HPC clusters
echo "=========================================="
echo "Server ready (verified by vLLM on compute node)"
echo "Proceeding with output features extraction..."
echo "=========================================="
echo ""

# Run output features extraction
if [ -z "$slurm" ]; then
  echo "Running output_features.sh directly..."
  bash scripts/data_generation/output_features.sh \
      --data="${data}" \
      --output_dir="${output_dir}" \
      --model_addr="${model_addr}" \
      --model_id="${model_id}" \
      --model_name="${model_name}" \
      --batch_size="${batch_size}" \
      --async_limiter="${async_limiter}"
else
  echo "Submitting output_features job to SLURM..."
  FEATURES_JOBID=$(sbatch --parsable scripts/data_generation/output_features.sh \
      --data="${data}" \
      --output_dir="${output_dir}" \
      --model_addr="${model_addr}" \
      --model_id="${model_id}" \
      --model_name="${model_name}" \
      --batch_size="${batch_size}" \
      --async_limiter="${async_limiter}")
  echo "✓ Submitted output_features job: $FEATURES_JOBID"
  echo ""

  # Wait for the job to start
  echo "Waiting for features extraction job to start..."
  sleep 5

  # Monitor the job
  echo "Monitoring job progress (check slurm-out/output_features_${FEATURES_JOBID}.out for details)..."
  while squeue -j $FEATURES_JOBID > /dev/null 2>&1; do
      sleep 20
  done
  
  echo ""
  echo "✓ Output features job finished."
  echo ""

  # Clean up vLLM server if we launched it
  if [ "$server_already_running" = false ] && [ -n "$VLLM_JOBID" ]; then
    echo "Cleaning up vLLM server..."
    echo "Killing vLLM server job: $VLLM_JOBID"
    scancel $VLLM_JOBID

    # Remove metadata files
    rm -f scripts/data_generation/tmp/${model_name}/host_port.txt
    rm -f scripts/data_generation/tmp/${model_name}/ready.flag
    rm -f scripts/data_generation/tmp/${model_name}/vllm_jobid.txt
    
    echo "✓ Cleanup complete"
  fi
fi

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="