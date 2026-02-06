#!/bin/bash
#SBATCH --job-name=lm-eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# =============================================================================
# LM Evaluation Harness - SLURM Array Job
# Each array task runs one evaluation task from eval_config.json
# =============================================================================

set -e
source /home/nsrikant/miniconda3/etc/profile.d/conda.sh
conda activate bbox_env

# Configuration
CONFIG_FILE="${CONFIG_FILE:-./eval_config.json}"
MODEL_PATH="${MODEL_PATH:-allenai/Olmo-3-1025-7B}"
MODEL_REVISION="${MODEL_REVISION:-main}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
BATCH_SIZE="${BATCH_SIZE:-auto}"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Get task index from SLURM array
TASK_IDX=${SLURM_ARRAY_TASK_ID}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Get total number of tasks for validation
NUM_TASKS=$(jq '.tasks | length' "$CONFIG_FILE")

if [ "$TASK_IDX" -ge "$NUM_TASKS" ]; then
    echo "Error: Task index $TASK_IDX exceeds number of tasks ($NUM_TASKS)"
    exit 1
fi

# Extract task configuration using jq
task_name=$(jq -r ".tasks[$TASK_IDX].name" "$CONFIG_FILE")
lm_eval_task=$(jq -r ".tasks[$TASK_IDX].lm_eval_task" "$CONFIG_FILE")
category=$(jq -r ".tasks[$TASK_IDX].category" "$CONFIG_FILE")
icl=$(jq -r ".tasks[$TASK_IDX].icl" "$CONFIG_FILE")
format=$(jq -r ".tasks[$TASK_IDX].format" "$CONFIG_FILE")
temp=$(jq -r ".tasks[$TASK_IDX].temp" "$CONFIG_FILE")
top_p=$(jq -r ".tasks[$TASK_IDX].top_p" "$CONFIG_FILE")
max_tokens=$(jq -r ".tasks[$TASK_IDX].max_tokens" "$CONFIG_FILE")
min_tokens=$(jq -r ".tasks[$TASK_IDX].min_tokens" "$CONFIG_FILE")

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on: $(hostname)"
echo "=========================================="
echo "Task: $task_name ($lm_eval_task)"
echo "Category: $category | Format: $format | ICL: $icl-shot"
echo "Model: $MODEL_PATH"
echo "Revision: $MODEL_REVISION"
echo "=========================================="
echo ""

# Build the base command
cmd="lm_eval --model vllm"
cmd+=" --model_args pretrained=$MODEL_PATH"
cmd+=",revision=$MODEL_REVISION"
cmd+=",gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
cmd+=",dtype=auto"
cmd+=",trust_remote_code=True"
cmd+=" --tasks $lm_eval_task"
cmd+=" --num_fewshot $icl"
cmd+=" --batch_size $BATCH_SIZE"
cmd+=" --output_path $OUTPUT_DIR/${task_name}"
cmd+=" --log_samples"

# Add generation kwargs for generative tasks (CoT, Code Exec, GenQA)
if [[ "$format" == "CoT EM" || "$format" == "Code Exec" || "$format" == "GenQA" ]]; then
    gen_kwargs=""
    
    # Add temperature if specified
    if [[ "$temp" != "null" && "$temp" != "" ]]; then
        if [[ "$gen_kwargs" != "" ]]; then
            gen_kwargs+=","
        fi
        gen_kwargs+="temperature=$temp"
    fi
    
    # Add top_p if specified
    if [[ "$top_p" != "null" && "$top_p" != "" ]]; then
        if [[ "$gen_kwargs" != "" ]]; then
            gen_kwargs+=","
        fi
        gen_kwargs+="top_p=$top_p"
    fi
    
    # Add max_tokens if specified
    if [[ "$max_tokens" != "null" && "$max_tokens" != "" ]]; then
        if [[ "$gen_kwargs" != "" ]]; then
            gen_kwargs+=","
        fi
        gen_kwargs+="max_gen_toks=$max_tokens"
    fi
    
    # Add do_sample based on temperature
    if [[ "$temp" != "null" && "$temp" != "" && "$temp" != "0" ]]; then
        if [[ "$gen_kwargs" != "" ]]; then
            gen_kwargs+=","
        fi
        gen_kwargs+="do_sample=True"
    fi

    if [[ "$min_tokens" != "null" && "$min_tokens" != "" ]]; then
        if [[ "$gen_kwargs" != "" ]]; then
            gen_kwargs+=","
        fi
        gen_kwargs+="min_tokens=$min_tokens"
    fi
    
    if [[ "$gen_kwargs" != "" ]]; then
        cmd+=" --gen_kwargs $gen_kwargs"
    fi
fi

echo "Command: $cmd"
echo ""

# Run the evaluation
eval $cmd

echo ""
echo "=========================================="
echo "✓ Completed: $task_name"
echo "Results saved to: $OUTPUT_DIR/${task_name}"
echo "Finished at $(date)"
echo "=========================================="