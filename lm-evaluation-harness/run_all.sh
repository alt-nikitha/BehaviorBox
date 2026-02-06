#!/bin/bash
#SBATCH --job-name=olmo-eval
#SBATCH --array=0-10%10      
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00     
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

# --- SETUP ---
modelname="olmo"
modelbase="allenai/OLMo-2-1124-7B"
CKPT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" ${modelname}_checkpoints.txt)
OUT_DIR="results/$modelname/hybrid/$(basename $CKPT)/"
mkdir -p $OUT_DIR

echo "Processing Checkpoint: $CKPT"

TASKS_FAST="drop,nq_open,triviaqa"


lm_eval --model vllm \
    --model_args pretrained=$modelbase,revision=$CKPT,dtype=bfloat16,gpu_memory_utilization=0.90 \
    --tasks $TASKS_FAST \
    --batch_size auto \
    --output_path $OUT_DIR \
    --gen_kwargs min_tokens=1


HEAVY_TASK_LIST="gsm8k mmlu_pro agieval"

for task in $HEAVY_TASK_LIST; do
    echo ">>> Running Task: $task ..."
    
    lm_eval --model vllm \
        --model_args pretrained=$modelbase,revision=$CKPT,dtype=bfloat16,gpu_memory_utilization=0.90 \
        --tasks $task \
        --batch_size auto \
        --output_path $OUT_DIR \
        --gen_kwargs min_tokens=1
        
    echo ">>> Finished $task"
done

echo "Evaluation Complete for $CKPT"