#!/bin/bash
#SBATCH --job-name=olmo-eval
#SBATCH --array=0-5%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -eo pipefail
shopt -s nullglob

source /home/nsrikant/miniconda3/etc/profile.d/conda.sh
conda activate bbox_env

modelname="olmo3"
modelbase="allenai/Olmo-3-1025-7B"
modelbase_safe="${modelbase//\//__}"
CKPT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /home/nsrikant/BehaviorBoxNew/checkpoints_info/olmo3_7b_checkpoints.txt)
OUT_DIR="eval_results_olmo3/$(basename $CKPT)"
mkdir -p "$OUT_DIR"
find "$OUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +

echo "Processing Checkpoint: $CKPT"

TASKS="arc_challenge,bbh,hellaswag,piqa,winogrande,commonsense_qa,medmcqa,mmlu_stem,mmlu_social_sciences,mmlu_other,blimp,coqa,gsm8k,lambada_openai,nq_open"

declare -A RENAME=( [commonsense_qa]=csqa [nq_open]=naturalqs [lambada_openai]=lambada )

lm_eval --model vllm \
    --model_args pretrained=$modelbase,revision=$CKPT,dtype=bfloat16,gpu_memory_utilization=0.90 \
    --tasks $TASKS \
    --batch_size auto \
    --output_path "$OUT_DIR" \
    --log_samples \
    --gen_kwargs min_tokens=1

SRC="$OUT_DIR/$modelbase_safe"
RESULTS_JSON=("$SRC"/results_*.json)
for f in "$SRC"/samples_*.jsonl; do
    actual=$(basename "$f" | sed -E 's/^samples_(.+)_[0-9]{4}-[0-9]{2}-[0-9]{2}T.*/\1/')
    friendly=${RENAME[$actual]:-$actual}
    dest="$OUT_DIR/$friendly/$modelbase_safe"
    mkdir -p "$dest"
    mv "$f" "$dest/"
    cp "${RESULTS_JSON[@]}" "$dest/"
done
rm -rf "$SRC"

echo "Evaluation Complete for $CKPT"