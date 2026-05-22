#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

TAU=0.75
ALPHA=5.0

OLMO_TASKS="arc_challenge,bbh,blimp,coqa,csqa,gsm8k,hellaswag,lambada,medmcqa,medqa,mmlu_humanities,mmlu_other,mmlu_pro,mmlu_social_sciences,mmlu_stem,naturalqs,piqa,sciq,squad,winogrande"
AMBER_TASKS="arc_challenge,bbh,blimp,coqa,csqa,gsm8k,hellaswag,lambada,medmcqa,minerva_math,mmlu_other,mmlu_social_sciences,mmlu_stem,naturalqs,piqa,winogrande"

echo "=== OLMo ==="
python prepare_training_data.py \
    --tasks "${OLMO_TASKS}" \
    --task_mapped_dir /data/user_data/nsrikant/bbox_data/causal_data/olmo/task_mapped_results \
    --word_ids_path /data/user_data/nsrikant/bbox_data/causal_data/olmo/raw_activations/word_ids.pkl \
    --source_data /data/user_data/nsrikant/bbox_data/data/olmo_256000_unseen.jsonl \
    --tokenizer allenai/Olmo-3-1025-7B \
    --tau ${TAU} --alpha ${ALPHA} \
    --output_dir /data/user_data/nsrikant/bbox_data/causal_data/olmo/expweight_tasks

# echo "=== Amber ==="
# python prepare_training_data.py \
#     --tasks "${AMBER_TASKS}" \
#     --task_mapped_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/task_mapped_results \
#     --word_ids_path /data/user_data/nsrikant/bbox_data/causal_data/amber/raw_activations/word_ids.pkl \
#     --source_data /data/user_data/nsrikant/bbox_data/data/amber_300_unseen.jsonl \
#     --tokenizer LLM360/Amber \
#     --tau ${TAU} --alpha ${ALPHA} \
#     --output_dir /data/user_data/nsrikant/bbox_data/causal_data/amber/expweight_tasks

echo "=== DONE ==="
