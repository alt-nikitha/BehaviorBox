#!/bin/bash
#SBATCH --job-name=autolabel
#SBATCH --output=/home/nsrikant/BehaviorBoxNew/slurm-out/sae/autolabel_%A_%a.out
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --mem=24GB
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1

# Batch-label the 4 family SAEs with gemini/gemini-2.5-pro (one array task per family).
# Needs GEMINI_API_KEY in the environment (export it before sbatch, or set it here).

set -euo pipefail
PY=/home/nsrikant/miniconda3/envs/bbox_env/bin/python
BASE=/data/user_data/nsrikant/bbox_data/sae_outputs/sae_outputs_olmo_256000_early_and_late

DIRS=(
 "$BASE/n_only_early_and_late_olmo3_68k_fam_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_fam1_68kjump_n3000000_N=3000_k=50_lp=None"
 "$BASE/n_only_early_and_late_olmo3_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_fam2_gradual_n3000000_N=3000_k=50_lp=None"
 "$BASE/n_only_early_and_late_olmo3_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_blimp_n3000000_N=3000_k=50_lp=None"
 "$BASE/n_only_early_and_late_olmo3_medmcqa_seed=42_ofw=0.0_nok_subset=sae_sample_by_family_medmcqa_n3000000_N=3000_k=50_lp=None"
)

SAE_DIR="${DIRS[$SLURM_ARRAY_TASK_ID]}"
echo "labeling: $SAE_DIR"

$PY /home/nsrikant/BehaviorBoxNew/analysis/label_features_litellm.py \
    --sae-dir "$SAE_DIR" \
    --model gemini/gemini-2.5-pro \
    --n-samples 10 \
    --min-acts 10 \
    --workers 8
