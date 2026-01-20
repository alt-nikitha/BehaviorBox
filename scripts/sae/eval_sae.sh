#!/bin/bash
#SBATCH --job-name=sae
#SBATCH --output=./slurm-out/sae/eval_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=20
#SBATCH --time=2:00:00
#SBATCH --partition=general


set -a 
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}


seed=42
data_seed=0

export NCCL_P2P_DISABLE=1
module load cuda-12.4

GPU_ID=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
         awk '{print NR-1 ":" $1}' | sort -t: -k2 -nr | head -n1 | cut -d: -f1)
export CUDA_VISIBLE_DEVICES=$GPU_ID


sae_dir="/home/nsrikant/bbox_outputs/sae_outputs_larger_pile/n_moreearly_larger_pile_seed=42_ofw=0.7_N=3000_k=50_lp=None"
exp_cfg="/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n_moreearly_blimp_full_trained_on_pile.json"


filename=$(basename "$exp_cfg" .json)
model_string="${filename#config_}"


temp_dir="/scratch/$USER/tmp"




mkdir -p $temp_dir

cd sae

# python eval_sae_cross.py \
#     --sae_dir $sae_dir \
#     --args $exp_cfg \
#     --model_string $model_string \
#     --spill_dir $temp_dir \
#     --temp_dir $temp_dir \
#     --k 50


python eval_sae_cross.py
    