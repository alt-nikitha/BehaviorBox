#!/bin/bash
#SBATCH --job-name=sae
#SBATCH --output=./slurm-out/sae/train_and_eval_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --partition=general

set -a 
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

usage() {
  echo "Usage: $0 [--exp_cfg=STRING] [--hp_cfg=STRING] [--prefix=STRING] [--seed=NUMBER] [--data_seed=NUMBER] [--help]"
  echo
  echo "Options:"
  echo "  --exp_cfg=STRING    Path to experiment config file (default: empty)"
  echo "  --hp_cfg=STRING    Path to hyperparam config file (default: empty)"
  echo "  --prefix=STRING    Additional prefix for the SAE name (default: empty)"
  echo "  --seed=NUMBER    Random seed value for model initialization (default: 42)"
  echo "  --data_seed=NUMBER    Random seed value for data shuffling (default: 0)"
  echo "  --help           Display this help message"
  exit 1
}

seed=42
data_seed=0

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --exp_cfg=*)
      exp_cfg="${1#*=}"
      shift
      ;;
    --hp_cfg=*)
      hp_cfg="${1#*=}"
      shift
      ;;
    --prefix=*)
      prefix="${1#*=}"
      shift
      ;;
    --seed=*)
      seed="${1#*=}"
      shift
      ;;
    --data_seed=*)
      data_seed="${1#*=}"
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

# Experimental and hyperparameter configs must be provided
if [ -z "$exp_cfg" ] || [ -z "$hp_cfg" ]; then
  echo "Error: --exp_cfg and --hp_cfg are required"
  usage
fi

# Validate seed is a number
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
  echo "Error: Seed must be a positive integer"
  usage
fi
if ! [[ "$data_seed" =~ ^[0-9]+$ ]]; then
  echo "Error: Data seed must be a positive integer"
  usage
fi

echo "Path to experiment arguents: $exp_cfg"
echo "Path to hyperparameter config: $hp_cfg"
echo "Seed: $seed"
echo "Data seed: $data_seed"

export NCCL_P2P_DISABLE=1
module load cuda-12.4

model_string=$(jq -r '.model_names | join("_")' "$args")
ofw=$(jq -r '.output_feature_weight' "$args")
if [ -n "$prefix" ] ; then
    sae_name_prefix="${prefix}_${model_string}_seed=${seed}_ofw=${ofw}"
else
    sae_name_prefix="${model_string}_seed=${seed}_ofw=${ofw}"
fi

# dask spill dir
temp_dir="/scratch/$USER/tmp"

mkdir -p $temp_dir

cd sae
python train_sae.py \
    --args $exp_cfg \
    --config_path $hp_cfg \
    --data_shuffling_seed $data_seed \
    --seed $seed \
    --sae_name_prefix $sae_name_prefix \
    --spill_dir $temp_dir \
    --temp_dir $temp_dir \
    --workers 20

if [ $? -ne 0 ]; then
    echo "Training failed with exit code $?. Terminating."
    rm -r $temp_dir
    echo "$temp_dir removed"
    exit 1
fi

echo "Training complete. Running eval."

python eval_sae.py \
    --args $exp_cfg \
    --config_path $hp_cfg \
    --sae_name_prefix $sae_name_prefix \
    --spill_dir $temp_dir \
    --temp_dir $temp_dir \
    --workers 16 \
    --save_activations True

rm -r $temp_dir
echo "$temp_dir removed"
