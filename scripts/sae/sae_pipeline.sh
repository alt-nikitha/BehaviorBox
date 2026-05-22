#!/bin/bash
#SBATCH --job-name=sae
#SBATCH --output=./slurm-out/sae/train_and_eval_%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=20
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --exclude=babel-m5-32,babel-n9-32,babel-n9-28,babel-p9-28


set -a 
source scripts/env_configs/.env
set +a

# Activate environment
source ${MINICONDA_PATH}
conda activate ${ENV_NAME}

usage() {
  echo "Usage: $0 [--exp_cfg=STRING] [--hp_cfg=STRING] [--prefix=STRING] [--seed=NUMBER] [--data_seed=NUMBER] [--num_epochs=NUMBER] [--use_delta_prob] [--use_freq_weighting] [--freq_smoothing=NUMBER] [--freq_power=NUMBER] [--normalize_per_part] [--output_dim_loss_weight=STRING] [--help]"
  echo
  echo "Options:"
  echo "  --exp_cfg=STRING                 Path to experiment config file (default: empty)"
  echo "  --hp_cfg=STRING                  Path to hyperparam config file (default: empty)"
  echo "  --prefix=STRING                  Additional prefix for the SAE name (default: empty)"
  echo "  --seed=NUMBER                    Random seed value for model initialization (default: 42)"
  echo "  --data_seed=NUMBER               Random seed value for data shuffling (default: 0)"
  echo "  --num_epochs=NUMBER              Number of training epochs (default: 1)"
  echo "  --use_delta_prob                 Use consecutive prob deltas (p_i - p_{i+1}) as output features"
  echo "  --use_delta_logprob              Use consecutive logprob deltas (log p_i - log p_{i+1}) as output features"
  echo "  --decoder_ortho=NUMBER           Decoder orthogonality penalty weight (default 0; e.g. 0.1)"
  echo "  --use_freq_weighting             Enable inverse frequency weighting for rare tokens"
  echo "  --freq_smoothing=NUMBER          Smoothing factor for frequency weights (default: 1.0)"
  echo "  --freq_power=NUMBER              Power to raise inverse frequency to (default: 1.0)"
  echo "  --normalize_per_part             Z-score embedding/prob blocks independently before training"
  echo "  --output_dim_loss_weight=STRING  Per-dim loss weight on prob block: 'auto' (=emb_dim/output_dim), a float, or unset"
  echo "  --variance_filter_top_pct=NUMBER Restrict TRAINING to top-X fraction of samples by per-sample prob-block variance (eval still uses full data)"
  echo "  --variance_filter_mode=STRING    'linear' (default) or 'log' -- metric for variance filter"
  echo "  --cluster_strat_k=NUMBER         If > 0, K-means cluster trajectories and sample evenly from each cluster (applied after variance filter)"
  echo "  --cluster_strat_per_cluster=NUMBER Max samples per trajectory cluster (default 2000)"
  echo "  --cluster_strat_mode=STRING      'linear' (default) or 'log' -- feature space for trajectory clustering"
  echo "  --superimpose_loss_weight=NUMBER  Within-feature curve consistency loss weight (default 0; e.g. 0.1)"
  echo "  --help                           Display this help message"
  exit 1
}

seed=42
data_seed=0
num_epochs=1
use_delta_prob=False
use_delta_logprob=False
decoder_ortho_loss_weight=0.0
use_freq_weighting=False
freq_smoothing=1.0
freq_power=1.0
normalize_per_part=False
output_dim_loss_weight=""
variance_filter_top_pct=""
variance_filter_mode=""
cluster_strat_k=""
cluster_strat_per_cluster=""
cluster_strat_mode=""
superimpose_loss_weight=""
gradnorm_target_sup=""
gradnorm_target_ortho=""
gradnorm_every=""
gradnorm_ema=""
gradnorm_min_weight=""
gradnorm_max_weight=""

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
    --num_epochs=*)
      num_epochs="${1#*=}"
      shift
      ;;
    --use_delta_prob)
      use_delta_prob=True
      shift
      ;;
    --use_delta_logprob)
      use_delta_logprob=True
      shift
      ;;
    --decoder_ortho=*)
      decoder_ortho_loss_weight="${1#*=}"
      shift
      ;;
    --use_freq_weighting)
      use_freq_weighting=True
      shift
      ;;
    --freq_smoothing=*)
      freq_smoothing="${1#*=}"
      shift
      ;;
    --freq_power=*)
      freq_power="${1#*=}"
      shift
      ;;
    --normalize_per_part)
      normalize_per_part=True
      shift
      ;;
    --output_dim_loss_weight=*)
      output_dim_loss_weight="${1#*=}"
      shift
      ;;
    --variance_filter_top_pct=*)
      variance_filter_top_pct="${1#*=}"
      shift
      ;;
    --variance_filter_mode=*)
      variance_filter_mode="${1#*=}"
      shift
      ;;
    --cluster_strat_k=*)
      cluster_strat_k="${1#*=}"
      shift
      ;;
    --cluster_strat_per_cluster=*)
      cluster_strat_per_cluster="${1#*=}"
      shift
      ;;
    --cluster_strat_mode=*)
      cluster_strat_mode="${1#*=}"
      shift
      ;;
    --superimpose_loss_weight=*)
      superimpose_loss_weight="${1#*=}"
      shift
      ;;
    --gradnorm_target_sup=*)
      gradnorm_target_sup="${1#*=}"
      shift
      ;;
    --gradnorm_target_ortho=*)
      gradnorm_target_ortho="${1#*=}"
      shift
      ;;
    --gradnorm_every=*)
      gradnorm_every="${1#*=}"
      shift
      ;;
    --gradnorm_ema=*)
      gradnorm_ema="${1#*=}"
      shift
      ;;
    --gradnorm_min_weight=*)
      gradnorm_min_weight="${1#*=}"
      shift
      ;;
    --gradnorm_max_weight=*)
      gradnorm_max_weight="${1#*=}"
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
echo "Use delta prob: $use_delta_prob"
echo "Use delta logprob: $use_delta_logprob"
echo "Decoder ortho loss weight: $decoder_ortho_loss_weight"
if [ "$use_delta_prob" = "True" ] && [ "$use_delta_logprob" = "True" ]; then
  echo "Error: --use_delta_prob and --use_delta_logprob are mutually exclusive"
  exit 1
fi
echo "Frequency weighting: $use_freq_weighting"
if [ "$use_freq_weighting" = "True" ]; then
  echo "  Smoothing: $freq_smoothing"
  echo "  Power: $freq_power"
fi
echo "Normalize per part: $normalize_per_part"
echo "Output dim loss weight: ${output_dim_loss_weight:-<none>}"
echo "Variance filter top pct: ${variance_filter_top_pct:-<none>}"

export NCCL_P2P_DISABLE=1
module load cuda-12.4

GPU_ID=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
         awk '{print NR-1 ":" $1}' | sort -t: -k2 -nr | head -n1 | cut -d: -f1)
export CUDA_VISIBLE_DEVICES=$GPU_ID

# model_string=$(jq -r '.model_names | join("_")' "$args")
filename=$(basename "$exp_cfg" .json)
model_string="${filename#config_}"

# ofw=$(jq -r '.output_feature_weight' "$args")
# if [ -n "$prefix" ] ; then
#     sae_name_prefix="${prefix}_${model_string}_seed=${seed}_ofw=${ofw}"
# else
#     sae_name_prefix="${model_string}_seed=${seed}_ofw=${ofw}"
# fi

# dask spill dir
temp_dir="/scratch/$USER/tmp"

# temp_dir="/home/$USER/tmp"
# temp_dir="/mnt/labshare/nsrikant/bbox_outputs/tmp"
# temp_dir="/datadrive/nsrikant/tmp"
mkdir -p $temp_dir

cd sae
# python train_sae.py \
#     --args $exp_cfg \
#     --config_path $hp_cfg \
#     --data_shuffling_seed $data_seed \
#     --seed $seed \
#     --sae_name_prefix $sae_name_prefix \
#     --spill_dir $temp_dir \
#     --temp_dir $temp_dir \
#     --workers 8

train_extra_args=()
if [ -n "$output_dim_loss_weight" ]; then
    train_extra_args+=(--output_dim_loss_weight "$output_dim_loss_weight")
fi
if [ -n "$variance_filter_top_pct" ]; then
    train_extra_args+=(--variance_filter_top_pct "$variance_filter_top_pct")
fi
if [ -n "$variance_filter_mode" ]; then
    train_extra_args+=(--variance_filter_mode "$variance_filter_mode")
fi
if [ -n "$cluster_strat_k" ]; then
    train_extra_args+=(--cluster_strat_k "$cluster_strat_k")
fi
if [ -n "$cluster_strat_per_cluster" ]; then
    train_extra_args+=(--cluster_strat_per_cluster "$cluster_strat_per_cluster")
fi
if [ -n "$cluster_strat_mode" ]; then
    train_extra_args+=(--cluster_strat_mode "$cluster_strat_mode")
fi
if [ -n "$superimpose_loss_weight" ]; then
    train_extra_args+=(--superimpose_loss_weight "$superimpose_loss_weight")
fi
if [ -n "$gradnorm_target_sup" ]; then
    train_extra_args+=(--gradnorm_target_sup "$gradnorm_target_sup")
fi
if [ -n "$gradnorm_target_ortho" ]; then
    train_extra_args+=(--gradnorm_target_ortho "$gradnorm_target_ortho")
fi
if [ -n "$gradnorm_every" ]; then
    train_extra_args+=(--gradnorm_every "$gradnorm_every")
fi
if [ -n "$gradnorm_ema" ]; then
    train_extra_args+=(--gradnorm_ema "$gradnorm_ema")
fi
if [ -n "$gradnorm_min_weight" ]; then
    train_extra_args+=(--gradnorm_min_weight "$gradnorm_min_weight")
fi
if [ -n "$gradnorm_max_weight" ]; then
    train_extra_args+=(--gradnorm_max_weight "$gradnorm_max_weight")
fi

python train_sae.py \
    --args $exp_cfg \
    --config_path $hp_cfg \
    --data_shuffling_seed $data_seed \
    --seed $seed \
    --model_string $model_string \
    --spill_dir $temp_dir \
    --temp_dir $temp_dir \
    --workers 8 \
    --num_epochs $num_epochs \
    --use_delta_prob $use_delta_prob \
    --use_delta_logprob $use_delta_logprob \
    --decoder_ortho_loss_weight $decoder_ortho_loss_weight \
    --use_freq_weighting $use_freq_weighting \
    --freq_weight_smoothing $freq_smoothing \
    --freq_weight_power $freq_power \
    --normalize_per_part $normalize_per_part \
    "${train_extra_args[@]}"

if [ $? -ne 0 ]; then
    echo "Training failed with exit code $?. Terminating."
    rm -r $temp_dir
    echo "$temp_dir removed"
    exit 1
fi

echo "Training complete. Running eval."

eval_extra_args=()
if [ -n "$output_dim_loss_weight" ]; then
    eval_extra_args+=(--output_dim_loss_weight "$output_dim_loss_weight")
fi
if [ -n "$variance_filter_top_pct" ]; then
    eval_extra_args+=(--variance_filter_top_pct "$variance_filter_top_pct")
fi
if [ -n "$variance_filter_mode" ]; then
    eval_extra_args+=(--variance_filter_mode "$variance_filter_mode")
fi
if [ -n "$cluster_strat_k" ]; then
    eval_extra_args+=(--cluster_strat_k "$cluster_strat_k")
fi
if [ -n "$cluster_strat_per_cluster" ]; then
    eval_extra_args+=(--cluster_strat_per_cluster "$cluster_strat_per_cluster")
fi
if [ -n "$cluster_strat_mode" ]; then
    eval_extra_args+=(--cluster_strat_mode "$cluster_strat_mode")
fi
if [ -n "$superimpose_loss_weight" ]; then
    eval_extra_args+=(--superimpose_loss_weight "$superimpose_loss_weight")
fi

python eval_sae.py \
    --args $exp_cfg \
    --config_path $hp_cfg \
    --model_string $model_string \
    --spill_dir $temp_dir \
    --temp_dir $temp_dir \
    --workers 8 \
    --seed $seed \
    --use_delta_prob $use_delta_prob \
    --use_delta_logprob $use_delta_logprob \
    --decoder_ortho_loss_weight $decoder_ortho_loss_weight \
    --normalize_per_part $normalize_per_part \
    --save_activations True \
    "${eval_extra_args[@]}"


rm -r $temp_dir
echo "$temp_dir removed"
