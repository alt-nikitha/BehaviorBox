# final and mid

sbatch scripts/data_generation/get_input_features.sh \
    --data=/home/nsrikant/BehaviorBoxNew/data/validation_split_1000.jsonl \
    --output_dir=/home/nsrikant/BehaviorBoxNew/output/validation_split_1000 \
    --batch_size=5

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/validation_split_1000.jsonl \
    --output_dir=/home/nsrikant/BehaviorBoxNew/output/validation_split_1000 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step70000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/validation_split_1000.jsonl \
    --output_dir=/home/nsrikant/BehaviorBoxNew/output/validation_split_1000 \
    --revision=step70000 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 

sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json

sbatch scripts/analysis/label_features.sh --sae_dir=/home/nsrikant/BehaviorBoxNew/sae_outputs/mid_final/_seed=42_ofw=_N=3000_k=50_lp=None --labeling_model="neulab/claude-sonnet-4-20250514"


# 100000 and 10000

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step100000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/validation_split_1000.jsonl \
    --output_dir=/home/nsrikant/BehaviorBoxNew/output/validation_split_1000 \
    --revision=step100000 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step10000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/validation_split_1000.jsonl \
    --output_dir=/home/nsrikant/BehaviorBoxNew/output/validation_split_1000 \
    --revision=step10000 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 

# 100000 and final

sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_100000_final.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json

sbatch scripts/analysis/label_features.sh --sae_dir=/home/nsrikant/BehaviorBoxNew/sae_outputs/100000_final/_seed=42_ofw=_N=3000_k=50_lp=None --labeling_model="neulab/claude-sonnet-4-20250514"


# 10000 and final
sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_10000_final.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json

sbatch scripts/analysis/label_features.sh --sae_dir=/home/nsrikant/BehaviorBoxNew/sae_outputs/10000_final/_seed=42_ofw=_N=3000_k=50_lp=None --labeling_model="neulab/claude-sonnet-4-20250514"

# 10000 and mid
sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_10000_mid.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json

sbatch scripts/analysis/label_features.sh --sae_dir=/home/nsrikant/BehaviorBoxNew/sae_outputs/10000_mid/_seed=42_ofw=_N=3000_k=50_lp=None --labeling_model="neulab/claude-sonnet-4-20250514"


# mid and 100000
sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_mid_100000.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json

sbatch scripts/analysis/label_features.sh --sae_dir=/home/nsrikant/BehaviorBoxNew/sae_outputs/mid_100000/_seed=42_ofw=_N=3000_k=50_lp=None --labeling_model="neulab/claude-sonnet-4-20250514"

