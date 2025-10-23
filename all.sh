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


# 1000

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step1000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/validation_split_1000.jsonl \
    --output_dir=/home/nsrikant/BehaviorBoxNew/output/validation_split_1000 \
    --revision=step1000 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 

# 1000 and final

sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_1000_final.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json

sbatch scripts/analysis/label_features.sh --sae_dir=/home/nsrikant/BehaviorBoxNew/sae_outputs/1000_final/_seed=42_ofw=_N=3000_k=50_lp=None --labeling_model="neulab/claude-sonnet-4-20250514"



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

# n models


sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json

sbatch scripts/analysis/label_features.sh --sae_dir=/mnt/labshare/nsrikant/bbox_outputs/sae_outputs/n_comparison/_seed=42_ofw=_N=3000_k=50_lp=None --labeling_model="neulab/claude-sonnet-4-20250514"



# earlier models more



 
bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step1 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step1 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 


bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step2 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step2 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 


bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step4 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step4 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 


bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step8 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step8 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step16 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step16 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step32 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step32 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm 

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step64 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step64 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step128 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step128 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step256 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step256 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm



bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step512 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step512 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm


bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step5000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/mnt/labshare/nsrikant/data/validation_split_1000.jsonl \
    --output_dir=/mnt/labshare/nsrikant/bbox_outputs/output/validation_split_1000 \
    --revision=step5000 \
    --batch_size=100 \
    --async_limiter=20 \
    --slurm

sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n_moreearly.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json

sbatch scripts/analysis/label_features.sh --sae_dir=/mnt/labshare/nsrikant/bbox_outputs/sae_outputs/n_moreearly/n_moreearly_seed=42_ofw=_N=3000_k=50_lp=None --labeling_model="neulab/claude-sonnet-4-20250514"
