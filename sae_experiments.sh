

sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n_moreearly_blimp_weight0_6.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50.json


sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n_moreearly_blimp.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=75.json

sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n_moreearly_blimp.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=25.json

# sbatch scripts/sae/sae_pipeline.sh \
#     --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n_moreearly_blimp.json \
#     --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=50_topk.json

sbatch scripts/sae/sae_pipeline.sh \
    --exp_cfg=/home/nsrikant/BehaviorBoxNew/scripts/sae/experiment_configs/config_n_moreearly_blimp_weight0_6.json \
    --hp_cfg=/home/nsrikant/BehaviorBoxNew/sae/hyperparam_configs/N=3000_k=75.json