


#!/bin/bash
# Run all Pythia-160m checkpoints on both BLiMP and Anthropic HH datasets
# Pattern: checkpoint1 dataset1 dataset2, checkpoint2 dataset1 dataset2, etc.

# ============================================================================
# Step 1
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step1 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step1 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm 



# ============================================================================
# Step 2
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step2 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step2 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 4
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step4 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step4 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 8
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step8 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step8 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 16
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step16 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step16 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 32
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step32 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step32 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 64
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step64 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step64 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 128
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step128 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step128 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 256
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step256 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step256 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 512
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step512 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step512 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 1000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step1000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step1000 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm


# ============================================================================
# Step 10000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step10000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step10000 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 70000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step70000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step70000 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 100000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m-step100000 \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=step100000 \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Final checkpoint (no revision specified)
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=pythia-160m \
    --model_id=EleutherAI/pythia-160m \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm












# ================ old code ================


# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step1 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/blimp_samples.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/blimp_samples \
#     --revision=step1 \
#     --batch_size=200 \
#     --async_limiter=40 \
#     --slurm 


# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step1 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/anthropic_hh_samples.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/anthropic_hh_samples \
#     --revision=step1 \
#     --batch_size=200 \
#     --async_limiter=40 \
#     --slurm


# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step2 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step2 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm 


# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step4 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step4 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm 


# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step8 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step8 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm 

# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step16 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step16 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm 

# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step32 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step32 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm 

# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step64 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step64 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm

# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step128 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step128 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm

# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step256 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step256 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm



# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step512 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step512 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm


# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step1000 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step1000 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm 


# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step10000 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step10000 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm

# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step70000 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step70000 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm

# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m-step100000 \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --revision=step100000 \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm


# bash scripts/data_generation/get_output_features.sh \
#     --model_name=pythia-160m \
#     --model_id=EleutherAI/pythia-160m \
#     --data=/home/nsrikant/BehaviorBoxNew/data/positive_negative_tasks.jsonl \
#     --output_dir=/home/nsrikant/bbox_outputs/output/positive_negative_tasks \
#     --batch_size=100 \
#     --async_limiter=20 \
#     --slurm

# # "pythia-160m-step10000",
# #         "pythia-160m-step70000",
# #         "pythia-160m-step100000",

# # 13000
# # 39000
# # 65000
# # 91000
# # 117000
# # 143000
