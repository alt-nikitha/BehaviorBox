


#!/bin/bash
# Run all olmo2B checkpoints on BLiMP datasets



# ============================================================================
# Step 0
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step0-tokens0B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step0-tokens0B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm 



# ============================================================================
# Step 1000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step1000-tokens3B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step1000-tokens3B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm 



# ============================================================================
# Step 2000
# ============================================================================


bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step2000-tokens5B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step2000-tokens5B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm


# ============================================================================
# Step 3000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step3000-tokens7B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step3000-tokens7B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 4000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step4000-tokens9B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step4000-tokens9B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm



# ============================================================================
# Step 5000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step5000-tokens11B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step5000-tokens11B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm


# ============================================================================
# Step 6000
# ============================================================================

bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step6000-tokens13B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step6000-tokens13B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm


# ============================================================================
# Step 7000
# ============================================================================


bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step7000-tokens15B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step7000-tokens15B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm


# ============================================================================
# Step 8000
# ============================================================================


bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step8000-tokens17B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step8000-tokens17B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm




# ============================================================================
# Step 9000
# ============================================================================


bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step9000-tokens19B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step9000-tokens19B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm


# ============================================================================
# Step 10000
# ============================================================================




bash scripts/data_generation/get_output_features.sh \
    --model_name=OLMo2-step10000-tokens21B \
    --model_id=allenai/OLMo-2-0425-1B-early-training \
    --data=/home/nsrikant/BehaviorBoxNew/data/blimp_full.jsonl \
    --output_dir=/home/nsrikant/bbox_outputs/output/blimp_full \
    --revision=stage1-step10000-tokens21B \
    --batch_size=1000 \
    --async_limiter=1000 \
    --slurm






