#!/bin/bash
#SBATCH --job-name=lm_eval
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=general

# Create logs directory if it doesn't exist
mkdir -p logs


# Activate your environment (adjust for your setup)
conda init
conda activate bbox_env

python checkpoints_eval.py