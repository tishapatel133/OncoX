#!/bin/bash

#SBATCH --job-name=oncox_inf

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/masked_inference_%j.log

#SBATCH --time=01:00:00

#SBATCH --mem=32G

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/classification



echo "========================================="

echo "Phase 1: Masked Inference Test"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



python masked_inference.py



echo "========================================="

echo "Done: $(date)"

echo "========================================="
