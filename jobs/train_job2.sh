#!/bin/bash

#SBATCH --job-name=oncox_job2

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1


#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/job2_%j.log



echo "========================================="

echo "Job 2: EfficientNet-B7, ConvNeXt-Base, MaxViT-Tiny"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_model_comparison.py --models efficientnet_b7 convnext_base maxvit_tiny_tf_224



echo "Done: $(date)"
