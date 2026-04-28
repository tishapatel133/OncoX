#!/bin/bash

#SBATCH --job-name=oncox_job3

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1


#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/job3_%j.log



echo "========================================="

echo "Job 3: SwinV2-Small, SwinV2-Base, SwinV2-CR-Tiny-384"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_model_comparison.py --models swinv2_small_window8_256 swinv2_base_window8_256 swinv2_cr_tiny_384



echo "Done: $(date)"
