#!/bin/bash

#SBATCH --job-name=oncox_b7

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/job2b_%j.log



echo "========================================="

echo "Job 2b: EfficientNet-B7 resume from fold 3"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_model_comparison.py --models efficientnet_b7 --start_fold 2



echo "Done: $(date)"
