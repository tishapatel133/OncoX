#!/bin/bash

#SBATCH --job-name=oncox_b3se_5fold

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/b3_se_5fold_%j.log



echo "========================================="

echo "Final: EfficientNet-B3 + SE, full 5-fold CV"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_b3_se_5fold.py



echo "Done: $(date)"
