#!/bin/bash

#SBATCH --job-name=oncox_modelcomp

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=48G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/model_comparison_%j.log



echo "========================================="

echo "Model comparison: B4 B5 B6"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_model_comparison.py



echo "Done: $(date)"
