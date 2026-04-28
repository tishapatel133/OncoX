#!/bin/bash

#SBATCH --job-name=oncox_hybrid

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=06:00:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/hybrid_%j.log



echo "========================================="

echo "Hybrid CNN+Transformer (85M params)"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_hybrid.py



echo "Done: $(date)"
