#!/bin/bash

#SBATCH --job-name=oncox_smote

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=08:00:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/smote_%j.log



echo "========================================="

echo "Exp 3: SMOTE in embedding space"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



pip install imbalanced-learn --quiet



python src/classification/train_smote.py



echo "Done: $(date)"
