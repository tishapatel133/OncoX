#!/bin/bash

#SBATCH --job-name=oncox_merged

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/b3_se_merged_%j.log



echo "========================================="

echo "EfficientNet-B3 + SE on merged 2019+2020 (51K images, 4959 mel)"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_b3_se_merged.py



echo "Done: $(date)"
