#!/bin/bash

#SBATCH --job-name=oncox_twostage

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=08:00:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/two_stage_%j.log



echo "========================================="

echo "Exp 1: Two-stage training"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_two_stage.py --img_size 384 --stage1_epochs 10 --stage2_epochs 15 --batch_size 16 --lr 2e-4 --focal_alpha 0.25 --focal_gamma 2.0 --patience 7 --workers 0



echo "Done: $(date)"
