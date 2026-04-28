#!/bin/bash

#SBATCH --job-name=oncox_eval

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=48G

#SBATCH --time=02:00:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/evaluate_%j.log



echo "========================================="

echo "Full evaluation — all checkpoints"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/evaluate_best.py



echo "Done: $(date)"
