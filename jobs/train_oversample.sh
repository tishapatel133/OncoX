#!/bin/bash

#SBATCH --job-name=oncox_oversample

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/oversample_%j.log



echo "========================================="

echo "Exp 4: Oversampling + weighted sampler"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_oversample.py



echo "Done: $(date)"
