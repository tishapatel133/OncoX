#!/bin/bash

#SBATCH --job-name=oncox_prep2019

#SBATCH --partition=short

#SBATCH --cpus-per-task=16

#SBATCH --mem=32G

#SBATCH --time=02:00:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/preprocess_isic2019_%j.log



echo "========================================="

echo "ISIC 2019 preprocessing (DullRazor + CLAHE)"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/data/preprocess_hair_isic2019.py



echo "Done: $(date)"
