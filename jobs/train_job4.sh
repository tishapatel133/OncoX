#!/bin/bash

#SBATCH --job-name=oncox_job4

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1


#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/job4_%j.log



echo "========================================="

echo "Job 4: PVTv2-B3, PVTv2-B4, PVTv2-B5"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



python src/classification/train_model_comparison.py --models pvt_v2_b3 pvt_v2_b4 pvt_v2_b5



echo "Done: $(date)"
