#!/bin/bash

#SBATCH --job-name=oncox_vit_st

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/vit_staged_%j.log



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

python src/classification/train_transformer_staged.py --model vit_small

echo "Done: $(date)"
