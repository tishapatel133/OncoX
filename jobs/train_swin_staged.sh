#!/bin/bash

#SBATCH --job-name=oncox_swin_st

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/swin_staged_%j.log



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

python src/classification/train_transformer_staged.py --model swinv2_small

echo "Done: $(date)"
