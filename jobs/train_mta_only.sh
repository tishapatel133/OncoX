#!/bin/bash

#SBATCH --job-name=oncox_mta

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --time=03:00:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/mta_%j.log



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

python src/classification/train_attention_comparison.py --experiments mta
