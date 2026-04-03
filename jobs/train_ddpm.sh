#!/bin/bash

#SBATCH --job-name=oncox_ddpm

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_ddpm_%j.log

#SBATCH --time=08:00:00

#SBATCH --mem=32G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



set -e



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/diffusion



echo "========================================="

echo "Onco-GPT-X Module 4: DDPM Training"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



python train_ddpm.py --epochs 200 --batch_size 16 --img_size 128 --lr 2e-4 --save_every 25 --sample_every 25



echo ""

echo ">>> Generating final counterfactual images..."

python generate.py



echo ""

echo "========================================="

echo "Module 4 Complete — $(date)"

echo "========================================="
