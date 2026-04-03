#!/bin/bash

#SBATCH --job-name=oncox_ft

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/finetune_%j.log

#SBATCH --time=08:00:00

#SBATCH --mem=32G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



set -e



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/diffusion



echo "========================================="

echo "Onco-GPT-X: Fine-tune Pretrained DDPM"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



python finetune_pretrained.py --epochs 200 --batch_size 8 --img_size 128 --lr 1e-4 --save_every 50 --sample_every 50



echo ""

echo "========================================="

echo "Fine-tuning Complete — $(date)"

echo "========================================="
