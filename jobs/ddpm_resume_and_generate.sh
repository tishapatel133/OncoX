#!/bin/bash

#SBATCH --job-name=oncox_ddpm2

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/ddpm_resume_%j.log

#SBATCH --time=08:00:00

#SBATCH --mem=32G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



set -e



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/diffusion



echo "========================================="

echo "Onco-GPT-X: DDPM Resume Training + SDEdit"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



echo ""

echo ">>> Phase 1: Resume DDPM training to 800 epochs..."

python resume_ddpm.py --total_epochs 800 --batch_size 16 --img_size 128 --lr 1e-4 --save_every 100 --sample_every 100



echo ""

echo ">>> Phase 2: Generate SDEdit counterfactuals..."

python sdedit_generate.py --n_samples 5 --strength 0.4



echo ""

echo ">>> Phase 3: Generate class-conditional samples..."

python generate.py



echo ""

echo "========================================="

echo "DDPM Complete — $(date)"

echo "========================================="

echo "Counterfactuals: /scratch/patel.tis/OncoX/results/diffusion/counterfactuals/"

echo "Samples: /scratch/patel.tis/OncoX/results/diffusion/samples/"
