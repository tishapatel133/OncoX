#!/bin/bash

#SBATCH --job-name=oncox_xai

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/xai_%j.log

#SBATCH --time=02:00:00

#SBATCH --mem=16G

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



set -e



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/xai



echo "========================================="

echo "Onco-GPT-X Module 3: Explainable AI"

echo "Started: $(date)"

echo "========================================="



echo ""

echo ">>> Grad-CAM for EfficientNet-B3 + CBAM..."

python gradcam.py --model efficientnet_cbam --n_samples 20



echo ""

echo ">>> Grad-CAM for SwinV2..."

python gradcam.py --model swinv2 --n_samples 20



echo ""

echo ">>> Grad-CAM for PVTv2..."

python gradcam.py --model pvtv2 --n_samples 20



echo ""

echo "========================================="

echo "Module 3 Complete — $(date)"

echo "========================================="

echo "Output: /scratch/patel.tis/OncoX/results/xai/"

ls -la /scratch/patel.tis/OncoX/results/xai/
