#!/bin/bash

#SBATCH --job-name=oncox_cls

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_cls_%j.log

#SBATCH --time=08:00:00

#SBATCH --mem=32G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



set -e



BASE=/scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd $BASE/src/classification



echo "========================================="

echo "Onco-GPT-X Module 1: Classification"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



echo ""

echo ">>> Training EfficientNet-B3 + CBAM..."

python train_ensemble.py --model efficientnet_cbam --epochs 25 --batch_size 32 --img_size 224 --lr 3e-4 --pos_weight 8.0



echo ""

echo ">>> Training SwinV2-Tiny..."

python train_ensemble.py --model swinv2 --epochs 25 --batch_size 24 --img_size 256 --lr 1e-4 --pos_weight 8.0



echo ""

echo ">>> Training PVTv2-B2..."

python train_ensemble.py --model pvtv2 --epochs 25 --batch_size 32 --img_size 224 --lr 1e-4 --pos_weight 8.0



echo ""

echo ">>> Training Meta-Ensemble..."

python train_ensemble.py --model meta --batch_size 32



echo ""

echo "========================================="

echo "Module 1 Complete — $(date)"

echo "========================================="
