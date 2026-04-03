#!/bin/bash

#SBATCH --job-name=oncox_seg

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_seg_%j.log

#SBATCH --time=08:00:00

#SBATCH --mem=32G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



set -e



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/segmentation



echo "========================================="

echo "Onco-GPT-X Module 2: Segmentation"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



echo ""

echo ">>> Model 1: U-Net + EfficientNet-B3 encoder (baseline)..."

python train_seg.py --model unet_efficientnet --epochs 100 --batch_size 8 --img_size 256 --lr 1e-4 --patience 15



echo ""

echo ">>> Model 2: PVTv2-B2 + UNet decoder..."

python train_seg.py --model pvtv2_unet --epochs 100 --batch_size 8 --img_size 256 --lr 5e-5 --patience 15



echo ""

echo ">>> Model 3: SwinV2-Tiny + UNet decoder..."

python train_seg.py --model swinv2_unet --epochs 100 --batch_size 8 --img_size 256 --lr 5e-5 --patience 15



echo ""

echo "========================================="

echo "Module 2 Complete — $(date)"

echo "========================================="

echo "Results:"

for m in unet_efficientnet pvtv2_unet swinv2_unet; do

    if [ -f "/scratch/patel.tis/OncoX/results/segmentation/${m}_log.csv" ]; then

        echo "  $m:"

        tail -1 /scratch/patel.tis/OncoX/results/segmentation/${m}_log.csv

    fi

done
