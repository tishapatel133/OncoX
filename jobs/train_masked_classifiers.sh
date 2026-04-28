#!/bin/bash

#SBATCH --job-name=oncox_msk

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_masked_%j.log

#SBATCH --time=08:00:00

#SBATCH --mem=32G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/classification



echo "========================================="

echo "Phase 2: Retrain on Masked Images"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



echo ""

echo ">>> Model 1: EfficientNet-CBAM + crop masking..."

python train_masked.py --model efficientnet_cbam --strategy hard_mask --epochs 25 --batch_size 32 --lr 3e-4 --pos_weight 8.0



echo ""

echo ">>> Model 2: SwinV2 + crop masking..."

python train_masked.py --model swinv2 --strategy hard_mask --epochs 25 --batch_size 24 --img_size 256 --lr 1e-4 --pos_weight 8.0



echo ""

echo "========================================="

echo "All Done: $(date)"

echo "========================================="

echo ""

echo "Results:"

for m in efficientnet_cbam swinv2; do

    f="/scratch/patel.tis/OncoX/results/classification/${m}_masked_crop_log.csv"

    if [ -f "$f" ]; then

        BEST=$(sort -t',' -k5 -rn "$f" | head -1 | cut -d',' -f5)

        echo "  $m masked best val AUC: $BEST"

    fi

done
