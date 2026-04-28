#!/bin/bash

#SBATCH --job-name=oncox_best

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_best_%j.log

#SBATCH --time=08:00:00

#SBATCH --mem=40G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX



echo "========================================="

echo "Best Single Model Pipeline"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



# ── Step 1: Hair removal + CLAHE preprocessing ──

echo ""

echo ">>> Step 1: DullRazor + CLAHE preprocessing..."

python src/data/preprocess_hair.py



# Verify clean images exist

CLEAN_COUNT=$(ls /scratch/patel.tis/OncoX/data/raw/isic2020_clean/*.jpg 2>/dev/null | wc -l)

echo "Clean images ready: $CLEAN_COUNT"



if [ "$CLEAN_COUNT" -lt 1000 ]; then

    echo "ERROR: Too few clean images — check preprocessing"

    exit 1

fi



# ── Step 2: Train EfficientNet-B3 (clean images, Focal Loss, 384x384) ──

echo ""

echo ">>> Step 2: Train EfficientNet-B3 — clean + Focal + 384px + TTA..."



python src/classification/train_best.py --model efficientnet_b3 --img_size 384 --epochs 25 --batch_size 16 --lr 2e-4 --focal_alpha 0.25 --focal_gamma 2.0 --patience 7 --workers 4 --use_clean --tta

echo ""

echo "========================================="

echo "Done: $(date)"

echo "========================================="

echo ""

echo "Results:"

LOGF="/scratch/patel.tis/OncoX/results/classification/efficientnet_b3_clean_384_log.csv"

if [ -f "$LOGF" ]; then

    BEST=$(sort -t',' -k5 -rn "$LOGF" | head -1 | cut -d',' -f5)

    echo "  Best Val AUC: $BEST"

fi
