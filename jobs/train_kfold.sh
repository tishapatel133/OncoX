#!/bin/bash

#SBATCH --job-name=oncox_kfold

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_kfold_%j.log

#SBATCH --time=08:00:00

#SBATCH --mem=48G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX



echo "========================================="

echo "5-Fold CV — EfficientNet-B5 + Metadata"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "Started: $(date)"

echo "========================================="



python src/classification/train_kfold.py --model efficientnet_b3 --img_size 384 --epochs 20 --batch_size 16 --lr 2e-4 --focal_alpha 0.25 --focal_gamma 2.0 --label_smooth 0.0 --mixup_alpha 0.0 --n_folds 5 --patience 5 --workers 4 --tta_folds 5



echo ""

echo "========================================="

echo "Done: $(date)"

echo "========================================="

echo ""

echo "Summary:"

cat /scratch/patel.tis/OncoX/results/classification/efficientnet_b5_kfold5_meta1_384_summary.csv
