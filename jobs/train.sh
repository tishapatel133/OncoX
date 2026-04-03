#!/bin/bash
#SBATCH --job-name=oncox_train
#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_%j.log
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -e

BASE=/scratch/patel.tis/OncoX

echo "========================================="
echo "OncoX Training Pipeline — $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "========================================="

# Activate environment
source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd $BASE

# ── Step 1: Verify download ──
echo ""
echo "Step 1: Verifying downloaded data..."
IMG_COUNT=$(find $BASE/data/raw/isic2020 -name "*.jpg" | wc -l)
echo "Found $IMG_COUNT images"

if [ "$IMG_COUNT" -lt 1000 ]; then
    echo "ERROR: Expected 30000+ images but found $IMG_COUNT"
    echo "Download may have failed. Contents of data/raw/isic2020:"
    ls -R $BASE/data/raw/isic2020/ | head -30
    exit 1
fi
echo "✅ Data verified"

# ── Step 2: Preprocess (skip if already done) ──
echo ""
echo "Step 2: Preprocessing..."
if [ -f "$BASE/data/metadata/train.csv" ] && [ -f "$BASE/data/metadata/img_dir.txt" ]; then
    echo "Split CSVs already exist — skipping preprocessing"
    echo "Image dir: $(cat $BASE/data/metadata/img_dir.txt)"
else
    python src/data/preprocess.py
fi
echo "✅ Preprocessing complete"

# ── Step 3: Train ──
echo ""
echo "Step 3: Training..."
python src/training/train.py \
    --epochs 25 \
    --batch_size 32 \
    --img_size 224 \
    --lr 3e-4 \
    --model efficientnet_b3 \
    --pos_weight 8.0 \
    --grad_clip 1.0 \
    --num_workers 4

echo ""
echo "========================================="
echo "Pipeline complete — $(date)"
echo "========================================="