#!/bin/bash

#SBATCH --job-name=download_all

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/download_all_%j.log

#SBATCH --time=02:00:00

#SBATCH --mem=4G

#SBATCH --cpus-per-task=2

#SBATCH --partition=short



set -e



BASE=/scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



echo "========================================="

echo "Downloading all datasets — $(date)"

echo "========================================="



# ── 1. HAM10000 ──

echo ""

echo ">>> Downloading HAM10000..."

mkdir -p $BASE/data/raw/ham10000

cd $BASE/data/raw/ham10000



if [ $(find . -name "*.jpg" 2>/dev/null | wc -l) -gt 1000 ]; then

    echo "HAM10000 already downloaded, skipping"

else

    kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p . --unzip

    echo "HAM10000 files:"

    ls -lh

    echo "Image count: $(find . -name '*.jpg' | wc -l)"

fi

echo "✅ HAM10000 done"



# ── 2. ISIC 2018 Segmentation ──

echo ""

echo ">>> Downloading ISIC 2018 Segmentation..."

mkdir -p $BASE/data/raw/isic2018_seg

cd $BASE/data/raw/isic2018_seg



if [ $(find . -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l) -gt 500 ]; then

    echo "ISIC 2018 seg already downloaded, skipping"

else

    kaggle datasets download -d tschandl/isic2018-challenge-task1-data-segmentation -p . --unzip

    echo "ISIC 2018 seg files:"

    ls -lh

    echo "Image count: $(find . -name '*.jpg' | wc -l)"

    echo "Mask count: $(find . -name '*.png' | wc -l)"

fi

echo "✅ ISIC 2018 Segmentation done"



echo ""

echo "========================================="

echo "All downloads complete — $(date)"

echo "========================================="

echo ""

echo "Summary:"

echo "  HAM10000 images: $(find $BASE/data/raw/ham10000 -name '*.jpg' | wc -l)"

echo "  ISIC 2018 images: $(find $BASE/data/raw/isic2018_seg -name '*.jpg' | wc -l)"

echo "  ISIC 2018 masks:  $(find $BASE/data/raw/isic2018_seg -name '*.png' | wc -l)"

echo "  SIIM-ISIC 2020:   $(find $BASE/data/raw/isic2020 -name '*.jpg' | wc -l)"

echo ""

du -sh $BASE/data/raw/*/
