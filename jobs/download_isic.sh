#!/bin/bash
#SBATCH --job-name=isic_download
#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/download_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=short

set -e

BASE=/scratch/patel.tis/OncoX
DATA=$BASE/data/raw/isic2020

echo "========================================="
echo "ISIC 2020 Download via curl — $(date)"
echo "========================================="

cd $DATA

# Read Kaggle credentials from kaggle.json
KAGGLE_USER=$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")
KAGGLE_KEY=$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['key'])")

echo "Kaggle user: $KAGGLE_USER"

# CSVs already present — skip
echo "✅ train.csv already present"
echo "✅ test.csv already present"

# Download full dataset using curl (streams to disk, no memory spike)
echo ""
echo "Downloading full competition zip via curl..."
echo "This is ~36GB — will take a while..."
echo ""

curl -L -o siim-isic.zip \
    -u "$KAGGLE_USER:$KAGGLE_KEY" \
    "https://www.kaggle.com/api/v1/competitions/data/download-all/siim-isic-melanoma-classification"

FILESIZE=$(stat -c%s "siim-isic.zip" 2>/dev/null || echo "0")
echo "Downloaded file size: $FILESIZE bytes"

if [ "$FILESIZE" -lt 1000000 ]; then
    echo "ERROR: Download too small — likely failed. Contents:"
    head -c 500 siim-isic.zip
    exit 1
fi

echo "✅ Download complete"

echo ""
echo "Unzipping (this takes a while for 36GB)..."
unzip -q -o siim-isic.zip
rm siim-isic.zip
echo "✅ Unzip complete"

echo ""
echo "========================================="
echo "Download Summary — $(date)"
echo "========================================="
echo "Directory structure:"
find . -type d | head -20
echo ""
echo "Image counts:"
echo "  Train JPEGs: $(find . -path '*/train/*.jpg' | wc -l)"
echo "  Test JPEGs:  $(find . -path '*/test/*.jpg' | wc -l)"
echo ""
echo "Total disk usage:"
du -sh $DATA