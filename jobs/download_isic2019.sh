#!/bin/bash

#SBATCH --job-name=oncox_dl2019

#SBATCH --partition=short

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G

#SBATCH --time=04:00:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/download_isic2019_%j.log



echo "========================================="

echo "ISIC 2019 download"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



DATA_DIR=/scratch/patel.tis/OncoX/data/raw/isic2019

META_DIR=/scratch/patel.tis/OncoX/data/metadata/isic2019

mkdir -p $DATA_DIR

mkdir -p $META_DIR



cd $DATA_DIR



echo ""

echo ">>> Step 1: Downloading metadata CSVs..."

wget -c --tries=3 --timeout=60 -O $META_DIR/ISIC_2019_Training_GroundTruth.csv https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv

wget -c --tries=3 --timeout=60 -O $META_DIR/ISIC_2019_Training_Metadata.csv https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv



echo ""

echo ">>> Step 2: Verifying metadata..."

ls -lh $META_DIR/

echo "GroundTruth preview:"

head -3 $META_DIR/ISIC_2019_Training_GroundTruth.csv

echo "Metadata preview:"

head -3 $META_DIR/ISIC_2019_Training_Metadata.csv



echo ""

echo ">>> Step 3: Downloading training images (~9GB zip)..."

wget -c --tries=5 --timeout=120 --progress=dot:giga -O $DATA_DIR/ISIC_2019_Training_Input.zip https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip



echo ""

echo ">>> Step 4: Verifying zip..."

ls -lh $DATA_DIR/ISIC_2019_Training_Input.zip

unzip -t $DATA_DIR/ISIC_2019_Training_Input.zip > /dev/null 2>&1 && echo "Zip OK" || echo "Zip CORRUPTED"



echo ""

echo ">>> Step 5: Extracting..."

cd $DATA_DIR

unzip -q ISIC_2019_Training_Input.zip



echo ""

echo ">>> Step 6: Verifying extraction..."

NUM=$(ls $DATA_DIR/ISIC_2019_Training_Input/ 2>/dev/null | wc -l)

echo "Total images extracted: $NUM"

echo "Expected: ~25,331"



if [ "$NUM" -gt 25000 ]; then

    echo "Extraction successful, deleting zip..."

    rm $DATA_DIR/ISIC_2019_Training_Input.zip

fi



du -sh $DATA_DIR/



echo "========================================="

echo "Done: $(date)"

echo "========================================="
