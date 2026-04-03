#!/bin/bash

# Launch the full overnight pipeline:

#   Job 1: Download ISIC dataset (short partition, no GPU)

#   Job 2: Preprocess + Train (gpu partition, waits for Job 1)



set -e



echo "Submitting download job..."

DOWNLOAD_ID=$(sbatch --parsable /scratch/patel.tis/OncoX/jobs/download_isic.sh)

echo "  Download job ID: $DOWNLOAD_ID"



echo "Submitting training job (will wait for download)..."

TRAIN_ID=$(sbatch --parsable --dependency=afterok:$DOWNLOAD_ID /scratch/patel.tis/OncoX/jobs/train.sh)

echo "  Training job ID: $TRAIN_ID"



echo ""

echo "========================================="

echo "Both jobs submitted!"

echo "  Download: $DOWNLOAD_ID (short partition)"

echo "  Training: $TRAIN_ID (gpu partition, starts after download)"

echo "========================================="

echo ""

echo "Monitor with:"

echo "  squeue -u patel.tis"

echo ""

echo "Check results in the morning:"

echo "  cat /scratch/patel.tis/OncoX/results/metrics/training_log.csv"

echo "  tail -50 /scratch/patel.tis/OncoX/models/logs/train_${TRAIN_ID}.log"
