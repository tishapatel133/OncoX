#!/bin/bash

#SBATCH --job-name=oncox_sg3

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=64G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/sg3_%j.log



echo "========================================="

echo "StyleGAN3 melanoma training"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



# Load CUDA module and set CUDA_HOME for StyleGAN3 kernel compilation

module load cuda/12.8.0

export CUDA_HOME=/shared/centos7/cuda/12.8.0

export PATH=$CUDA_HOME/bin:$PATH

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH



echo "CUDA_HOME: $CUDA_HOME"

which nvcc

nvcc --version



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



DATASET=/scratch/patel.tis/OncoX/data/gan_training/melanoma_256.zip

OUTDIR=/scratch/patel.tis/OncoX/models/gan_checkpoints

SG3DIR=/scratch/patel.tis/OncoX/external/stylegan3



mkdir -p $OUTDIR

cd $SG3DIR



LATEST_PKL=$(ls -t $OUTDIR/*/network-snapshot-*.pkl 2>/dev/null | head -1)



if [ -z "$LATEST_PKL" ]; then

    echo ">>> No existing checkpoint found, starting fresh training"

    RESUME_ARG=""

else

    echo ">>> Resuming from: $LATEST_PKL"

    RESUME_ARG="--resume=$LATEST_PKL"

fi



python train.py --outdir=$OUTDIR --cfg=stylegan3-r --data=$DATASET --gpus=1 --batch=16 --gamma=2 --mirror=1 --kimg=1000 --snap=5 --metrics=fid50k_full $RESUME_ARG



echo "Done: $(date)"
