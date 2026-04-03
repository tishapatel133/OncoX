#!/bin/bash

#SBATCH --job-name=oncox_sw2

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_seg_swinv2_%j.log

#SBATCH --time=04:00:00

#SBATCH --mem=32G

#SBATCH --cpus-per-task=8

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/segmentation

python train_seg.py --model swinv2_unet --epochs 100 --batch_size 8 --img_size 256 --lr 5e-5 --patience 15
