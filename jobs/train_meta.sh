#!/bin/bash

#SBATCH --job-name=oncox_meta

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/train_meta_%j.log

#SBATCH --time=01:00:00

#SBATCH --mem=16G

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/classification

python train_ensemble.py --model meta --batch_size 32
