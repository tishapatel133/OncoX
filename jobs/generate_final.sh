#!/bin/bash

#SBATCH --job-name=oncox_gen

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/generate_%j.log

#SBATCH --time=02:00:00

#SBATCH --mem=16G

#SBATCH --cpus-per-task=4

#SBATCH --gres=gpu:1

#SBATCH --partition=gpu



source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

cd /scratch/patel.tis/OncoX/src/diffusion



echo "========================================="

echo "Generating counterfactuals + samples"

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

echo "========================================="



echo ">>> SDEdit counterfactuals..."

python sdedit_generate.py --n_samples 5 --strength 0.4



echo ">>> Class-conditional samples..."

python generate.py



echo "Done! $(date)"
