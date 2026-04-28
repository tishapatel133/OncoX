#!/bin/bash

#SBATCH --job-name=oncox_tf_eval

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=01:30:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/tf_eval_%j.log



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate

python src/classification/evaluate_transformer_test.py

echo "Done: $(date)"
