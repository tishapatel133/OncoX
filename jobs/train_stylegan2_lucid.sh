#!/bin/bash

#SBATCH --job-name=oncox_sg2l

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=48G

#SBATCH --time=07:59:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/sg2lucid_%j.log



echo "========================================="

echo "lucidrains StyleGAN2 melanoma training"

echo "Job ID: $SLURM_JOB_ID"

echo "Start: $(date)"

echo "========================================="



cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



DATA_DIR=/scratch/patel.tis/OncoX/data/gan_training/melanoma_raw

RESULTS_DIR=/scratch/patel.tis/OncoX/models/gan_results

MODELS_DIR=/scratch/patel.tis/OncoX/models/gan_checkpoints_lucid



mkdir -p $RESULTS_DIR $MODELS_DIR



# lucidrains auto-resumes from last checkpoint on re-run

stylegan2_pytorch \

    --data $DATA_DIR \

    --name melanoma \

    --results_dir $RESULTS_DIR \

    --models_dir $MODELS_DIR \

    --image-size 256 \

    --batch-size 8 \

    --gradient-accumulate-every 2 \

    --network-capacity 16 \

    --num-train-steps 150000 \

    --aug-prob 0.25 \

    --aug-types '[translation,cutout,color]' \

    --attn-layers '[1,2]' \

    --save-every 2000 \

    --evaluate-every 2000



echo "Done: $(date)"
