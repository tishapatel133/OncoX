#!/bin/bash

#SBATCH --job-name=oncox_gan_gen

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=24G

#SBATCH --time=00:30:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/gan_gen_%j.log



echo "Start: $(date)"

cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



mkdir -p /scratch/patel.tis/OncoX/data/gan_output/samples_at_36k



stylegan2_pytorch --generate --num_image_tiles 8 --models_dir /scratch/patel.tis/OncoX/models --name default --load-from 36 --results_dir /scratch/patel.tis/OncoX/data/gan_output/samples_at_36k



echo "Done: $(date)"

ls -lh /scratch/patel.tis/OncoX/data/gan_output/samples_at_36k/
