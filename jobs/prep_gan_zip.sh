#!/bin/bash

#SBATCH --job-name=oncox_gan_prep

#SBATCH --partition=short

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=01:00:00

#SBATCH --output=/scratch/patel.tis/OncoX/models/logs/gan_prep_%j.log



echo "Start: $(date)"

cd /scratch/patel.tis/OncoX

source /scratch/patel.tis/skin_cancer/dsnet_env/bin/activate



# Remove partial zip if exists

rm -f /scratch/patel.tis/OncoX/data/gan_training/melanoma_256.zip



cd /scratch/patel.tis/OncoX/external/stylegan3

python dataset_tool.py --source=/scratch/patel.tis/OncoX/data/gan_training/melanoma_raw --dest=/scratch/patel.tis/OncoX/data/gan_training/melanoma_256.zip --resolution=256x256



echo "Done: $(date)"

ls -lh /scratch/patel.tis/OncoX/data/gan_training/
