#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=celebA_log_db4_boundary
#SBATCH --output=prepare_dataset_celebA_log_db4_boundary-%j.out
#SBATCH --error=prepare_dataset_celebA_log_db4_boundary-%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --mem=90gb

python -m freqdect.prepare_dataset \
  /nvme/mwolter/celeba/celeba_align_png_cropped \
  --train-size 100000 --test-size 30000 --val-size 20000 \
  --log-packets --wavelet db4 --boundary boundary 
  