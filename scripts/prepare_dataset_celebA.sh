#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=celeba_prepare_dataset
#SBATCH --output=prepare_dataset-%j.out
#SBATCH --error=prepare_dataset-%j.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=16


DATASETS="/home/ndv/projects/wavelets/datasets_moritz"
DATESET="celeba_align_png_cropped"

module load PyTorch
module load Pillow
module load Anaconda3
module load Python

pip install -q -e .

python -m freqdect.prepare_dataset_batched "$DATASET" \
  --train-size 100000 \
  --test-size 30000 \
  --val-size 20000 \
  -p
