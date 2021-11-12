#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=lsun_log_sym4_boundary
#SBATCH --output=prepare_dataset_lsun_log_sym4_boundary-%j.out
#SBATCH --error=prepare_dataset_lsun_log_sym4_boundary-%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --mem=90gb

python -m freqdect.prepare_dataset \
  /nvme/mwolter/lsun/lsun_bedroom_200k_png \
  --train-size 100000 --test-size 30000 --val-size 20000 \
  --log-packets --wavelet sym4 --boundary boundary 
