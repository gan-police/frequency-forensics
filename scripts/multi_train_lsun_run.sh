#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=multi_celeba_train_cnn
#SBATCH --output=multi_celeba_train_cnn-%j.out
#SBATCH --error=multi_celeba_train_cnn-%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=200gb

source activate ~/env/idp38

PYTHONPATH=. python ./scripts/multi_train_lsun.py

wait
