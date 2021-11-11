#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=multi_train_ffhq_log
#SBATCH --output=train_multi_ffhq-%j.out
#SBATCH --error=train_multi_ffhq-%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=10gb

PYTHONPATH=. python ./scripts/multi_train_lsun.py
