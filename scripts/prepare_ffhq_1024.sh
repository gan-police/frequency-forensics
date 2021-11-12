#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=prepare_dataset
#SBATCH --output=prepare_dataset-%j.out
#SBATCH --error=prepare_dataset-%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=25
#SBATCH --time=48:00:00
#SBATCH --mem=200gb

python -m freqdect.prepare_dataset /nvme/mwolter/ffhq1024x1024 --train-size 5000 --val-size 2000 --test-size 500 --batch-size 100 --log-packets
