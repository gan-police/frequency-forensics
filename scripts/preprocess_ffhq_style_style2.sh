#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=prepare_ffhq
#SBATCH --output=prepare_ffhq-%j.out
#SBATCH --error=prepare_ffhq-%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --mem=90gb

python -m freqdect.prepare_dataset /nvme/mwolter/ffhq128/source_data --log-packets --wavelet haar --boundary boundary --train-size 62000
python -m freqdect.prepare_dataset /nvme/mwolter/ffhq128/source_data --raw --train-size 62000