#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_lsun
#SBATCH --output=train_lsun-%j.out
#SBATCH --error=train_lsun-%j.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres gpu:v100:8
#SBATCH --cpus-per-task=2

DATASET_RAW="../datasets/lsun_bedroom_200k_png_raw"
DATASET_PACKETS="../datasets/lsun_bedroom_200k_png_packets"
ANACONDA_ENV="~/myconda-env"

module load CUDA
module load Anaconda3
module load PyTorch
source activate $ANACONDA_ENV

pip install -q -e .

for i in 0 1 2 3 4
do
   echo "packet experiment no: $i "
   python -m freqdect.train_classifier \
	--features packets \
	--seed $i \
	--data-prefix $DATASET_PACKETS \
	--nclasses 4 \
	--calc-normalization
done

for i in 0 1 2 3 4
do
   echo "raw experiment no: $i "
   python -m freqdect.train_classifier \
        --features raw \
        --seed $i \
        --data-prefix $DATASET_RAW \
        --nclasses 4 \
        --calc-normalization
done

