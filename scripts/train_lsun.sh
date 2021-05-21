#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=regression-lsun
#SBATCH --output=regression-lsun-%j.out
#SBATCH --error=regression-lsun-%j.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=16
# Set time limit to override default limit
#SBATCH --time=48:00:00

ANACONDA_ENV="$HOME/env/intel38"

DATASETS_DIR="/home/ndv/projects/wavelets/frequency-forensics_felix/data"

LSUN_DATASET_LOGPACKETS="lsun_bedroom_200k_png_logpackets"
LSUN_DATASET_PACKETS="lsun_bedroom_200k_png_packets"
LSUN_DATASET_RAW="lsun_bedroom_200k_png_raw"

CELEBA_DATASET_LOGPACKETS="celeba_align_png_cropped_logpackets"
CELEBA_DATASET_PACKETS="celeba_align_png_cropped_packets"
CELEBA_DATASET_RAW="celeba_align_png_cropped_raw"

module load CUDA
module load Anaconda3
source activate "$ANACONDA_ENV"

train_on_dataset() {
  # $1 : dataset name
  # $2 : feature
  echo "training for $1 started at `date +"%T"`"

  for i in 0 1 2 3 4
  do
    echo "$1 experiment no: $i "
    python -m freqdect.train_classifier \
      --features $2 \
      --seed $i \
      --data-prefix ${DATASETS_DIR}/$1 \
      --nclasses 5 \
      --calc-normalization
  done

  echo "training for $1 ended at `date +"%T"`"
}

train_on_dataset $LSUN_DATASET_LOGPACKETS "packets"

train_on_dataset $LSUN_DATASET_PACKETS "packets"

train_on_dataset $LSUN_DATASET_RAW "raw"

exit
