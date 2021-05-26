#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=confusion-matrix-celeba-CNN-packet
#SBATCH --output=confusion-matrix-celeba-CNN-packet-%j.out
#SBATCH --error=confusion-matrix-celeba-CNN--packet-%j.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=4
# Set time limit to override default limit
#SBATCH --time=2:00:00

ANACONDA_ENV="$HOME/env/intel38"

DATASETS_DIR="/nvme/mwolter"

LSUN_DATASET_LOGPACKETS="lsun_bedroom_200k_png_logpackets"
LSUN_DATASET_PACKETS="lsun_bedroom_200k_png_packets"
LSUN_DATASET_RAW="lsun_bedroom_200k_png_raw"

CELEBA_DATASET_LOGPACKETS="celeba_align_png_cropped_logpackets"
CELEBA_DATASET_PACKETS="celeba_align_png_cropped_packets"
CELEBA_DATASET_RAW="celeba_align_png_cropped_raw"

LSUN_MEAN_STD_LOGPACKETS="0.3113 0.3354 0.3428 4.2545 4.2146 4.1707"
LSUN_MEAN_STD_PACKETS="21.4172 19.7907 18.3627 178.8472 168.1840 159.3333"
LSUN_MEAN_STD_RAW="170.5132 157.4605 146.0014 58.0541 62.3195 66.3918"

CELEBA_MEAN_STD_LOGPACKETS="0.7412 0.7452 0.7287 3.5217 3.4766 3.4535"
CELEBA_MEAN_STD_PACKETS="20.2251 17.0874 15.4814 174.8270 150.5715 139.2445"
CELEBA_MEAN_STD_RAW="162.1359 136.7685 123.7591 68.4507 65.2545 65.6688"

CHOSEN_DATASET=$CELEBA_DATASET_PACKETS
CHOSEN_NORMALIZATION=$CELEBA_MEAN_STD_PACKETS
BATCH_SIZE="2048"
MODEL="CNN"
SUFFIX="batch_size_1024_epochs_20"

module load CUDA
module load Anaconda3
source activate "$ANACONDA_ENV"

echo "confusion matrices for ${CHOSEN_DATASET}:"
for i in 0 1 2 3 4
do
  echo "confusion matrix for: $i "
  python -m freqdect.confusion_matrix \
    --features packets \
    --classifier_path log/models/${MODEL}/${CHOSEN_DATASET}_${MODEL}_${i}_${SUFFIX}.pt \
    --data ${DATASETS_DIR}/${CHOSEN_DATASET}_test \
    --normalize ${CHOSEN_NORMALIZATION} \
    --batch-size $BATCH_SIZE \
    --model $MODEL
done