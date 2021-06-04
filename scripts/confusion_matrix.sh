#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=confusion-matrix-celeba-CNN-packet
#SBATCH --output=confusion-matrix-celeba-CNN-packet-%j.out
#SBATCH --error=confusion-matrix-celeba-CNN-packet-%j.err
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

LSUN_MEAN_STD_LOGPACKETS_MISSING_4="0.1834 0.2001 0.2211 4.6584 4.6449 4.5823"
LSUN_MEAN_STD_PACKETS_MISSING_4="20.9556 19.2935 17.8027 176.0325 165.1051 155.9335"
LSUN_MEAN_STD_RAW_MISSING_4="166.8336 153.5023 141.5420 59.9429 63.7886 67.8090"

CELEBA_MEAN_STD_LOGPACKETS_MISSING_4="0.7664 0.7643 0.7522 3.6087 3.5644 3.5418"
CELEBA_MEAN_STD_PACKETS_MISSING_4="19.7641 16.4999 14.8420 172.0986 146.7396 135.0101"
CELEBA_MEAN_STD_RAW_MISSING_4="158.4538 132.0428 118.6135 70.0066 66.1019 66.1729"

CHOSEN_DATASET=${CELEBA_DATASET_PACKETS}_missing_4
CHOSEN_NORMALIZATION=$CELEBA_MEAN_STD_PACKETS_MISSING_4
BATCH_SIZE="2048"
MODEL="CNN"
SUFFIX="" #"_batch_size_1024_epochs_20"
NCLASSES="2"

module load CUDA
module load Anaconda3
source activate "$ANACONDA_ENV"

echo "confusion matrices for ${CHOSEN_DATASET}:"
for i in 0 1 2 3 4
do
  CLASSIFIER_FILE="${CHOSEN_DATASET}_${MODEL}_${i}${SUFFIX}"
  echo "confusion matrix for: $i "
  python -m freqdect.confusion_matrix \
    --features packets \
    --classifier-path log/${CLASSIFIER_FILE}.pt \
    --data ${DATASETS_DIR}/${CHOSEN_DATASET}_test \
    --normalize ${CHOSEN_NORMALIZATION} \
    --batch-size $BATCH_SIZE \
    --nclasses ${NCLASSES} \
    --model $MODEL \
    --generalized
done
echo "ended at `date +"%T"`"
