#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=confusion-matrix-celeba-CNN-packet
#SBATCH --output=conf-db2-cnn-generalized.out
#SBATCH --error=conf-db2-cnn-generalized.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=4
# Set time limit to override default limit
#SBATCH --time=2:00:00

ANACONDA_ENV="$HOME/env/idp38"

DATASETS_DIR="/nvme/fblanke"

LSUN_DATASET_LOGPACKETS="lsun_bedroom_200k_png_log_packets"
LSUN_DATASET_PACKETS="lsun_bedroom_200k_png_packets"
LSUN_DATASET_RAW="lsun_bedroom_200k_png_raw"

WAVELET="db2"
CHOSEN_DATASET=${LSUN_DATASET_LOGPACKETS}_${WAVELET}_boundary_missing_4
BATCH_SIZE="2048"
MODEL="cnn"
SUFFIX=""
NCLASSES="2"

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
    --data-prefix ${DATASETS_DIR}/${CHOSEN_DATASET} \
    --calc-normalization \
    --batch-size $BATCH_SIZE \
    --nclasses ${NCLASSES} \
    --model $MODEL \
    --store-path confusion-matrix-${CHOSEN_DATASET}_${MODEL}_${i}-generalized.npy \
    --generalized
done
echo "ended at `date +"%T"`"
