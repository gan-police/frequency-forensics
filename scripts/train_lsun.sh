#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_lsun
#SBATCH --output=train_lsun-%j.out
#SBATCH --error=train_lsun-%j.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=16

DATASETS="/home/ndv/projects/wavelets/datasets"
DATASET_RAW="lsun_bedroom_200k_png_raw"
DATASET_PACKETS="lsun_bedroom_200k_png_packets"

ANACONDA_ENV="$HOME/myconda-env"

# save working directory
ORIG_PWD=${PWD}

RAW_PREFIX=$DATASETS
PACKETS_PREFIX=$DATASETS

if [ -f ${DATASETS}/${DATASET_RAW}.tar ]; then
  echo "Tarred raw input folder exists, copying to $TMPDIR"
  cp "${DATASETS}/${DATASET_RAW}.tar" "${TMPDIR}"
  cd "$TMPDIR"
  tar -xf "${DATASET_RAW}.tar"
  RAW_PREFIX="${TMPDIR}/${DATASET_RAW}"
fi

if [ -f ${DATASETS}/${DATASET_PACKETS}.tar ]; then
  echo "Tarred packets input folder exists, copying to $TMPDIR"
  cp "${DATASETS}/${DATASET_PACKETS}.tar" "${TMPDIR}"
  cd "$TMPDIR"
  tar -xf "${DATASET_PACKETS}.tar"
  PACKETS_PREFIX="${TMPDIR}/${DATASET_PACKETS}"
fi

cd "$ORIG_PWD"

module load CUDA
module load Anaconda3
module load PyTorch
source activate "$ANACONDA_ENV"

pip install -q -e .

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix "$PACKETS_PREFIX" \
	  --nclasses 4 \
	  --calc-normalization
done

for i in 0 1 2 3 4
do
  echo "raw experiment no: $i "
  python -m freqdect.train_classifier \
    --features raw \
    --seed $i \
    --data-prefix "$RAW_PREFIX" \
    --nclasses 4 \
    --calc-normalization
done
