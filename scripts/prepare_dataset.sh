#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=prepare_dataset
#SBATCH --output=prepare_dataset-%j.out
#SBATCH --error=prepare_dataset-%j.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=16

#DATASET="../datasets/lsun_bedroom_200k_png"
ANACONDA_ENV="~/myconda-env"

DATASETS="/home/ndv/projects/wavelets/datasets"
DATESET="lsun_bedroom_200k_png"

if [ -f ${DATASETS}/${DATASET}.tar ]; then
  echo "Tarred input folder exists, copying to $TMPDIR"
  cp "${DATASETS}/${DATASET}.tar" $TMPDIR/
  cd $TMPDIR
  tar xf ${DATASET}.tar
  DATASET="${TMPDIR}/$DATASET"
fi

module load PyTorch
module load Pillow
module load Anaconda3
module load Python
source activate $ANACONDA_ENV

pip install -q -e .

python -m freqdect.prepare_dataset_batched $DATASET \
  --train-size 100000 \
  --test-size 30000 \
  --val-size 20000 \
  -p

#if [ -d "${TMPDIR}/${DATASET}_raw_train" ]; then
if ls "${TMPDIR}/${DATASET}_*" > /dev/null 2>&1; then
  cp ""${TMPDIR}/${DATASET}_*" "${DATASETS}"
fi
