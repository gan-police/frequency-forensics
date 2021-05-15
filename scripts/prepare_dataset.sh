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

echo prepare_dataset.sh started at `date +"%T"`

ANACONDA_ENV="$HOME/myconda-env"

DATASETS="/home/ndv/projects/wavelets/datasets"
DATASET="lsun_bedroom_200k_png"
DATA_PREFIX="${DATASETS}/${DATASET}"

# save working directory
ORIG_PWD=${PWD}

module load PyTorch
module load Pillow
module load Anaconda3
source activate "$ANACONDA_ENV"

pip install -q -e .

if [ -f ${DATASETS}/${DATASET}.tar ]; then
  echo "Tarred input folder exists, copying to $TMPDIR"
  cp "${DATASETS}/${DATASET}.tar" "$TMPDIR"/
  cd "$TMPDIR"
  echo "Unpacking tarred input folder"
  tar xf "${DATASET}.tar"
  DATA_PREFIX="${TMPDIR}/${DATASET}"
fi

cd $ORIG_PWD

echo "Preparing data"
python -m freqdect.prepare_dataset_batched "$DATA_PREFIX" \
  --train-size 100000 \
  --test-size 30000 \
  --val-size 20000 \
  --packets

if ls ${TMPDIR}/${DATASET}_* > /dev/null 2>&1; then
  echo "Copying results back to ${DATASETS}"
  cp -r ${TMPDIR}/${DATASET}_* ${DATASETS}
fi

echo prepare_dataset.sh finished at `date +"%T"`
