#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH -n 32
#SBATCH --tasks-per-node=1
#SBATCH --job-name=baselines
#SBATCH --output=baselines-%j.out
#SBATCH --error=baselines-%j.err
#SBATCH --ntasks=1
#SBATCH --time=120:00
# Send the USR1 signal 120 seconds before end of time limit
#SBATCH --signal=B:USR1@120


echo baseline.sh started at `date +"%T"`

ANACONDA_ENV="$HOME/myconda-env"

OUTPUT_DIR="baselines/results"
DATASETS_DIR="/home/ndv/projects/wavelets/frequency-forensics_felix/data"
LSUN_DATASET_RAW="lsun_bedroom_200k_png_raw_baseline"
LSUN_DATASET_PACKETS="lsun_bedroom_200k_png_packets_baseline"

TAR_NAME="lsun_bedroom_200k_png_baseline.tar"

# select baseline to compute from {"knn", "prnu", "eigenfaces"}
BASELINE="knn"

cp_results_from_tmp()
{
  if [ -d ${TMPDIR}/${OUTPUT_DIR} ]; then
    echo "Copying results back to ${SLURM_SUBMIT_DIR}"
    cp -r ${TMPDIR}/${OUTPUT_DIR} ${SLURM_SUBMIT_DIR}
  fi
}

# Define the signal handler function
finalize_job()
{
  echo Signal USR1 trapped at `date +"%T"`
  cp_results_from_tmp
  exit
}

# Call finalize_job function as soon as we receive USR1 signal
trap 'finalize_job' USR1

module load Anaconda3
source activate "$ANACONDA_ENV"

pip install -q -e .

if [ -f ${DATASETS_DIR}/${TAR_NAME} ]; then
  echo "Tarred raw input folder exists, copying to $TMPDIR"
  cp "${DATASETS_DIR}/${TAR_NAME}" "$TMPDIR"/
  cd "$TMPDIR"
  echo "Unpacking tarred input folder"
  tar xf ${TAR_NAME}
  DATASETS_DIR=${TMPDIR}
fi

echo "Calculating baseline data"
python -m freqdect.baselines.baselines \
  --command grid_search \
  --output_dir $OUTPUT_DIR \
  --datasets_dir $DATASETS_DIR \
  --datasets $LSUN_DATASET_PACKETS \
  --datasets $LSUN_DATASET_RAW \
  $BASELINE

trap - USR1

cp_results_from_tmp

echo baseline.sh finished at `date +"%T"`

exit
