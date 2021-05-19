#!/bin/bash
#
#SBATCH --nodes=1
# Use all CPUs on the node
#SBATCH --cpus-per-task=32
#SBATCH --job-name=baselines
#SBATCH --output=baselines-%j.out
#SBATCH --error=baselines-%j.err
# Send the USR1 signal 120 seconds before end of time limit
#SBATCH --signal=B:USR1@120
# Set time limit to override default limit
#SBATCH --time=24:00:00


echo baseline.sh started at `date +"%T"`

ANACONDA_ENV="$HOME/env/intel38"

OUTPUT_DIR="baselines/results"
DATASETS_DIR="/home/ndv/projects/wavelets/frequency-forensics_felix/data"
LSUN_DATASET_RAW="lsun_bedroom_200k_png_raw_baseline"
LSUN_DATASET_PACKETS="lsun_bedroom_200k_png_packets_baseline"
LSUN_DATASET_TEST="ex_lsun_bedroom_200k_png_packets"
TAR_NAME="lsun_bedroom_200k_png_baseline.tar"

# select baseline to compute from {"knn", "prnu", "eigenfaces"}
BASELINE="knn"

# first three are channelwise mean, last three channelwise std
MEAN_STD_RAW_CHANNELWISE="175.4984 163.5837 152.6461 56.3215 60.2467 64.2528"
MEAN_STD_PACKETS_CHANNELWISE="0.1826 0.2155 0.2154 4.4256 4.3896 4.3595"

# first is overall mean, last is overall std
MEAN_STD_PACKETS="0.2045 4.3917"
MEAN_STD_RAW="163.9094 61.0777"

cp_results_from_tmp()
{
  if [ -d ${TMPDIR}/${OUTPUT_DIR} ]; then
    echo "Copying results back to ${SLURM_SUBMIT_DIR}"

    # make sure that the output dir exists
    mkdir -p ${SLURM_SUBMIT_DIR}/${OUTPUT_DIR}
    cp -r ${TMPDIR}/${OUTPUT_DIR}/. ${SLURM_SUBMIT_DIR}/${OUTPUT_DIR}
  fi
}

# Define the signal handler function
finalize_job()
{
  echo Signal USR1 trapped at `date +"%T"`
  cp_results_from_tmp
  exit
}

# Call finalize_job function as soon as we receive USR1 signal (2 min before timeout)
trap 'finalize_job' USR1

module load Anaconda3
source activate "$ANACONDA_ENV"

if [ -f ${DATASETS_DIR}/${TAR_NAME} ]; then
  echo "Tarred raw input folder exists, copying to $TMPDIR"
  cp "${DATASETS_DIR}/${TAR_NAME}" "$TMPDIR"/
  cd "$TMPDIR"
  echo "Unpacking tarred input folder"
  tar xf ${TAR_NAME}
  DATASETS_DIR=${TMPDIR}

  # delete .tar file, which is not needed anymore
  rm ${TAR_NAME}
fi

# work on scratch dir
cd $TMPDIR

# copy existing results to avoid repetition
if [ -d ${SLURM_SUBMIT_DIR}/${OUTPUT_DIR} ]; then
  echo "Copying existing results to ${TMPDIR}"
  mkdir -p ${TMPDIR}/${OUTPUT_DIR}
  cp -r ${SLURM_SUBMIT_DIR}/${OUTPUT_DIR}/. ${TMPDIR}/${OUTPUT_DIR}
fi

echo "Calculating baseline data"

# -u: unbuffered stdout for "live" updates in the output file
python -u -m freqdect.baselines.baselines \
  --command grid_search \
  --output_dir $OUTPUT_DIR \
  --datasets_dir $DATASETS_DIR \
  --datasets $LSUN_DATASET_PACKETS \
  --normalize $MEAN_STD_PACKETS \
  --n_jobs 32 \
  $BASELINE

# release signal
trap - USR1

# save the results
cp_results_from_tmp

echo baseline.sh finished at `date +"%T"`

exit
