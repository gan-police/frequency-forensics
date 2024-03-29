#!/bin/bash
#
#SBATCH --nodes=1
# Use all CPUs on the node
#SBATCH --cpus-per-task=32
#SBATCH --job-name=l-eigenfaces-lsun
#SBATCH --output=l-eigenfaces-lsun-%j.out
#SBATCH --error=l-eigenfaces-lsun-%j.err
# Send the USR1 signal 120 seconds before end of time limit
#SBATCH --signal=B:USR1@120
# Set time limit to override default limit
#SBATCH --time=48:00:00


echo baseline.sh started at `date +"%T"`

ANACONDA_ENV="$HOME/env/intel38"

OUTPUT_DIR="baselines/results"
DATASETS_DIR="/home/ndv/projects/wavelets/frequency-forensics_felix/data"

LSUN_DATASET_LOGPACKETS="lsun_bedroom_200k_png_baseline_logpackets"
LSUN_DATASET_PACKETS="lsun_bedroom_200k_png_baseline_packets"
LSUN_DATASET_RAW="lsun_bedroom_200k_png_baseline_raw"

CELEBA_DATASET_LOGPACKETS="celeba_align_png_cropped_baselines_logpackets"
CELEBA_DATASET_PACKETS="celeba_align_png_cropped_baselines_packets"
CELEBA_DATASET_RAW="celeba_align_png_cropped_baselines_raw"


# select baseline to compute from {"knn", "prnu", "eigenfaces"}
BASELINE="eigenfaces"

# first three are channelwise mean, last three channelwise std
MEAN_STD_RAW_CHANNELWISE="175.4984 163.5837 152.6461 56.3215 60.2467 64.2528"
MEAN_STD_PACKETS_CHANNELWISE="0.1826 0.2155 0.2154 4.4256 4.3896 4.3595"

# first is overall mean, last is overall std
LSUN_MEAN_STD_LOGPACKETS="0.3281 4.2175"
LSUN_MEAN_STD_PACKETS="19.8486 168.9453"
LSUN_MEAN_STD_RAW="157.9363 63.1872"

CELEBA_MEAN_STD_LOGPACKETS="0.7375 3.4890"
CELEBA_MEAN_STD_PACKETS="17.5999 155.5985"
CELEBA_MEAN_STD_RAW="140.8967 68.3285"

CHOSEN_DATASET=$LSUN_DATASET_LOGPACKETS
CHOSEN_NORMALIZATION=$LSUN_MEAN_STD_LOGPACKETS


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

if [ -f ${DATASETS_DIR}/${CHOSEN_DATASET}.tar ]; then
  echo "Tarred raw input folder exists, copying to $TMPDIR"
  cp "${DATASETS_DIR}/${CHOSEN_DATASET}.tar" "$TMPDIR"/
  cd "$TMPDIR"
  echo "Unpacking tarred input folder"
  tar xf ${CHOSEN_DATASET}.tar
  DATASETS_DIR=${TMPDIR}

  # delete .tar file, which is not needed anymore
  rm ${CHOSEN_DATASET}.tar
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
  --datasets $CHOSEN_DATASET \
  --normalize $CHOSEN_NORMALIZATION \
  --n_jobs 32 \
  $BASELINE

# release signal
trap - USR1

# save the results
cp_results_from_tmp

echo baseline.sh finished at `date +"%T"`

exit
