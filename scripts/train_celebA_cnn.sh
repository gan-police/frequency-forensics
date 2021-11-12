#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=celeba_train_cnn
#SBATCH --output=train_celeba_packets_log_packets_cnn-%j.out
#SBATCH --error=train_celeba_packets_log_packets_cnn-%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=200gb

source activate ~/env/idp38

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --epochs 20 \
	  --data-prefix /nvme/fblanke/celeba_align_png_cropped_log_packets_db4_boundary \
	  --nclasses 5 \
	  --calc-normalization \
	  --model cnn
done
