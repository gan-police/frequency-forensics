#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=celeba_train
#SBATCH --output=train_celeba-%j.out
#SBATCH --error=train_celeba-%j.err
#SBATCH --ntasks=1
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=16

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python src/freqdect/train_classifier.py \
	  --features packets \
	  --seed $i \
	  --data-prefix /home/ndv/projects/wavelets/datasets_moritz/celeba_align_png_cropped_packets \
	  --nclasses 4 \
	  --calc-normalization
done
