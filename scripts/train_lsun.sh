#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=train_lsun_log_packets_db2_boundary
#SBATCH --output=train_lsun_log_packets_db2_boundary-%j.out
#SBATCH --error=train_lsun_log_packets_db2_boundary-%j.err
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1
#SBATCH --cpus-per-task=9
#SBATCH --time=48:00:00
#SBATCH --mem=200gb

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix /nvme/mwolter/lsun/lsun_bedroom_200k_png_log_packets_db2_boundary \
	  --nclasses 5 \
	  --calc-normalization
done

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix /nvme/mwolter/lsun/lsun_bedroom_200k_png_log_packets_sym2_boundary \
	  --nclasses 5 \
	  --calc-normalization
done

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix /nvme/mwolter/lsun/lsun_bedroom_200k_png_log_packets_db4_boundary \
	  --nclasses 5 \
	  --calc-normalization
done

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix /nvme/mwolter/lsun/lsun_bedroom_200k_png_log_packets_sym4_boundary \
	  --nclasses 5 \
	  --calc-normalization
done


for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix /nvme/mwolter/lsun/lsun_bedroom_200k_png_log_packets_db2_boundary \
	  --nclasses 5 \
	  --calc-normalization \
	  --model cnn
done

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix /nvme/mwolter/lsun/lsun_bedroom_200k_png_log_packets_sym2_boundary \
	  --nclasses 5 \
	  --calc-normalization \
	  --model cnn
done

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix /nvme/mwolter/lsun/lsun_bedroom_200k_png_log_packets_db4_boundary \
	  --nclasses 5 \
	  --calc-normalization \
	  --model cnn
done

for i in 0 1 2 3 4
do
  echo "packet experiment no: $i "
  python -m freqdect.train_classifier \
	  --features packets \
	  --seed $i \
	  --data-prefix /nvme/mwolter/lsun/lsun_bedroom_200k_png_log_packets_sym4_boundary \
	  --nclasses 5 \
	  --calc-normalization \
	  --model cnn
done