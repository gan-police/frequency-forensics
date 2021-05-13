CUDA_VISIBLE_DEVICES=1 python src/freqdect/prepare_dataset_batched.py ./data/ffhq_stylegan_large --batch-size 128 --train-size 10000 --val-size 500 --test-size 2000 --packets
