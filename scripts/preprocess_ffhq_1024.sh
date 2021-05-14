CUDA_VISIBLE_DEVICES=1 python src/freqdect/prepare_dataset_batched.py ./data/ffhq_stylegan_large --batch-size 100 --train-size 60000 --val-size 2000 --test-size 8000 --packets
