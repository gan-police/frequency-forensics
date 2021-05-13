for i in 0 1 2 3 4
do
   echo "packet experiment no: $i "
   CUDA_VISIBLE_DEVICES=0 python src/freqdect/train_classifier.py --features packets --seed $i --data-prefix ./data/source_data_packets
done

for i in 0 1 2 3 4
do
   echo "packet experiment no: $i "
   CUDA_VISIBLE_DEVICES=0 python src/freqdect/train_classifier.py --features raw --seed $i --data-prefix ./data/source_data_raw
done

