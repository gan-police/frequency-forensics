for i in 0 1 2 3 4
do
   echo "packet experiment no: $i "
   python -m freqdect.train_classifier --features packets --seed $i --data-prefix /nvme/mwolter/source_data_log_packets
done

for i in 0 1 2 3 4
do
   echo "packet experiment no: $i "
   python -m freqdect.train_classifier --features raw --seed $i --data-prefix /nvme/mwolter/source_data_raw
done

