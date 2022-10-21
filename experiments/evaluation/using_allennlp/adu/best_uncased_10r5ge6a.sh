allennlp evaluate \
-o "{\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits\":{\"test\":\"30:\"}, \"model.calculate_weak_span_f1\":true}" \
--cuda-device 0 \
--batch-size 8 \
--output-file experiments/evaluation/using_allennlp/adu/best_uncased_10r5ge6a.json \
experiments/training/adu/uncased_best_adu/uncased_best/10r5ge6a \
./dataset_scripts/sciarg.json@test

