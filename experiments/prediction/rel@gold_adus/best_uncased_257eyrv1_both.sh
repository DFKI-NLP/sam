allennlp \
predagg \
--predictor brat-store \
-o "{\"data_loader.shuffle\":false,\"dataset_reader.show_gold\":true,\"dataset_reader.show_prediction\":true,\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits.test\":\"30:\",\"dataset_reader.add_negative_relations_portion\":-1.0}" \
--use-dataset-reader \
--cuda-device 0 \
--batch-size 128 \
--silent \
--output-file experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_both \
experiments/training/rel/best_uncased/257eyrv1 \
./dataset_scripts/sciarg.json@test