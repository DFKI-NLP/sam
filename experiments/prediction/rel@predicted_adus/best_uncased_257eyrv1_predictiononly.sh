allennlp \
predagg \
--predictor brat-store \
-o "{\"data_loader.shuffle\":false,\"dataset_reader.show_gold\":false,\"dataset_reader.show_prediction\":true,\"dataset_reader.add_negative_relations_portion\":-1.0,\"dataset_reader.num_shards\":null,\"dataset_reader.dataset_splits.test\":\":\"}" \
--use-dataset-reader \
--cuda-device 0 \
--batch-size 128 \
--silent \
--output-file experiments/prediction/rel@predicted_adus/best_uncased_257eyrv1_predictiononly \
experiments/training/rel/best_uncased/257eyrv1 \
experiments/prediction/adu/best_3yy7nim5_predictiononly@test