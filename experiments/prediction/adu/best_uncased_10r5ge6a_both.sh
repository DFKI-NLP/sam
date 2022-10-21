python run.py predagg --predictor brat-store \
-o "{dataset_reader:{show_gold:true,show_prediction:true,num_shards:null,dataset_splits:{test:\"30:\"}}}" \
--use-dataset-reader \
--cuda-device 0 \
--output-file experiments/prediction/adu/best_uncased_10r5ge6a_both \
--batch-size 16 \
--silent \
experiments/training/adu/uncased_best_adu/uncased_best/10r5ge6a \
./dataset_scripts/sciarg.json@test