python analysis/calculate_metric.py \
--path_gold experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_goldonly \
--path_predicted experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_predictiononly \
--out_dir experiments/evaluation/using_pipeline/rel@gold_adus/best_uncased_257eyrv1_dont_merge \
--type_blacklist "semantically_same"