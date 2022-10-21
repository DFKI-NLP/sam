python analysis/calculate_metric.py \
--path_gold experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_goldonly \
--path_predicted experiments/prediction/rel@gold_adus/best_uncased_257eyrv1_predictiononly \
--out_dir experiments/evaluation/using_pipeline/rel@gold_adus/best_uncased_merged_via_gold_257eyrv1 \
--type_blacklist "semantically_same" \
--merge_span_annotations_with_relation_label parts_of_same \
--use_gold_rels_for_merging