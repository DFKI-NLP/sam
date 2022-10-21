import os

from analysis.calculate_metric import get_metrics


def calculate_filtered_predictions_for_tasks_and_types(annotation_types, tasks, **kwargs):
    for task in tasks:
        for gold in annotation_types + ['UNASSIGNABLE']:
            for predicted in annotation_types + ['UNDETECTED']:
                if gold == 'UNASSIGNABLE' and predicted == 'UNDETECTED':
                    continue
                get_metrics(
                    filter_evaluation_task=task, filter_gold_type=gold, filter_predicted_type=predicted, **kwargs
                )


def main(filter_output_dir: str, prediction_dir: str, do_entity: bool = True, do_rel: bool = True):

    if do_entity:
        calculate_filtered_predictions_for_tasks_and_types(
            annotation_types=['background_claim', 'own_claim', 'data'],
            tasks=['entity', 'entity_weak'],
            path_gold=os.path.join(prediction_dir, 'adu/best_uncased_10r5ge6a_goldonly'),
            path_predicted=os.path.join(prediction_dir, 'adu/best_uncased_10r5ge6a_predictiononly'),
            out_dir=None,
            exclude_labels_with_suffix='GOLD',
            type_blacklist=None,
            file_whitelist=None,
            merge_span_annotations_with_relation_label=None,
            filter_output_dir=os.path.join(filter_output_dir, 'adu')
        )

    if do_rel:
        calculate_filtered_predictions_for_tasks_and_types(
            annotation_types=['supports', 'contradicts'],
            tasks=['rel'],
            path_gold=os.path.join(prediction_dir, 'rel@gold_adus/best_uncased_257eyrv1_goldonly'),
            path_predicted=os.path.join(prediction_dir, 'rel@gold_adus/best_uncased_257eyrv1_predictiononly'),
            out_dir=None,
            exclude_labels_with_suffix=None,
            type_blacklist='semantically_same',
            file_whitelist=None,
            merge_span_annotations_with_relation_label='parts_of_same',
            filter_output_dir=os.path.join(filter_output_dir, 'rel@gold_adu')
        )

        calculate_filtered_predictions_for_tasks_and_types(
            annotation_types=['supports', 'contradicts'],
            tasks=['rel'],
            path_gold=os.path.join(prediction_dir, 'rel@gold_adus/best_uncased_257eyrv1_goldonly'),
            path_predicted=os.path.join(prediction_dir, 'rel@gold_adus/best_uncased_257eyrv1_predictiononly'),
            out_dir=None,
            exclude_labels_with_suffix=None,
            type_blacklist='semantically_same',
            file_whitelist=None,
            merge_span_annotations_with_relation_label='parts_of_same',
            filter_output_dir=os.path.join(filter_output_dir, 'rel_merged_via_gold@gold_adu'),
            use_gold_rels_for_merging=True,
        )


if __name__ == "__main__":
    main(filter_output_dir="experiments/prediction/filtered", prediction_dir="experiments/prediction")
