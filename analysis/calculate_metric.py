import argparse
import copy
import glob
import json
import logging
import os
import shutil
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, DefaultDict, Any, Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from sam.dataset_reader.annotation import AnnotationLayer
from sam.dataset_reader.brat_annotation import BratAnnotationCollection, BratSpanAnnotation, BratSpanAnnotationLayer, \
    BratRelationAnnotationLayer, BratRelation, BratRelationAnnotation, BRAT_SPAN_LAYER_NAME, \
    BRAT_RELATION_LAYER_NAME, BratMultiSlice, BratAnnotation
from sam.metrics.weak_span_based_f1_measure import has_weak_overlap

logger = logging.getLogger(__name__)


def get_metrics_from_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """
    This method calculates precision, recall and f1 score using the true positives, false positive and false negatives
    count provided in 'counts'. It returns calculated f1 score.

    :param counts: Dictionary with keys referring to true positives, false positive and false negatives and values
                    referring to their counts.
    :return: f1 score based on given counts or 0.0 if count of true positives is 0.
    """
    if counts.get('true_positives', 0) == 0:
        result = {'precision': 0.0, 'recall': 0.0}
    else:
        result = {
            'precision': counts.get('true_positives', 0) / (
                    counts.get('true_positives', 0) + counts.get('false_positives', 0)),
            'recall': counts.get('true_positives', 0) / (
                    counts.get('true_positives', 0) + counts.get('false_negatives', 0))
        }
    product = result['precision'] * result['recall']
    if product == 0:  # to avoid nan
        result['f1'] = 0.0
    else:
        result['f1'] = 2 * product / (result['precision'] + result['recall'])
    return result


def read_brat_folder(path: str, file_whitelist: Optional[List[str]] = None) -> Dict[str, BratAnnotationCollection]:
    """
    This method takes in path of brat folder which contains annotation files (*.ann). For each file in this folder, it
    retrieve BratAnnotation using from_file() method of BratAnnotationCollection class and store in a dictionary with
    base name of file (eg: A32[1787:4935]) as key and annotation collections as value.

    :param file_whitelist: List of filenames. If set, use only these filenames from input directory for evaluation
    :param path: path to brat folder containing annotation files
    :return: a dictionary with filename as keys and annotation collection as value.
    """
    if file_whitelist is None:
        filenames = glob.glob(path + '/*.ann')
    else:
        filenames = [os.path.join(path, filename + '.ann') for filename in file_whitelist]

    logger.info(f'load {len(filenames)} annotation files from {path}')
    annotation_collections = {}
    for fn in filenames:
        base_fn, ext = os.path.splitext(fn)
        base_name = os.path.basename(base_fn)
        annotation_collections[base_name] = BratAnnotationCollection.from_file(base_fn)
    return annotation_collections


def get_weak_match_from_annotations(predicted_entity: BratSpanAnnotation,
                                    gold_entities: List[BratSpanAnnotation]) -> Optional[int]:
    """
    This method uses predicted entity against all gold entities to check if predicted entity has weak match (overlap)
    with any of the gold entities. If there is a weak match then it returns the index of matched gold entity otherwise
    it returns None.

    :param predicted_entity: BratSpanAnnotation of predicted entity containing label (labels/types for ADU recognition:
        OWN_CLAIM, BACKGROUND_CLAIM, DATA) and span of the entity.
    :param gold_entities: It is a list of true BratSpanAnnotation of entities.
    :return: index of matched gold entity or None
    """
    match_found = None
    predicted_type = predicted_entity.label
    # TODO Check if slices are really sorted
    predicted_indices = (predicted_entity.target.slices[0].start, predicted_entity.target.slices[-1].end)

    for idx, gold_entity in enumerate(gold_entities):
        gold_type = gold_entity.label
        # TODO Check if slices are really sorted
        gold_indices = (gold_entity.target.slices[0].start, gold_entity.target.slices[-1].end)
        if predicted_type == gold_type and has_weak_overlap(predicted_indices, gold_indices):
            match_found = idx
            break
    return match_found


def construct_weak_entity_layer(predicted_entities: BratSpanAnnotationLayer, gold_entities: BratSpanAnnotationLayer) \
        -> Tuple[BratSpanAnnotationLayer, Dict[BratSpanAnnotation, BratSpanAnnotation]]:
    """
    This method creates a new BratSpanAnnotationLayer based on weak span match for predicted entities. For each
    predicted entity, it checks if there exist a weak span match with any of the gold entities. If there is a weak
    match then it adds matched gold entity to a dictionary ('new_entities') with key as current 'predicted_entity' and
    value as matched gold entity. Otherwise it adds current predicted entity to itself in the same dictionary. Finally,
    this dictionary is passed to BratSpanAnnotationLayer to create new entity layer.

    :param predicted_entities: list of predicted BratSpanAnnotation
    :param gold_entities: list of true BratSpanAnnotation
    :return: a tuple of new BratSpanAnnotationLayer and a dictionary mapping old predicted entity to new entity.
    """
    new_entities: Dict[BratSpanAnnotation, BratSpanAnnotation] = {}
    gold_entities_list = list(gold_entities)
    for predicted_entity in predicted_entities:
        gold_match_idx = get_weak_match_from_annotations(predicted_entity, gold_entities_list)
        # gold_match_idx can have value as 0, so we use 'is not None'
        if gold_match_idx is not None:
            gold_match = gold_entities_list[gold_match_idx]
            # to make sure that a gold entity is matched to exactly one prediction, delete from gold entries
            del gold_entities_list[gold_match_idx]
            new_entities[predicted_entity] = gold_match
        else:
            new_entities[predicted_entity] = predicted_entity
    new_layer = BratSpanAnnotationLayer(annotations=tuple(new_entities.values()))
    return new_layer, new_entities


def create_weakly_matching_predictions(
        predicted_entities: BratSpanAnnotationLayer,
        predicted_rels: BratRelationAnnotationLayer,
        gold_entities: BratSpanAnnotationLayer
) -> Tuple[BratSpanAnnotationLayer, BratRelationAnnotationLayer]:
    """
    This method constructs a weak span annotation layer from predicted span annotations. Using modified span annotations,
    it updates the relation annotations which contain modified spans.
    :param predicted_entities: predicted span annotations
    :param predicted_rels: predicted relation annotations
    :param gold_entities: gold span annotations
    :return: modified predicted entities and relations
    """
    predicted_entities, entity_mapping = construct_weak_entity_layer(
        predicted_entities=predicted_entities,
        gold_entities=gold_entities
    )
    new_rel_annotations = []
    for annotation in predicted_rels:
        new_relation = BratRelation.from_dict(
            **{arg_name: entity_mapping[arg_val] for arg_name, arg_val
               in annotation.target.arguments._asdict().items()}
        )
        new_relation_annotation = BratRelationAnnotation(label=annotation.label, target=new_relation)
        new_rel_annotations.append(new_relation_annotation)

    predicted_rels = BratRelationAnnotationLayer(annotations=tuple(new_rel_annotations))
    return predicted_entities, predicted_rels


def merge_span_annotations(span_annotations: List[BratSpanAnnotation]):
    """
    Given a list of span annotations, it merges all spans into one span with multiple slices.
    :param span_annotations: list of span annotations to be merged
    :return: single merged span annotation
    """
    assert len(span_annotations) > 0, "no span annotations to merge"

    slices = span_annotations[0].target.slices
    for other in span_annotations[1:]:
        assert span_annotations[0].label == other.label, "label mismatch, can not merge span annotations"
        assert span_annotations[
                   0].target.base == other.target.base, "base string mismatch, can not merge span annotations"
        slices = slices + other.target.slices
    sorted_slices = tuple(sorted(slices, key=lambda s: s.start))
    return BratSpanAnnotation(label=span_annotations[0].label,
                              target=BratMultiSlice(slices=sorted_slices, base=span_annotations[0].target.base))


def merge_span_annotations_by_relations(
        span_annotations: BratSpanAnnotationLayer,
        relation_annotations: BratRelationAnnotationLayer,
        merge_relation_label: str,
        merge_relation_annotations: Optional[BratRelationAnnotationLayer] = None,
) -> Tuple[BratSpanAnnotationLayer, BratRelationAnnotationLayer]:
    """
    This method merges spans depending upon relation label provided. If there is a relation with merge_relation_label
    then arguments (spans) of that relation are merged into single span with multiple slices. Then it modifies
    the relations which contain spans that were merged, by replacing them with the newly created spans.

    :param span_annotations: span annotation layer containing a list of span annotations
    :param relation_annotations: relation annotation layer containing a list of relation annotations
    :param merge_relation_label: relation label whose argument spans will be merged into single spans
    :return: the modified span and relation annotations based on merged spans if any
    """

    # convert list of relations to a graph to easily calculate connected components to merge
    G = nx.Graph()
    merge_relations = []
    if merge_relation_annotations is None:
        merge_relation_annotations = relation_annotations
    for rel in merge_relation_annotations:
        if rel.label == merge_relation_label:
            merge_relations.append(rel)
            args = list(rel.target.arguments)
            # never merge spans that have not the same label
            if args[0].label == args[1].label:
                G.add_edge(args[0], args[1])
            # else:
            #    logger.debug(f"spans to merge do not have the same label, do not merge them: {rel.target}")

    merged_spans = {}
    for connected_components in nx.connected_components(G):
        new_span_annot = merge_span_annotations(list(connected_components))
        for span_annot in connected_components:
            merged_spans[span_annot] = new_span_annot

    # collect span annotations that were not replaced
    keep_span_annots = tuple(span_annot for span_annot in span_annotations if span_annot not in merged_spans)
    all_span_annots = sorted(keep_span_annots + tuple(merged_spans.values()), key=lambda span: (span.target.slices[0].start + 1) * (span.target.slices[-1].end - span.target.slices[0].start + 1))

    targets_to_rels = defaultdict(set)
    n_mapped = 0
    for rel in relation_annotations:
        if rel.label != merge_relation_label:
            # recreate relation target with mapped argument spans, if they are in the merged spans mapping
            for arg_name, arg_val in rel.target.arguments._asdict().items():
                if arg_val in merged_spans:
                    n_mapped += 1
            new_relation = BratRelation.from_dict(
                **{arg_name: merged_spans.get(arg_val, arg_val) for arg_name, arg_val
                   in rel.target.arguments._asdict().items()}
            )
            new_relation_annotation = BratRelationAnnotation(label=rel.label, target=new_relation)
            targets_to_rels[rel.target].add(new_relation_annotation)

    # after merging, some entity pairs may have multiple relation labels assigned
    new_rel_annotations = []
    for target, rels in targets_to_rels.items():
        if len(rels) > 1:
            logger.debug(
                f"entity pair has multiple relations {[rel.label for rel in rels]}, relations will be omitted: {target}"
            )
        else:
            new_rel_annotations.append(list(rels)[0])

    return BratSpanAnnotationLayer(annotations=all_span_annots), \
           BratRelationAnnotationLayer(annotations=tuple(new_rel_annotations))


def calculate_counts_for_collection(
        gold_collection: BratAnnotationCollection,
        predicted_collection: BratAnnotationCollection,
        exclude_labels_with_suffix: Optional[str] = None,
        type_blacklist: Optional[List[str]] = None,
        confusion_key_undetected: str = 'UNDETECTED',
        confusion_key_unassignable: str = 'UNASSIGNABLE',
        merge_span_annotations_with_relation_label: Optional[str] = None,
        filter_gold_type: Optional[str] = None,
        filter_predicted_type: Optional[str] = None,
        filter_evaluation_task: Optional[str] = None,
        use_gold_rels_for_merging: bool = False,
) -> Tuple[Dict[Tuple[str, str, str], int], Dict[str, Dict[str, int]], Optional[
    List[Tuple[Optional[BratAnnotation], Optional[BratAnnotation]]]]]:
    """
    This method calculates counts of true positives, false positives, false negatives for both BratSpanAnnotation
    Collection and BratRelationAnnotationCollection. Count is calculated for both strict and weak version which is
    stored in 'counts' with keys 'entity', 'rel' for strict version of ADU and relation respectively and 'entity_weak',
    'rel_weak' otherwise.

    :param gold_collection: list of true BratAnnotation
    :param predicted_collection: list of predicted BratAnnotation
    :param exclude_labels_with_suffix: All labels of BratSpanAnnotation containing this suffix needs to be excluded from
        metric calculation.
    :param type_blacklist TODO
    :param confusion_key_undetected: The name of the row in the confusion matrices that hold the counts for gold
        annotations that were not detected at all (i.e. occur only in the gold annotations).
    :param confusion_key_unassignable: The name of the column in the confusion matrices that hold the counts for
        predicted annotations that are not assignable at all (i.e. occur only in the predicted annotations).
    :return: count of true positives, false positives, false negatives. It is a dictionary containing a tuple
        (annotation_type, metric_label_type, label_type) as key and an integer as count for this combination as value.
        Here annotation_type can be ['rel', 'rel_weak', 'entity', 'entity_weak'], metric_label_type can be
        ['true_positives', 'false_negatives', 'false_positives'] and label_type depends upon annotation_type.
        e.g: an instance of returned value could be ('rel', 'false_positives', 'support'): 14
    """

    type_blacklist = type_blacklist or []
    counts = defaultdict(lambda: 0)

    gold_entities: BratSpanAnnotationLayer = gold_collection[BRAT_SPAN_LAYER_NAME]
    predicted_entities: BratSpanAnnotationLayer = predicted_collection[BRAT_SPAN_LAYER_NAME]
    gold_rels: BratRelationAnnotationLayer = gold_collection[BRAT_RELATION_LAYER_NAME]
    predicted_rels: BratRelationAnnotationLayer = predicted_collection[BRAT_RELATION_LAYER_NAME]

    predicted_entities_weak, predicted_rels_weak = create_weakly_matching_predictions(
        predicted_entities=predicted_entities, predicted_rels=predicted_rels, gold_entities=gold_entities
    )
    original_gold_rels = gold_rels
    if merge_span_annotations_with_relation_label is not None:
        gold_entities, gold_rels = merge_span_annotations_by_relations(
            span_annotations=gold_entities, relation_annotations=gold_rels,
            merge_relation_label=merge_span_annotations_with_relation_label
        )
        predicted_entities, predicted_rels = merge_span_annotations_by_relations(
            span_annotations=predicted_entities, relation_annotations=predicted_rels,
            merge_relation_annotations=original_gold_rels if use_gold_rels_for_merging else predicted_rels,
            merge_relation_label=merge_span_annotations_with_relation_label
        )
        predicted_entities_weak, predicted_rels_weak = merge_span_annotations_by_relations(
            span_annotations=predicted_entities_weak, relation_annotations=predicted_rels_weak,
            merge_relation_annotations=original_gold_rels if use_gold_rels_for_merging else predicted_rels,
            merge_relation_label=merge_span_annotations_with_relation_label
        )

        if use_gold_rels_for_merging:
            assert gold_entities == predicted_entities, "gold and predicted does not match"

    if exclude_labels_with_suffix is not None:
        def exclude_annotation_filter(annotation):
            return annotation.label.endswith(exclude_labels_with_suffix)
    else:
        exclude_annotation_filter = None
    counts_entity = predicted_entities.count_tp_fp_fn(
        gold=gold_entities, exclude_annotation_filter=exclude_annotation_filter
    )
    counts_entity_weak = predicted_entities_weak.count_tp_fp_fn(
        gold=gold_entities, exclude_annotation_filter=exclude_annotation_filter
    )
    counts.update({('entity',) + k: v for k, v in counts_entity.items() if k[1] not in type_blacklist})
    counts.update({('entity_weak',) + k: v for k, v in counts_entity_weak.items() if k[1] not in type_blacklist})

    counts_rel = predicted_rels.count_tp_fp_fn(
        gold=gold_rels, exclude_annotation_filter=exclude_annotation_filter
    )
    counts_rel_weak = predicted_rels_weak.count_tp_fp_fn(
        gold=gold_rels, exclude_annotation_filter=exclude_annotation_filter
    )
    counts.update({('rel',) + k: v for k, v in counts_rel.items() if k[1] not in type_blacklist})
    counts.update({('rel_weak',) + k: v for k, v in counts_rel_weak.items() if k[1] not in type_blacklist})

    confusion_matrices = {
        "entity": get_type_prediction_distribution(
            gold_entities=gold_entities, predicted_entities=predicted_entities, key_undetected=confusion_key_undetected,
            key_unassignable=confusion_key_unassignable, type_blacklist=type_blacklist
        ),
        "entity_weak": get_type_prediction_distribution(
            gold_entities=gold_entities, predicted_entities=predicted_entities_weak,
            key_undetected=confusion_key_undetected, key_unassignable=confusion_key_unassignable,
            type_blacklist=type_blacklist
        ),
        "rel": get_type_prediction_distribution(
            gold_entities=gold_rels, predicted_entities=predicted_rels, key_undetected=confusion_key_undetected,
            key_unassignable=confusion_key_unassignable, type_blacklist=type_blacklist
        ),
        "rel_weak": get_type_prediction_distribution(
            gold_entities=gold_rels, predicted_entities=predicted_rels_weak, key_undetected=confusion_key_undetected,
            key_unassignable=confusion_key_unassignable, type_blacklist=type_blacklist
        ),
    }
    get_filtered_evaluation_result_kwargs = dict(
        gold_type=filter_gold_type, predicted_type=filter_predicted_type, key_undetected=confusion_key_undetected,
        key_unassignable=confusion_key_unassignable,
    )
    if filter_evaluation_task == 'entity':
        filtered_evaluation_result = get_filtered_evaluation_result(gold_entities=gold_entities,
                                                                    predicted_entities=predicted_entities,
                                                                    **get_filtered_evaluation_result_kwargs)
    elif filter_evaluation_task == 'entity_weak':
        filtered_evaluation_result = get_filtered_evaluation_result(gold_entities=gold_entities,
                                                                    predicted_entities=predicted_entities_weak,
                                                                    **get_filtered_evaluation_result_kwargs
                                                                    )
    elif filter_evaluation_task == 'rel':
        filtered_evaluation_result = get_filtered_evaluation_result(gold_entities=gold_rels,
                                                                    predicted_entities=predicted_rels,
                                                                    **get_filtered_evaluation_result_kwargs
                                                                    )
    elif filter_evaluation_task == 'rel_weak':
        filtered_evaluation_result = get_filtered_evaluation_result(gold_entities=gold_rels,
                                                                    predicted_entities=predicted_rels_weak,
                                                                    **get_filtered_evaluation_result_kwargs
                                                                    )
    else:
        if filter_evaluation_task is not None:
            raise ValueError(f'Invalid filter_evaluation_task: {filter_evaluation_task}')
        filtered_evaluation_result = None

    return counts, confusion_matrices, filtered_evaluation_result


def get_filtered_evaluation_result(
        gold_entities: AnnotationLayer,
        predicted_entities: AnnotationLayer,
        gold_type: str,
        predicted_type: str,
        key_undetected='UNDETECTED',
        key_unassignable='UNASSIGNABLE',
):
    result: List[Tuple[Optional[BratAnnotation], Optional[BratAnnotation]]] = []
    false_predictions = {}
    true_predicted = []
    for annotation in predicted_entities:
        if annotation in gold_entities:
            if annotation.label == gold_type and annotation.label == predicted_type:
                result.append((annotation, annotation))
            true_predicted.append(annotation.target)
        else:
            false_predictions[annotation.target] = annotation
    for annotation in gold_entities:
        if annotation.target in false_predictions:
            if annotation.label == gold_type and false_predictions[annotation.target].label == predicted_type:
                result.append((annotation, false_predictions[annotation.target]))
            del false_predictions[annotation.target]
        elif annotation.target not in true_predicted:
            if annotation.label == gold_type and key_undetected == predicted_type:
                result.append((annotation, None))
    for target, annotation in false_predictions.items():
        if key_unassignable == gold_type and annotation.label == predicted_type:
            result.append((None, annotation))

    return result


def calculate_metrics(counts: DefaultDict[Tuple[str, str, str], int], merge_key_sep: Optional[str] = '/') \
        -> Dict[Tuple[Any, str, Any], Any]:
    """
    This method calculates metric for entity and relation prediction at different levels. It calculates metrics like
    precision, recall and f1 score with averaging type micro and macro for both annotations (entity, rel). It also
    calculates metrics with respect to different annotation labels like 'support' (rel), 'data' (entity), etc.
    On top of this, there is another layer which separates this calculation in two parts: strict and relaxed version of
    annotations.

    :param counts: count of true positives, false positives, false negatives. It is a dictionary containing a tuple
        (annotation_type, metric_label_type, label_type) as key and an integer as count for this combination as value.
    :param merge_key_sep: if set (default: "/"), use this string to join the individual parts of the keys in the final
        metrics dict to ease dumping to json
    :return: dict containing metrics for different annotations wrt different annotation labels and metric averaging
    types.
    """
    counts = pd.Series(counts)
    metrics = {}
    for metric_type in counts.index.unique(level=0):
        counts_for_metric_type = counts.xs(metric_type)
        counts_for_all_labels = None
        metrics_for_labels = {}
        for label in counts_for_metric_type.index.unique(level=1):
            counts_for_label = counts_for_metric_type.xs(label, level=1)
            metric_for_current_label = get_metrics_from_counts(counts_for_label.to_dict())
            metrics_for_labels.update({(label, k): v for k, v in metric_for_current_label.items()})
            if counts_for_all_labels is None:
                counts_for_all_labels = counts_for_label
            else:
                # Add 0s for missing values
                for idx in set(counts_for_all_labels.index) - set(counts_for_label.index):
                    counts_for_label[idx] = 0
                for idx in set(counts_for_label.index) - set(counts_for_all_labels.index):
                    counts_for_all_labels[idx] = 0
                counts_for_all_labels += counts_for_label
        metrics.update({(metric_type,) + k: v for k, v in metrics_for_labels.items()})
        metrics_for_labels = pd.Series(metrics_for_labels)
        for m in metrics_for_labels.index.unique(level=1):
            metrics[(metric_type, 'macro', m)] = metrics_for_labels.xs(m, level=1).mean()
        if counts_for_all_labels is not None:
            metric_for_all_label = get_metrics_from_counts(counts_for_all_labels.to_dict())
            metrics.update({(metric_type, 'micro', k): v for k, v in metric_for_all_label.items()})
    # convert tuple keys to strings
    if merge_key_sep is not None:
        metrics = {merge_key_sep.join(k): v for k, v in metrics.items()}
    return metrics


def add_to_dict(dd: Union[defaultdict, int, float], d: [Dict, int, float]):
    """
    Add all numerical entries from the second dict to the respective entries from the first (create, if not already
    there) in a recursive manner.
    NOTE: THIS MODIFIES THE FIRST DICT!

    :param dd: first dict
    :param d: second dict
    :return: the modified first dict
    """

    if isinstance(dd, (int, float)):
        assert isinstance(d, (int, float)), f'd is not an instance of int or float, but of type: {type(d)}'
        return dd + d
    assert isinstance(dd, dict), f'dd is not an instance of dict, but of type: {type(dd)}'
    for k in list(d):
        if k in dd:
            dd[k] = add_to_dict(dd[k], d[k])
        else:
            dd[k] = copy.deepcopy(d[k])
    return dd


def unflatten_dict(d):
    res = {}
    for k, v in d.items():
        assert isinstance(k, (tuple, list)), f'key has to be either a list or tuple, but it is of type: {type(k)}'
        assert len(k) > 0, f'key has to contain at least one element'
        ref = res
        for e in k[:-1]:
            if e not in ref:
                ref[e] = {}
            ref = ref[e]
        ref[k[-1]] = v
    return res


def get_metrics(
        path_gold: str,
        path_predicted: str,
        out_dir: Optional[str] = None,
        type_blacklist: Optional[str] = None,
        file_whitelist: Optional[str] = None,
        exclude_labels_with_suffix: Optional[str] = None,
        confusion_key_undetected: str = "UNDETECTED",
        confusion_key_unassignable: str = "UNASSIGNABLE",
        merge_span_annotations_with_relation_label: Optional[str] = None,
        filter_gold_type: Optional[str] = None,
        filter_predicted_type: Optional[str] = None,
        filter_evaluation_task: Optional[str] = None,
        filter_output_dir: Optional[str] = None,
        use_gold_rels_for_merging: bool = False,

):
    """
    This method gathers count for true positives, false positives, false negatives of entity and relation prediction and
    then uses it to calculate metrics. Metric is then stored in a file 'path_out', if this is provided, or printed to
    the console otherwise.

    :param file_whitelist: comma separated string for filenames that is to be used for evaluation
    :param path_gold: location of true brat annotation files
    :param path_predicted: location of predicted brat annotation files
    :param out_dir: if provided, dump metrics and confusion matrices into this directory. Otherwise, just print to
        console.
    :param type_blacklist: comma separated string for entity or relation types that are not considered for final metric
        calculations TODO: is this also about labels? should this be renamed to label_blacklist?
    :param exclude_labels_with_suffix: All labels of BratSpanAnnotation containing this suffix needs to be excluded from
        metric calculation.
    :param confusion_key_undetected: The name of the row in the confusion matrices that hold the counts for gold
        annotations that were not detected at all (i.e. occur only in the gold annotations).
    :param confusion_key_unassignable: The name of the column in the confusion matrices that hold the counts for
        predicted annotations that are not assignable at all (i.e. occur only in the predicted annotations).
    :return: dict containing metrics for different annotations wrt different annotation labels and metric averaging
    types.
    """
    type_blacklist = type_blacklist.split(",") if type_blacklist is not None else []
    file_whitelist = file_whitelist.split(",") if file_whitelist is not None else None
    gold_annotations = read_brat_folder(path=path_gold, file_whitelist=file_whitelist)
    predicted_annotations = read_brat_folder(path=path_predicted, file_whitelist=file_whitelist)
    counts = defaultdict(lambda: 0)
    confusion_matrices = {}
    if filter_evaluation_task is not None:
        gold_folder_name = filter_gold_type+'-GOLD' if filter_gold_type != confusion_key_unassignable else filter_gold_type
        filter_output_dir = os.path.join(filter_output_dir,filter_evaluation_task,gold_folder_name,filter_predicted_type)
        if os.path.exists(filter_output_dir):
            logger.warning(f'filtered_output_dir={filter_output_dir} already exists and it will be overwritten')
            shutil.rmtree(filter_output_dir)
        os.makedirs(filter_output_dir)

    empty_collections = BratAnnotationCollection(base="", layers=[BratSpanAnnotationLayer(annotations=()),
                                                                  BratRelationAnnotationLayer(annotations=())])
    for nc in set(list(predicted_annotations) + list(gold_annotations)):
        predicted_collection = predicted_annotations.pop(nc, None)
        gold_collection = gold_annotations.pop(nc, None)
        current_counts, current_confusion_matrices, filtered_annotations = calculate_counts_for_collection(
            predicted_collection=predicted_collection or empty_collections,
            gold_collection=gold_collection or empty_collections,
            exclude_labels_with_suffix=exclude_labels_with_suffix,
            type_blacklist=type_blacklist,
            confusion_key_undetected=confusion_key_undetected,
            merge_span_annotations_with_relation_label=merge_span_annotations_with_relation_label,
            filter_gold_type=filter_gold_type,
            filter_predicted_type=filter_predicted_type,
            filter_evaluation_task=filter_evaluation_task,
            use_gold_rels_for_merging=use_gold_rels_for_merging,
        )

        counts = add_to_dict(counts, current_counts)
        confusion_matrices = add_to_dict(confusion_matrices, current_confusion_matrices)
        if filter_evaluation_task is not None and len(filtered_annotations) != 0:
            filtered_gold_annotations, filtered_predicted_annotations = zip(*filtered_annotations)
            if gold_collection is not None:
                text = gold_collection.base
            elif predicted_collection is not None:
                text = predicted_collection.base
            else:
                raise ValueError(f'identifier {nc} not found in gold and predicted annotation collection')
            filtered_predicted_annotations_without_none = tuple(
                a for a in filtered_predicted_annotations if a is not None)
            filtered_gold_annotations_without_none = tuple(a for a in filtered_gold_annotations if a is not None)
            if filter_evaluation_task in ['entity', 'entity_weak']:
                renamed_gold_annotations = [
                    BratSpanAnnotation(label=annotation.label + "-GOLD", target=annotation.target) for annotation in
                    filtered_gold_annotations_without_none]
                layers = [BratSpanAnnotationLayer(
                    annotations=renamed_gold_annotations + list(filtered_predicted_annotations_without_none))]
            elif filter_evaluation_task in ['rel', 'rel_weak']:
                entities = set()
                renamed_gold_annotations = [
                    BratRelationAnnotation(label=annotation.label + "-GOLD", target=annotation.target) for annotation in
                    filtered_gold_annotations_without_none]
                for rel in renamed_gold_annotations + list(filtered_predicted_annotations_without_none):
                    entities.add(rel.target.arguments[0])
                    entities.add(rel.target.arguments[1])
                layers = [BratSpanAnnotationLayer(annotations=tuple(entities)),
                          BratRelationAnnotationLayer(annotations=renamed_gold_annotations +
                                                                  list(filtered_predicted_annotations_without_none))]
            else:
                raise ValueError(f'unknown filter evaluation task: {filter_evaluation_task}')
            filtered_collection = BratAnnotationCollection(base=text, layers=layers)
            filtered_collection.to_files(os.path.join(filter_output_dir, nc))

    metrics = calculate_metrics(counts=counts, merge_key_sep=None)
    metrics_nested = unflatten_dict(metrics)
    metrics_flat = {"/".join(k): v for k, v in metrics.items()}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path_metrics = os.path.join(out_dir, "metrics_all.json")
        logger.info(f'Write output to {path_metrics}')
        json.dump(metrics_flat, open(path_metrics, 'w'), indent=2, sort_keys=True)

        for k, v in metrics_nested.items():
            if len(v) > 0:
                metrics_df = pd.DataFrame(v).T
                metrics_df = metrics_df.sort_index()
                # move micro, macro to the front
                for idx in ["macro", "micro"]:
                    if idx in metrics_df.index:
                        metrics_df = pd.concat([
                            metrics_df.loc[metrics_df.index == idx],
                            metrics_df.loc[metrics_df.index != idx]
                        ])
                # dumping as markdown requires the package "tabulate"
                metrics_df.to_markdown(os.path.join(out_dir, f'metrics_{k}.md'))
                # add the k to the index for better integration in final latex
                metrics_df.index = pd.MultiIndex.from_tuples(tuples=[(k, idx) for idx in metrics_df.index])
                metrics_df.to_latex(os.path.join(out_dir, f'metrics_{k}.tex'), float_format="%.2f")
            else:
                logger.warning(f'do not write metrics for "{k}" because it has no entries')

        for k, v in confusion_matrices.items():
            if len(v) > 0:
                with open(os.path.join(out_dir, f'confusion_{k}.json'), 'w') as f:
                    json.dump(v, f, sort_keys=True, indent=2)
                plot_confusion_matrix(v, filename=os.path.join(out_dir, f'confusion_{k}.png'),
                                      special_idx=confusion_key_undetected, special_column=confusion_key_unassignable)
            else:
                logger.warning(f'do not plot confusion matrix for "{k}" because it has no entries')

    else:
        logger.info(f'metrics:\n{json.dumps(metrics_nested, indent=2, sort_keys=True)}')
        logger.info(f'confusion matrices:\n{json.dumps(confusion_matrices, indent=2)}')

    return metrics


def get_type_prediction_distribution(
        gold_entities: AnnotationLayer,
        predicted_entities: AnnotationLayer,
        type_blacklist: List[str],
        key_undetected='UNDETECTED',
        key_unassignable='UNASSIGNABLE',
) -> Dict[str, Dict[str, int]]:
    """
    This method calculates true positives and distribution of false positive and false negative over different types.
    For example, in case of ENTITY type, it calculate true positive for "own_claim" and distribution of false positive
    over "data" and "background_claim". It creates a dictionary as
    { own_claim:{ own_claim:count, data:count, background_claim:count, UNDETECTED:count}, ... }
    UNDETECTED here means false negative, since these are gold labels which are not predicted.
    :param gold_entities: list of true Span or Relation AnnotationLayer
    :param predicted_entities: list of predicted Span or Relation AnnotationLayer
    :param type_blacklist: comma separated string for entity or relation types that are not considered for distribution
    :param key_undetected: name of the key that holds the counts for undetected gold entries
    :param key_unassignable: name of the key that holds the counts for unassignable predicted entries
    :return: updated dictionary with prediction distribution
    """

    type_dict = defaultdict(lambda: defaultdict(int))
    false_predictions = {}
    true_predicted = []
    for annotation in predicted_entities:
        if annotation.label in type_blacklist:
            continue
        if annotation in gold_entities:
            type_dict[annotation.label][annotation.label] += 1
            true_predicted.append(annotation.target)
        else:
            false_predictions[annotation.target] = annotation.label
    for annotation in gold_entities:
        if annotation.label in type_blacklist:
            continue
        if annotation.target in false_predictions:
            type_dict[annotation.label][false_predictions[annotation.target]] += 1
            del false_predictions[annotation.target]
        elif annotation.target not in true_predicted:
            type_dict[annotation.label][key_undetected] += 1
    for annotation, label in false_predictions.items():
        type_dict[key_unassignable][label] += 1
    return type_dict


def plot_confusion_matrix(type_matrix, filename: str, special_idx: str, special_column: str, indices_sorted: Optional[List[str]] = None):
    """
    This method create confusion matrix and saves it to given path with filename
    :param type_matrix: Dataframe with type prediction distribution
    :param filename: name of confusion matrix image
    :param special_idx: special index, this row will be moved to the end
    :param special_column: special column, this column will be moved to the end
    :return: None
    """
    if not isinstance(type_matrix, pd.DataFrame):
        type_matrix = pd.DataFrame(type_matrix)

    # create quadratic matrix, if entries are missing
    missing_index = (set(type_matrix.columns) | {special_idx}) - (set(type_matrix.index) | {special_column})
    missing_columns = (set(type_matrix.index) | {special_column}) - (set(type_matrix.columns) | {special_idx})
    # add missing columns
    type_matrix = type_matrix.T
    for idx in missing_index:
        type_matrix[idx] = 0
    type_matrix = type_matrix.T
    # add missing rows
    for column in missing_columns:
        type_matrix[column] = 0

    # fill missing values and convert to int
    type_matrix = type_matrix.fillna(0).astype(int)

    # move undetected to the end and sort other by name
    if indices_sorted is None:
        indices_sorted = sorted(set(type_matrix.index) - {special_idx, special_column})
    type_matrix = type_matrix.loc[indices_sorted + [special_idx], reversed(indices_sorted + [special_column])]
    type_matrix.columns = [col.replace("_", "\n") for col in type_matrix.columns]
    type_matrix.index = [col.replace("_", "\n") for col in type_matrix.index]

    plt.figure(figsize=(12, 12))
    ax = plt.subplot()
    # font_scale can be adjusted to change font size of values inside the heatmap grids
    sns.set(font_scale=4.0)
    cmap = sns.color_palette("Blues", as_cmap=True)
    #cmap = sns.color_palette("light:b", as_cmap=True)
    sns.heatmap(type_matrix, annot=True, fmt='g', ax=ax, cbar=False, cmap=cmap)
    # labelsize can be adjusted to change ticks label font size
    ax.tick_params(axis='both', which='major', labelsize=25)
    # size can be varied to change font size of xlabel and ylabel
    label_font = {'size': '30'}
    ax.set_xlabel('GOLD VALUES', fontdict=label_font)
    ax.set_ylabel('PREDICTED VALUES', fontdict=label_font)
    ax.set_yticklabels(labels=ax.get_yticklabels(), va="center")
    #plt.tight_layout()
    # use bbox_inches='tight', pad_inches=0.01 to remove whitespace around the plot
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    plt.show()
    plt.clf()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(
        description='Calculate precision, recall and f1 metrics for a predicted Brat collection with respect to a gold '
                    'Brat collection. The metrics are calculated for relations and entities, each in a exact and a '
                    'relaxed (weak) version. For the later, only the half of the spans (regarding the shorter one) '
                    'have to overlap to count for a match.')

    parser.add_argument(
        '--path_gold',
        help='path to the directory containing gold Brat annotations',
        type=str, required=True
    )

    parser.add_argument(
        '--path_predicted',
        help='path to the directory containing predicted Brat annotations',
        type=str, required=True
    )

    parser.add_argument(
        '--out_dir',
        help='if provided, save the final metrics (in json and markdown) and confusion matrices (in json and png) '
             'at this location',
        type=str, required=False
    )

    parser.add_argument(
        '--exclude_labels_with_suffix',
        help='all labels with given suffix will be excluded',
        type=str, required=False
    )

    parser.add_argument(
        '--type_blacklist',
        help='comma separated string for entity or relation types that are not considered for final metric calculations',
        type=str, required=False
    )

    parser.add_argument(
        '--file_whitelist',
        help='comma separated string for filenames that is to be used for evaluation',
        type=str, required=False
    )

    parser.add_argument(
        '--merge_span_annotations_with_relation_label',
        help='a relation label, that will identify the relations to use for merging entity spans, e.g. parts_of_same',
        type=str, required=False
    )

    parser.add_argument(
        '--filter_gold_type',
        help='gold annotation type',
        type=str, required=False
    )

    parser.add_argument(
        '--filter_predicted_type',
        help='predicted annotation type',
        type=str, required=False
    )

    parser.add_argument(
        '--filter_evaluation_task',
        help='evaluation tasks include ent, ent_weak, rel, rel_weak ',
        type=str, required=False
    )

    parser.add_argument(
        '--filter_output_dir',
        help='directory where filtered output will be stored',
        type=str, required=False
    )

    parser.add_argument(
        "--use_gold_rels_for_merging",
        help="merge the spans with gold relations instead of predicted relations",
        action='store_true',
    )

    args = parser.parse_args()
    metrics = get_metrics(**vars(args))
