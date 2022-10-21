import json
import logging
import random
import re
import time
from collections import Counter, defaultdict
from os import path
from typing import Dict, Sequence, Iterable, List, Optional, Tuple, cast

from allennlp.common import JsonDict
from allennlp.common.checks import ConfigurationError
from allennlp.data import Tokenizer, Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul, bio_tags_to_spans, bioul_tags_to_spans
from allennlp.data.dataset_readers.dataset_utils.span_utils import iob1_tags_to_spans, bmes_tags_to_spans, \
    TypedStringSpan
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from datasets import load_dataset, DatasetDict, concatenate_datasets
from numpy import ndarray
from overrides import overrides

from sam.dataset_reader.annotation import MultiSlice
from sam.dataset_reader.brat_annotation import BratSpanAnnotation, BratRelationAnnotation, Slice, \
    BratMultiSlice, BratSlice, BratRelation, BratAnnotationCollection, BratSpanAnnotationLayer, \
    BratRelationAnnotationLayer, BRAT_SPAN_LAYER_NAME, BRAT_RELATION_LAYER_NAME
from sam.models.basic_typed_classifier import BasicTypedClassifier

logger = logging.getLogger(__name__)


def counter_to_dict(counter: Counter, top_k: Optional[int] = None) -> Dict:
    """
    This method retrieve top_k (most common) items from the given Counter object. If number of items in given object is
    more than top_k, then it sums the values of remaining items and assign it to a key '...' of the dictionary in which
    top_k items are stored.
    :param counter: Counter containing key value pairs
    :param top_k: threshold value for most common element retrieval from the counter
    :return: dictionary containing top_k items
    """
    if not isinstance(counter, Counter):
        return counter
    if top_k is None:
        return dict(counter)
    d_top_k = dict(counter.most_common(top_k))
    if len(d_top_k) == len(counter):
        return d_top_k
    remaining_key = '...'
    assert remaining_key not in d_top_k, f'remaining key "{remaining_key}" not allowed in counter'
    d_top_k[remaining_key] = sum((v for k, v in counter.items() if k not in d_top_k))
    return d_top_k


def is_special(token: Token) -> bool:
    """
    Checks if given Token is special token such as [CLS] or [SEP] or not
    :param token: Token object containing text, idx etc. as attributes.
    :return: True if token is special token else false
    """
    return token.idx is None


def tags_to_spans(tags: List[str], encoding: str) -> List[TypedStringSpan]:
    """
    Given a sequence of tags and encoding scheme, this method returns the span from sequence of tags based on the
    encoding scheme.
    :param tags: Sequence of tags such as ['O','O','B-data','I-data','L-data','O',....]
    :param encoding: encoding scheme such as BIO, BIOUL etc
    :return: Identified spans (span label, span slice) from the sequence of tags
    """
    if encoding == "BIO":
        func = bio_tags_to_spans
    elif encoding == "IOB1":
        func = iob1_tags_to_spans
    elif encoding == "BIOUL":
        func = bioul_tags_to_spans
    elif encoding == "BMES":
        func = bmes_tags_to_spans
    else:
        raise ValueError(f"Unexpected coding scheme '{encoding}'. Use one of BIO, IOB1, BIOUL, or BMES.")
    spans = func(tags)
    return spans


def span_annotations_to_tags(span_annotations: Dict[str, BratSpanAnnotation], base_sequence: TextField, namespace: str,
                             text_id: str, coding_scheme: str, use_annotation_ids_as_labels: bool = False,
                             offset: int = 0, valid_token_slice: Slice = None) \
        -> Tuple[SequenceLabelField, List[Tuple[BratSpanAnnotation, str]]]:
    """
    Given sequence of tokens and its respective span annotations, this method create sequence of tags for sequence of
    tokens based on the coding_scheme (BIO, BIOUL etc). It returns SequenceLabelField containing tags and also returns
    annotations which are not valid as errors.
    :param span_annotations: dictionary containing annotation id (eg: T1, T2) as keys and BratSpanAnnotation as value.
    :param base_sequence: TextField containing tokens and token_indexers
    :param namespace: label namespace
    :param text_id: file name with text slice attached eg: A1[2327:6166]
    :param coding_scheme: encoding scheme to be used for tags
    :param use_annotation_ids_as_labels: boolean value to specify if annotation id (eg: Arg1, Arg2) to be used as labels
    :param offset: offset so that correct tags and tokens is mapped in given fixed size input
    :param valid_token_slice: to verify if given annotation is within this given valid_token_slice or not. This is done
    since we have fixed token window size for Relation extraction instances
    :return: a tuple containing SequenceLabelField and errors. SequenceLabelField contains tags and error contains
    annotations whose slices are not valid_token_slice
    """
    # create IOB1 encoding
    tags = ['O'] * len(base_sequence)
    errors = []
    for annot_id, annot in span_annotations.items():
        label = annot.label if not use_annotation_ids_as_labels else annot_id
        # for slice in annot.target.slices:
        _start = annot.target.slices[0].start - offset
        _end = annot.target.slices[-1].end - offset
        # TODO: check why this is happening!
        if len(annot.target.slices) > 1:
            logger.warning(
                f'annotation {annot_id} ({text_id.split("/")[-1]}): separated spans ({annot.target.slices}), '
                f'use maximal overlap: {base_sequence[_start:_end]}')

        if valid_token_slice is not None and (_start not in valid_token_slice or (_end - 1) not in valid_token_slice):
            errors.append((annot, f'target [{_start}:{_end}] not in valid_token_slice: {valid_token_slice}'))
            continue

        previous_tags = tags[_start:_end]
        assert previous_tags == ['O'] * len(previous_tags), \
            f'annotation {annot_id}: tags already set [{previous_tags}] ({text_id})'
        # create IOB1 encoding
        tags[_start] = f'B-{label}'
        tags[_start + 1:_end] = [f'I-{label}'] * (len(previous_tags) - 1)

    # Recode the labels if necessary.
    if coding_scheme == "BIOUL":
        coded_adu = (
            to_bioul(tags, encoding="IOB1")
            if tags is not None
            else None
        )
    else:
        # the default IOB1
        coded_adu = tags

    return SequenceLabelField(coded_adu, base_sequence, namespace), errors


def tags_to_span_annotations(tags: SequenceLabelField, tokens: TextField, text: str, encoding: str,
                             label_suffix: str = '', split_kwargs: Optional[Dict] = None) \
        -> List[BratSpanAnnotation]:
    """
    Given sequence of tags, tokens and encoding scheme, this method converts it into BratSpanAnnotations.
    :param tags: SequenceLabelField containing sequence of tags.
    :param tokens: TextField containing list of tokens and token_indexers.
    :param text: original text span
    :param encoding: encoding scheme in which sequence of tags is encoded with.
    :param label_suffix: suffix to be used in labels, such as '-GOLD', to distinguish from predicted labels
    :param split_kwargs: split the target slice based on the pattern provided in this parameter
    :return: list of span annotations found in given sequence of tags
    """
    span_annots = [
        BratSpanAnnotation(
            # set no id
            label=label + label_suffix,
            target=BratMultiSlice(
                slices=(BratSlice(start=tokens[start].idx, end=tokens[end].idx_end),),
                base=text
            )
        )
        for i, (label, (start, end)) in enumerate(tags_to_spans(tags.labels, encoding=encoding))
    ]
    # remove newlines, for instance, from span annotations
    if split_kwargs is not None:
        span_annots = [span_annot.split_target_slices(**split_kwargs) for span_annot in span_annots]

    return span_annots


class RelationArgumentError(Exception):
    pass


@DatasetReader.register("brat")
class BratDatasetReader(DatasetReader):
    """
    This class inherits DatasetReader class and it aims to create data instances that is to be used for ADU
    identification and relation extraction. First it loads dataset based on the dataset_splits or split mentioned in
    dataset file_path. Fot training with cross validation, it combines train and dev data splits and then divide it
    based on num_shards. Each instance in the dataset is a file which is converted into BratAnnotations and then further
    divided into span and relation annotations. Each file is divided into small sub sections based on split_pattern.
    From each subsection noisy text is removed based on delete_pattern and then remaining text is linked with respective
    span annotation. Now in case of ANNOTATION_TYPE=ENTITY_TYPE, this subsection is immediately converted into a
    training instance containing tokens from text, respective tags and some metadata. For ANNOTATION_TYPE=RELATION_TYPE,
    first we extract corresponding relations for given subsection from set of all relations in the file and then for
    each relation in this subset, we create an instance if distance between arguments is less than max_argument_distance.
    Along with this we create a reverted relation for each non symmetric relation.
    We also create more relation candidates by using all combination of entity pairs which are already not a relation
    and have distance between them less than max_argument_distance. These candidates are added based on
    add_relation_portion parameter. If this parameter is a negative then we add all candidates as instance of relation
    type otherwise we only add portion (add_relation_portion*num of original relations) of these candidates.

    # Parameters

    token_indexers : We use this to define the input representation for the text.
    tokenizers : We use this to splits text into tokens
    instance_type : We define two types of instance ENTITY_TYPE and RELATION_TYPE.
    entity_coding_scheme: It is coding scheme used to encode ADU tokens in token sequence, Default is "IOB1"
    relation_argument_coding_scheme: It is coding scheme used to encode tokens that represent relation arguments in
    token sequence, Default is "IOB1"
    label_namespace: Namespace for labels, default is "labels".
    none_label: Unique notation for None label, default is "NONE"
    tag_namespace: Namespace for tags, default is "adu_tags".
    type_namespace: Namespace for type, default is "type_tags"
    dataset_splits: dictionary containing split information for data, key represent split name and value represent range
    of data for that particular split.
    all_data_split_name: it ensures that if no data split is provided then all dataset is stored into the name
    specified by this parameter, default is "train"
    relation_argument_names: It defines the reference names for the arguments in a relation. default is ("Arg1", "Arg2")
    add_reverted_relations: boolean value to decide if we want to add reversed(!) (unfortunately the parameter name is misspelled) relations as data instance or not.
    symmetric_relations: sequence of strings that specifies the names of symmetric relations

    add_negative_relations_portion: same as add_relation_portion, used for backward compatibility

    add_relations_portion: portion of relation candidates to be added as relation instance.
    use_all_gold_relations: boolean value to decide if we want to add all gold relations as instance or not. If it is
    true then even distance between arguments of relation is greater than max_argument_distance, relation will be
    considered as valid data instance. By default it is False
    negative_examples_max_distance: same as max_argument_distance, used for backward compatibility
    max_argument_distance: Maximum token level inner distance allowed between arguments of a relation. Inner distance is
    defined as distance between end of first argument and beginning of second argument.
    token_window_size: this is used to fix number of tokens to be used for a relation instance. Number of tokens in a
    relation instance can be less than or equal to this value. Candidate relations which do not fit within this window
    are discarded.
    delete_pattern: this pattern is used to remove noisy text from the text files used to create instances.
    split_pattern: this pattern is used to split text file instance to subsections.
    show_gold: boolean value to decide if we want to have gold annotation in prediction output, default is False
    show_prediction: boolean value to decide if we want to have prediction annotation in prediction output, default is
    True
    debug_text_id: text_id is the name for subsection of file created using split pattern, we can pass that name to this
    parameter, if we want to debug that subsection. Only text_ids mentioned in this parameter will be handled other will
    be ignored.
    num_shards: Number of subsets of the dataset to be created for cross validation training
    dev_shard_idx: Current index of the shards that is to be used as dev set.
    cross_fold_splits: split names for the cross fold. All splits mentioned in this variable is used as cross fold data,
    default is ("train", "dev") i.e train and dev combined is used as cross fold data. Depending on num_shards and
    dev_shard_idx cross fold data is divided into train and dev.
    relation_type_blacklist: list of relations (label) which is not be used as data instance.

    """

    ENTITY_TYPE = "T"
    RELATION_TYPE = "R"
    KNOWN_ANNOTATION_TYPES = [ENTITY_TYPE, RELATION_TYPE]
    RELATION_TYPE_DEFAULT_PLACEHOLDER = '_default'

    def __init__(
        self,
        tokenizers: Dict[str, Tokenizer] = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        instance_type: str = ENTITY_TYPE,
        feature_labels: Sequence[str] = (),
        entity_coding_scheme: str = "IOB1",
        relation_argument_coding_scheme: str = "IOB1",
        label_namespace: str = "labels",
        none_label: str = BasicTypedClassifier.DEFAULT_NONE_LABEL,
        tag_namespace: str = BasicTypedClassifier.DEFAULT_TAG_NAMESPACE,
        type_namespace: str = BasicTypedClassifier.DEFAULT_TYPE_NAMESPACE,
        dataset_splits: Optional[Dict[str, str]] = None,
        # all_data_split_name should be the same as referenced by subdirectory_mapping in brat config
        all_data_split_name: Optional[str] = "train",
        relation_argument_names: Dict[str, Tuple[str, str]] = {RELATION_TYPE_DEFAULT_PLACEHOLDER: ('Arg1', 'Arg2')},
        add_reverted_relations: bool = False,
        symmetric_relations: Sequence[str] = (),
        add_negative_relations_portion: float = 0.0,
        add_relations_portion: float = 0.0,
        use_all_gold_relations: bool = False,
        negative_examples_max_distance: Optional[int] = None,
        max_argument_distance: Optional[int] = None,
        token_window_size: Optional[int] = None,
        delete_pattern: str = None,
        split_pattern: str = None,
        show_gold: bool = False,
        show_prediction: bool = True,
        debug_text_id: str = None,
        num_shards: Optional[int] = None,
        dev_shard_idx: Optional[int] = None,
        cross_fold_splits: Tuple[str] = ("train", "dev"),
        relation_type_blacklist: Sequence[str] = (),
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=True,
                         manual_multiprocess_sharding=True, **kwargs)
        self._tokenizers = tokenizers or {"tokens": WhitespaceTokenizer()}
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if instance_type is not None and instance_type not in BratDatasetReader.KNOWN_ANNOTATION_TYPES:
            raise ConfigurationError("unknown tag label type: {}".format(instance_type))
        for label in feature_labels:
            if label not in BratDatasetReader.KNOWN_ANNOTATION_TYPES:
                raise ConfigurationError("unknown feature label type: {}".format(label))

        self.instance_type = instance_type
        self.label_namespace = label_namespace
        self.tag_namespace = tag_namespace
        self.type_namespace = type_namespace
        self.none_label = none_label
        self.relation_argument_names = {k: tuple(v) for k, v in relation_argument_names.items()}
        self.add_reverted_relations = add_reverted_relations
        self.symmetric_relations = symmetric_relations
        # backwards compatibility
        self.add_relations_portion = add_relations_portion or add_negative_relations_portion
        self.use_all_gold_relations = use_all_gold_relations
        # backwards compatibility
        self.max_argument_distance = max_argument_distance or negative_examples_max_distance
        self.token_window_size = token_window_size
        self.dataset_splits = dataset_splits
        self.all_data_split_name = all_data_split_name
        self.delete_pattern = delete_pattern
        self.split_pattern = split_pattern
        self.show_gold = show_gold
        self.show_prediction = show_prediction
        self.debug_text_id = debug_text_id
        self.num_shards = num_shards
        self.cross_fold_splits = cross_fold_splits
        self.dev_shard_idx = dev_shard_idx
        self._stats = {}
        self.relation_type_blacklist = relation_type_blacklist
        self.entity_coding_scheme = entity_coding_scheme
        self.relation_argument_coding_scheme = relation_argument_coding_scheme

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        This method loads data via load_dataset from huggingface dataset with the script from ./dataset_scripts/brat.py.
        Data is located at given file_path. Every file is divided into subsections and then data instance
        (ENTITY_TYPE and RELATION_TYPE) from these sections are created. Each subsection can have zero or more data
        instances. It also handles data sharding for cross validation training.

        :param file_path: If file_path points to a single json file, this will be loaded and content passed as kwargs to
         load_dataset. Otherwise, load_dataset will simply called with url=file_path. file_path can be appended by
         '@SPLIT', i.e @SPLIT is optional. if present, it is assumed that the loaded dataset contains a split with name
        SPLIT and only that data will be used.
        :return: iterator of data instance which can be used for ADU identification or Relation extraction
        """
        self._stats = {'labels': Counter(),
                       'removed_text': Counter(),
                       'lengths': Counter(),
                       'discarded_entities': [],
                       'relations': {'available': 0, 'beyond_text_split': defaultdict(lambda: 0)},
                       'time': defaultdict(lambda: 0),
                       }
        start_read = time.time()
        if self.add_relations_portion != 0.0:
            self._stats['relation_candidates'] = {'available': 0, 'added': 0, 'relation_argument_error': 0}
        if self.max_argument_distance is not None:
            self._stats['relations']['beyond_max_distance'] = defaultdict(lambda: 0)
            if self.add_relations_portion != 0.0:
                self._stats['relation_candidates']['beyond_max_distance'] = 0

        # "@" is in file we assume that we have a certain split
        split = None
        if '@' in path.basename(file_path):
            file_parts = file_path.split('@')
            split = file_parts[-1]
            file_path = '@'.join(file_parts[:-1])

        dataset_kwargs = {}
        # backwards compatibility
        if file_path.endswith('.py'):
            file_path = './dataset_scripts/sciarg.json'
            logger.warning(f'Loading a special python script is deprecated. We instead load the brat.py script with '
                           f'parameters from {file_path}.')

        assert path.exists(file_path), f'file_path does not point to a local file or directory: {file_path}'
        # if file_path points to a json file, load the content and pass as kwargs to load_dataset
        if file_path.endswith('.json'):
            assert path.exists(file_path) and path.isfile(file_path), \
                f'brat dataset reader config file not found: {file_path}'
            dataset_kwargs = json.load(open(file_path))
            file_path = ''
        elif path.isdir(file_path):
            file_path = file_path
        else:
            raise ValueError(
                f'unknown file_path type: "{file_path}". Expected a json file (Brat config) or a directory '
                f'(which should contain Brat annotations, i.e. .txt and .ann files)'
            )
        dataset_kwargs['path'] = "./dataset_scripts/brat.py"

        # # since datasets 1.6 this is required to enable caching,
        # # see https://github.com/huggingface/datasets/issues/2387
        # dataset_kwargs['keep_in_memory'] = False

        # currently, the file_path is discarded (we set the script_path later on to fixed value)
        if file_path.strip() != "":
            dataset_kwargs['url'] = file_path
        if self.dataset_splits is not None:
            def get_split(split_string, default_split):
                m = re.match(r'^([^\[]+)\[([^]]+)]$', split_string)
                if m is not None:
                    _split, _indices = m.groups()
                else:
                    _split, _indices = default_split, split_string
                return f'{_split}[{_indices}]'

            dataset = DatasetDict({k: load_dataset(split=get_split(v, default_split=self.all_data_split_name),
                                                   **dataset_kwargs)
                                   for k, v in self.dataset_splits.items()})
        else:
            dataset = load_dataset(**dataset_kwargs)

        split = split or self.all_data_split_name
        if self.num_shards is not None:
            assert self.dev_shard_idx is not None, \
                f'dev_shard_idx has to be set if num_folds is set which implies cross fold evaluation'
            logger.info(f"execute cross fold training/evaluation with {self.num_shards} "
                        f"folds (current dev shard index: {self.dev_shard_idx})")
            # Use data from all splits mentioned in cross_fold_splits. Per default, use only train and dev, but not test
            cross_fold_data = concatenate_datasets([dataset[s] for s in self.cross_fold_splits])
            # split into shards
            dataset = DatasetDict({
                "dev": cross_fold_data.shard(num_shards=self.num_shards, index=self.dev_shard_idx, contiguous=True),
                "train": concatenate_datasets([
                    cross_fold_data.shard(num_shards=self.num_shards, index=i, contiguous=True)
                    for i in range(self.num_shards) if i != self.dev_shard_idx
                ])
            })

        assert split in dataset, \
            f'requested dataset split "{split}" not available. Use one of {list(dataset.keys())} ' \
            f'or modify the dataset_splits: {self.dataset_splits}'
        dataset = dataset[split]

        for instance in self.shard_iterable(dataset):
            brat_annotations = BratAnnotationCollection.from_arrow(
                instance=instance,
                relation_argument_names_dict=self.relation_argument_names,
                relation_argument_names_default_key=BratDatasetReader.RELATION_TYPE_DEFAULT_PLACEHOLDER
            )
            all_span_annotations: BratSpanAnnotationLayer = brat_annotations[BRAT_SPAN_LAYER_NAME]
            all_relation_annotations: BratRelationAnnotationLayer = brat_annotations[BRAT_RELATION_LAYER_NAME]
            # TODO: what's that for? instance['context'] seems to be a single string
            all_text = ''.join(instance['context'])
            fn_short = instance['file_name']
            used_relation_annotation_names = []  # stores relation names such as R1, R2, R3,...
            beyond_max_distance_rel_names = []  # relations with distance between argument more than max distance
            if self.split_pattern is not None:
                # Split text based on split pattern. Get span annotations and span offset from each split.
                slices = BratSlice.from_base_with_split_pattern(base=all_text, pattern=self.split_pattern,
                                                                destructive=False)
                slices = [s for s in slices if s.start != s.end]
                partition_layers, partition_backrefs = all_span_annotations.partition_with_slices(slices=slices)
                partition_texts = [all_text[s.start:s.end] for s in slices]
                partition_offsets = [s.start for s in slices]
                partitions = zip(partition_texts, partition_offsets, partition_layers)
            else:
                partitions = [(all_text, 0, all_span_annotations)]

            for text, partition_offset, span_annotations in partitions:
                # For each partition of file, create instance from respective span annotation

                # remove noisy text
                if self.delete_pattern is not None:
                    slices = BratSlice.from_base_with_split_pattern(base=text, pattern=self.delete_pattern,
                                                                    destructive=True)

                    # collect statistics
                    self._stats['removed_text'].update([text[_slice.end:slices[i+1].start]
                                                        for i, _slice in enumerate(slices[:-1])
                                                        if text[_slice.end:slices[i+1].start] != ''])

                    # filter empty spans
                    slices = [slice for slice in slices if slice.start != slice.end]

                    _layers, _backrefs = span_annotations.partition_with_slices(slices=slices)
                    _texts = [text[s.start:s.end] for s in slices]
                    span_annotations, annot_backrefs = BratSpanAnnotationLayer.merge(layers=_layers, bases=_texts)
                    text = ''.join(_texts)
                    slices = [slice.shift(partition_offset) for slice in slices]
                else:
                    slices = [Slice(start=partition_offset, end=partition_offset+len(text))]

                text_id = fn_short
                if len(slices) > 1 or slices[0].start != 0 or slices[-1].end != len(all_text):
                    # Assign text id based on the span slice
                    text_id += '[' + ';'.join([f'{slice.start}:{slice.end}' for slice in slices]) + ']'

                if self.debug_text_id is not None and text_id != self.debug_text_id:
                    logger.warning(f'skip input with text_id: {text_id} (debug_text_id: {self.debug_text_id})')
                    continue

                tokens = self._tokenizers['tokens'].tokenize(text)
                self._stats['lengths'].update([len(tokens)])
                span_annotations, entities_removed = span_annotations.remap(
                    mapping=[Slice(start=t.idx, end=t.idx_end) for t in tokens],
                    #base=tokens
                    base=None  # set to None, otherwise results in excessive hashing times
                )
                if len(entities_removed) > 0:
                    errors_str = '\n'.join(
                        [f'{entity}: {e}. SKIP THIS entity! ({annot_id} from {text_id})' for annot_id, (entity, e) in
                         entities_removed.items()])
                    logger.warning(
                        f'skipped {len(entities_removed)} out of {len(span_annotations)} entities ({len(entities_removed) * 100 // len(span_annotations)}%) from {text_id}:\n{errors_str}')
                    self._stats['discarded_entities'].extend(
                        [f'{entity}: {e}' for annot_id, (entity, e) in entities_removed.items()])

                if self.instance_type == BratDatasetReader.ENTITY_TYPE:
                    self._stats['labels'].update((entity.label for entity in span_annotations))
                    instance = self.text_to_instance(text=text, entities=span_annotations.as_name_dict(),
                                                     text_id=text_id, tokens=tokens)
                    yield instance
                elif self.instance_type == BratDatasetReader.RELATION_TYPE:
                    rel_entity_pairs = {}  # stores argument distance for pair of arguments
                    span_annotations_dict = span_annotations.as_hash_dict()
                    # take only relations where all entities are in the current partition.
                    # use name as reference because the hash has changed since the base was modified (split/deleted)
                    relation_annotations = all_relation_annotations.select_subset_with_valid_targets_via_names(
                        span_annotations, link=True
                    )
                    for r in relation_annotations:
                        # for each relation found in partition of file, create an instance
                        if r.label in self.relation_type_blacklist:
                            continue

                        start = time.time()

                        self._stats['labels'].update([r.label])
                        args = tuple(r.target.arguments._asdict().values())
                        span_combination = (args[0].hash(), args[1].hash())
                        argument_distance = MultiSlice.distance(args[0].target, args[1].target)
                        rel_entity_pairs[span_combination] = argument_distance
                        if not self.use_all_gold_relations and self.max_argument_distance is not None \
                                and argument_distance > self.max_argument_distance:
                            self._stats['relations']['beyond_max_distance'][r.label] += 1
                            beyond_max_distance_rel_names.append(r._name)
                            end = time.time()
                            self._stats['time']['default_relations'] += end - start
                            continue

                        try:
                            instance = self.text_to_instance(text=text, entities=span_annotations.as_name_dict(),
                                                             rel=r, text_id=text_id, tokens=tokens)
                            used_relation_annotation_names.append(r._name)
                            end = time.time()
                            self._stats['time']['default_relations'] += end - start
                        except AssertionError as e:
                            logger.warning(e)
                            end = time.time()
                            self._stats['time']['default_relations'] += end - start
                            continue

                        yield instance
                        if self.add_reverted_relations:
                            start = time.time()
                            args_list = list(r.target.arguments._asdict().items())
                            assert len(args_list) == 2, f'expected two relation arguments, but found: {args_list}'
                            args_rev = {args_list[0][0]: args_list[1][1], args_list[1][0]: args_list[0][1]}
                            arg_rev1, arg_rev2 = tuple(args_rev.values())
                            span_combination_reverted = (arg_rev1.hash(), arg_rev2.hash())
                            if span_combination_reverted in rel_entity_pairs:
                                logger.warning(f'reverted span combination ({fn_short}:{span_combination_reverted}) '
                                               f'already in collected span combinations')
                                end = time.time()
                                self._stats['time']['reverted_relations'] += end - start
                            else:
                                rel_entity_pairs[span_combination_reverted] = MultiSlice.distance(arg_rev1.target, arg_rev2.target)
                                label = r.label if r.label in self.symmetric_relations else f'{r.label}_rev'
                                self._stats['labels'].update([label])
                                r_rev = BratRelationAnnotation(_name=f'{r._name}-REV', label=label,
                                                               target=BratRelation.from_dict(**args_rev))
                                instance = self.text_to_instance(text=text, entities=span_annotations.as_name_dict(),
                                                                 rel=r_rev, text_id=text_id, tokens=tokens)
                                end = time.time()
                                self._stats['time']['reverted_relations'] += end - start
                                yield instance

                    if abs(self.add_relations_portion) > 0.0:
                        start = time.time()
                        entity_pairs_candidate = []  # list of entity pairs which are not already forming a relation
                        neg_distances_human = []  # list of overlapping entity pairs

                        for h1, v1 in span_annotations_dict.items():
                            for h2, v2 in span_annotations_dict.items():
                                if h1 != h2 and (h1, h2) not in rel_entity_pairs:
                                    distance = MultiSlice.distance(v1.target, v2.target)
                                    if distance < 0:
                                        neg_distances_human.append((v1._name, v2._name))
                                    entity_pairs_candidate.append(((h1, h2), distance))

                        if len(neg_distances_human) > 0:
                            logger.warning(f'some entity combination candidates in instance {text_id} have negative '
                                           f'distances (overlap): {neg_distances_human}')

                        n_pairs_available = len(entity_pairs_candidate)
                        self._stats['relation_candidates']['available'] += n_pairs_available
                        if self.max_argument_distance is not None:
                            entity_pairs_candidate = [
                                (pair, d) for pair, d in entity_pairs_candidate
                                if d <= self.max_argument_distance
                            ]
                            self._stats['relation_candidates']['beyond_max_distance'] += n_pairs_available - len(
                                entity_pairs_candidate)

                        if self.add_relations_portion > 0.0:
                            # sort by distances NOT ANYMORE!
                            # entity_pairs_candidate = sorted(entity_pairs_candidate,
                            #                                            key=lambda kd: kd[1])
                            random.shuffle(entity_pairs_candidate)

                            n_add = int(len(rel_entity_pairs) * self.add_relations_portion)
                            entity_pairs_add = entity_pairs_candidate[:n_add]
                        else:
                            entity_pairs_add = entity_pairs_candidate

                        rel_arg_keys = self.relation_argument_names[BratDatasetReader.RELATION_TYPE_DEFAULT_PLACEHOLDER]
                        end = time.time()
                        self._stats['time']['relation_candidates_init'] += end - start
                        #logger.info(f'add {len(entity_pairs_add)} negative relation instances')
                        for i, ((h1, h2), d) in enumerate(entity_pairs_add):
                            start = time.time()
                            t1 = span_annotations_dict[h1]
                            t2 = span_annotations_dict[h2]
                            r_canidate = BratRelationAnnotation(_name=f'R{i}-CANDIDATE', label=self.none_label,
                                                                target=BratRelation.from_dict(**{rel_arg_keys[0]: t1,
                                                                                                 rel_arg_keys[1]: t2}))
                            try:
                                instance = self.text_to_instance(text=text, entities=span_annotations.as_name_dict(),
                                                                 rel=r_canidate, text_id=text_id, tokens=tokens)
                            except RelationArgumentError as e:
                                self._stats['relation_candidates']['relation_argument_error'] += 1
                                end = time.time()
                                self._stats['time']['relation_candidates'] += end - start
                                continue
                            self._stats['relation_candidates']['added'] += 1
                            end = time.time()
                            self._stats['time']['relation_candidates'] += end - start

                            yield instance
                    #self._stats['distances'].extend(rel_entity_pairs.values())
                else:
                    raise AttributeError(f'unknown tag_label "{self.instance_type}"')
            if self.instance_type == BratDatasetReader.RELATION_TYPE:
                for r in all_relation_annotations:
                    if r._name not in used_relation_annotation_names + beyond_max_distance_rel_names:
                        self._stats['relations']['beyond_text_split'][r.label] += 1
                self._stats['relations']['available'] += len(all_relation_annotations)

        self._stats['time']['read'] = time.time() - start_read
        # NOTE: for counter objects show only the 10 most common entries
        self._stats = {k: counter_to_dict(v, top_k=10) for k, v in self._stats.items()}
        # use warning to still show up in --silent mode
        logger.warning(f'\ndataset stats:\n{json.dumps(self._stats, indent=2)}')

    def text_to_instance(  # type: ignore
        self,
        text: str,
        entities: Dict[str, BratSpanAnnotation] = None,
        rel: BratRelationAnnotation = None,
        text_id: str = None,
        tokens: Optional[List[Token]] = None,
    ) -> Instance:
        """
        Given sequence of tokens from a text, this method returns an instance that is to be used for training. For ADU,
        Instance contains tokens, tags and some meta data. For REL, Instance consist of tokens (of fixed length), tags
        containing entity tags, meta data, types containing Argument tags and label specifying relation.
        :param text: Raw text from which tokens are extracted
        :param entities: dictionary containing annotation id (eg: T1, T2) as keys and BratSpanAnnotation as value.
        :param rel: RelationAnnotation containing label for relation and targets of relation (two arguments)
        :param text_id: file name with text slice attached eg: A1[2327:6166]
        :param tokens: sequence of words taken from the text
        :return: Instance containing tokens, tags and metadata.
        """

        if tokens is None:
            assert entities is None and rel is None, \
                f'if text is tokenized inside text_to_instance, no entities and relation is allowed'
            tokens = self._tokenizers['tokens'].tokenize(text)

        slice_is_text = None
        shift_tokens = 0
        if self.token_window_size is not None and rel is not None:
            # Here we make sure that token_window_size constraint is fulfilled by span of both arguments. We do it by
            # finding relation center and then using tokens around the center such that total tokens is equal to
            # token_window_size
            arg1_center = rel.target.arguments._asdict()['Arg1'].target.center_position()
            arg2_center = rel.target.arguments._asdict()['Arg2'].target.center_position()
            rel_center = round(arg1_center / 2 + arg2_center / 2)
            text_slice_source = Slice(start=round(rel_center - self.token_window_size / 2 + 0.1),
                                      end=round(rel_center + self.token_window_size / 2 + 0.1))
            assert text_slice_source.end - text_slice_source.start == self.token_window_size, f'wrong calculations'
            if text_slice_source.end > len(tokens):
                text_slice_source = text_slice_source.shift(len(tokens) - text_slice_source.end)
            if text_slice_source.start < 0:
                text_slice_source = text_slice_source.shift(-text_slice_source.start)
            n = min(len(text_slice_source), len(tokens))
            text_slice_source = Slice(start=text_slice_source.start, end=text_slice_source.start + n)

            new_tokens = tokens[text_slice_source.start:text_slice_source.end]
            shift_tokens = text_slice_source.start
            slice_is_text = Slice(start=0, end=len(new_tokens))

            if len(tokens) > 0:
                if is_special(tokens[0]):
                    new_tokens[0] = tokens[0]
                    slice_is_text = slice_is_text.shift_start(1)
                if is_special(tokens[-1]):
                    new_tokens[-1] = tokens[-1]
                    slice_is_text = slice_is_text.shift_end(-1)

            tokens = new_tokens

        sequence = TextField(tokens)
        instance_fields: Dict[str, Field] = {
            "tokens": sequence,
            "metadata": MetadataField({"words": [x.text for x in tokens], "text": text, "text_id": text_id}),
        }

        if entities is not None:
            if rel is not None:
                arg_annots: Dict[str, BratSpanAnnotation] = rel.target.arguments._asdict()
                arg_tags, arg_errors = span_annotations_to_tags(
                    span_annotations=arg_annots, base_sequence=sequence, namespace=self.type_namespace, text_id=text_id,
                    coding_scheme=self.relation_argument_coding_scheme, use_annotation_ids_as_labels=True,
                    offset=shift_tokens, valid_token_slice=slice_is_text)
                if len(arg_errors) > 0:
                    raise RelationArgumentError(f'errors when creating argument type tags from relation arguments: {arg_errors}')

                instance_fields["types"] = arg_tags
                instance_fields["label"] = LabelField(rel.label, label_namespace=self.label_namespace)

            entity_tags_namespace = self.label_namespace if self.instance_type == BratDatasetReader.ENTITY_TYPE else self.tag_namespace
            entity_tags, entity_errors = span_annotations_to_tags(
                span_annotations=entities, base_sequence=sequence, namespace=entity_tags_namespace,
                text_id=text_id, coding_scheme=self.entity_coding_scheme, offset=shift_tokens,
                valid_token_slice=slice_is_text)
            instance_fields["tags"] = entity_tags

        return Instance(instance_fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"].token_indexers = self._token_indexers

    def prediction_to_labeled_instance(self, instance: Instance, outputs: JsonDict) -> Instance:
        """
        This method duplicates the current instance and add a field ,predicted tags or label, to new instance depending
        upon type of instance (ADU or REL).
        :param instance: Instance of dataset reader
        :param outputs: Dictionary containing predicted tags or labels for given instance.
        :return: list of labelled instances
        """
        start = time.time()
        # entity case
        if self.instance_type == BratDatasetReader.ENTITY_TYPE:
            instance.add_field("tags_predicted", SequenceLabelField(outputs["tags"], instance['tokens']))
        elif self.instance_type == BratDatasetReader.RELATION_TYPE:
            instance.add_field("label_predicted", LabelField(outputs["label"]))
        else:
            raise ConfigurationError(f"unknown instance type: {self.instance_type}. "
                                     f"Use one of: {', '.join(BratDatasetReader.KNOWN_ANNOTATION_TYPES)}")
        self._stats['time']['predictions_to_labeled_instances'] += time.time() - start
        return instance

    def labeled_instance_to_json(self, labeled_instance: Instance, obj_store: Optional[Dict] = None) -> List[JsonDict]:
        """
        This method converts labeled instance into annotations and then hashes annotation into a dictionary mapping hash
        to their content (Annotation label, target, text, etc).
        :param labeled_instance: Labeled dataset instance containing prediction for tags or labels
        :param obj_store: calculated objects are stored by their hashes and this allows for deduplication
        :return: list of annotations in typeddict
        """
        start = time.time()
        _labeled_instance = {
            'tokens': cast(TextField, labeled_instance['tokens']),
            'metadata': cast(MetadataField, labeled_instance['metadata']),
        }
        if 'tags' in labeled_instance:
            _labeled_instance['tags'] = cast(SequenceLabelField, labeled_instance['tags'])
        if 'tags_predicted' in labeled_instance:
            _labeled_instance['tags_predicted'] = cast(SequenceLabelField, labeled_instance['tags_predicted'])
        if 'label' in labeled_instance:
            _labeled_instance['label'] = cast(LabelField, labeled_instance['label'])
        if 'label_predicted' in labeled_instance:
            _labeled_instance['label_predicted'] = cast(LabelField, labeled_instance['label_predicted'])
        if 'types' in labeled_instance:
            _labeled_instance['types'] = cast(SequenceLabelField, labeled_instance['types'])
        span_annots = []

        split_kwargs = {'pattern': r"\s*\n\s*", 'use_regex': True, 'destructive': True}
        destructive = True
        if self.show_gold or self.instance_type != BratDatasetReader.ENTITY_TYPE:
            span_annots = tags_to_span_annotations(
                tags=_labeled_instance['tags'],
                tokens=_labeled_instance['tokens'],
                text=_labeled_instance['metadata']['text'],
                encoding=self.entity_coding_scheme,
                label_suffix='-GOLD' if (self.show_gold and self.show_prediction) and self.instance_type == BratDatasetReader.ENTITY_TYPE else '',
                split_kwargs=split_kwargs,
            )
        span_annots_predicted = []
        if 'tags_predicted' in _labeled_instance and self.show_prediction:
            span_annots_predicted = tags_to_span_annotations(
                tags=_labeled_instance['tags_predicted'],
                tokens=_labeled_instance['tokens'],
                text=_labeled_instance['metadata']['text'],
                encoding=self.entity_coding_scheme,
                #label_suffix='-PREDICTED' if self.show_gold else ''
                split_kwargs=split_kwargs,
            )
        span_annot_dicts = [annot.as_typeddict(json_store=obj_store) for annot in span_annots + span_annots_predicted]
        if self.instance_type == BratDatasetReader.ENTITY_TYPE:
            results = span_annot_dicts

        elif self.instance_type == BratDatasetReader.RELATION_TYPE:
            arg_annots = tags_to_span_annotations(
                tags=_labeled_instance['types'],
                tokens=_labeled_instance['tokens'],
                text=_labeled_instance['metadata']['text'],
                encoding=self.relation_argument_coding_scheme,
                split_kwargs=split_kwargs,
            )
            span_annots_dict = {span_annot.target.hash(): span_annot for span_annot in span_annots}
            rel_args = {arg_annot.label: span_annots_dict[arg_annot.target.hash()] for arg_annot in arg_annots}

            def create_rel(label: str, args: Dict[str, BratSpanAnnotation], label_suffix: str = ''):
                if label.endswith('_rev'):
                    label = label[:-len('_rev')]
                    assert len(args) == 2, 'only arguments with two entries are back-revertible'
                    args = dict(zip(args.keys(), reversed(args.values())))
                if label in self.symmetric_relations:
                    # If the relation is an instance of a symmetric relation, e.g. parts_of_same or semantically_same,
                    # then we normalize it by independently sorting argument names (by string) and their targets (by
                    # start index of the target).
                    # Background: When reading corpus data, we add symmetric relations in both directions to simulate
                    # undirected-ness since the BRAT specification only allows to define directed relations. So, the
                    # code below deduplicates semantically same relation instances.
                    assert len(args) == 2, 'only arguments with two entries can be symmetric'
                    args = dict(zip(
                        sorted(args.keys()),
                        sorted(args.values(), key=lambda span_annotation: span_annotation.target.first_start())
                    ))
                return BratRelationAnnotation(label=label + label_suffix, target=BratRelation.from_dict(**args))

            rels = []
            if self.show_prediction:
                rel_label = labeled_instance['label_predicted'].label
                if rel_label != self.none_label:
                    rel = create_rel(rel_label, rel_args)#, label_suffix='-PREDICTED' if self.show_gold else '')
                    rel_dict = rel.as_typeddict(json_store=obj_store)
                    rels.append(rel_dict)
            rel_label = labeled_instance['label'].label
            if self.show_gold and rel_label != self.none_label:
                rel = create_rel(rel_label, rel_args, label_suffix='-GOLD' if self.show_prediction else '')
                rel_dict = rel.as_typeddict(json_store=obj_store)
                rels.append(rel_dict)
            results = span_annot_dicts + rels
        else:
            raise ConfigurationError(f"unknown instance type: {self.instance_type}. "
                                     f"Use one of: {', '.join(BratDatasetReader.KNOWN_ANNOTATION_TYPES)}")
        self._stats['time']['labeled_instance_to_json'] += time.time() - start
        return results

    def model_output_to_json(self, instance: Instance, outputs: Dict[str, ndarray], obj_store: Optional[Dict] = None) \
            -> JsonDict:
        """
        This method creates annotations (ADU or REL) by using predicted labels or tags and then convert those
        annotations into dictionary to be stored as json.
        :param instance: data instance containing tokens with token indexer, some meta data and tags or label depending
        on type of instance.
        :param outputs: contains predicted tags or labels for given data instance.
        :param obj_store: calculated objects are stored by their hashes and this allows for deduplication
        :return: it returns text_id, annotations and text for given instance in a JsonDict format
        """
        start = time.time()
        labeled_instance = self.prediction_to_labeled_instance(instance, outputs)
        res = {
            'text_id': instance['metadata']['text_id'],
            'ann': self.labeled_instance_to_json(labeled_instance, obj_store=obj_store),
            'txt': instance['metadata']['text']
        }

        end = time.time()
        self._stats['time']['model_output_to_json'] += end - start
        return res
