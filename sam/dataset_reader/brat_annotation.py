import logging
import re
from abc import ABC
from collections import defaultdict, namedtuple
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple, Union, Iterable, cast

from typing_extensions import TypeAlias

from sam.dataset_reader.annotation import MultiSlice, Slice, Relation, SpanAnnotation, RelationAnnotation, \
    AnnotationLayer, AnnotationCollection, SpanAnnotationLayer
from sam.dataset_reader.dataclass_tools import FrozenDataclass

logger = logging.getLogger(__name__)


def dl_to_ld(dl):
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


@dataclass(frozen=True)
class BratSlice(Slice):

    @staticmethod
    def from_string(position_string: str, separator: str = ' '):
        parts = position_string.split(separator)
        return Slice(start=int(parts[0]), end=int(parts[1]))

    def to_string(self):
        return f'{self.start} {self.end}'

    @staticmethod
    def from_base_with_split_pattern(base, pattern, destructive: bool = False, offset: int = 0,
                                     use_regex: bool = False) -> List[Slice]:
        # split input by delete_pattern but keep matched text
        if use_regex:
            assert '(' not in pattern and ')' not in pattern, \
                f'"(" and ")" not allowed in pattern if use_regex: {pattern}'
            splits = re.split(pattern=f'({pattern})', string=base)
        else:
            splits = split_string(pattern=pattern, string=base)
        slices = []

        pos = 0
        s_odd = ''
        for i, s in enumerate(list(splits)):
            # every second element was matched with delete_pattern
            if i % 2 == 1:
                # if destructive splitting (i.e. the elements that match the split_pattern are removed) is used,
                # just increase the offset ...
                if destructive:
                    pos += len(s)
                # ... otherwise collect the content
                else:
                    s_odd = s
            else:
                if not destructive:
                    s = s_odd + s

                assert base[pos:pos + len(s)] == s, f'slice mismatch: "{base[pos:pos + len(s)]}" != "{s}"'
                new_slice = BratSlice(start=pos + offset, end=pos + len(s) + offset)
                slices.append(new_slice)

                pos += len(s)

        return slices


BRAT_MULTISLICE_GLUE_SEQUENCE = " "
TEXT_MULTISLICE_GLUE_SEQUENCE = "\n"


@dataclass(frozen=True)
class BratMultiSlice(MultiSlice):

    def verify_spans(self, expected_concatenated_spans: str, glue_sequence: str = ' '):
        concatenated_spans = glue_sequence.join(self.get_spans())
        assert concatenated_spans == expected_concatenated_spans, \
            f'concatenated_spans ["{concatenated_spans}"] does not match expected_concatenated_spans ' \
            f'["{expected_concatenated_spans}"]'

    def fix_spans(self, expected_concatenated_spans):
        self.verify_spans(expected_concatenated_spans=expected_concatenated_spans,
                          glue_sequence=BRAT_MULTISLICE_GLUE_SEQUENCE)

        return self.merge_slices(glue_sequence=TEXT_MULTISLICE_GLUE_SEQUENCE)

    @staticmethod
    def from_string(target_string, base, text_separator='\t'):
        position_string, span_string = target_string.split(text_separator, maxsplit=1)
        slices = tuple(BratSlice.from_string(part, separator=' ') for part in position_string.split(';'))
        res = BratMultiSlice(slices=slices, base=base)
        return res.fix_spans(expected_concatenated_spans=span_string)

    @staticmethod
    def from_arrow(locations: Dict[str, List[int]], text: str, base: str):
        locations_list = [dict(zip(locations, t)) for t in zip(*locations.values())]
        res = BratMultiSlice(slices=tuple(Slice(**location) for location in locations_list), base=base)
        return res.fix_spans(expected_concatenated_spans=text)

    def to_string(self):
        # TODO: investigate, why we have to re-create the BratSlice. This should not be necessary, but the sclie is a simepl Slice in some cases here
        positions_str = ';'.join(BratSlice(start=slice.start, end=slice.end).to_string() for slice in self.slices)
        spans_str = BRAT_MULTISLICE_GLUE_SEQUENCE.join(cast(Tuple[str], self.get_spans()))
        return f'{positions_str}\t{spans_str}'

    def split_slices(self, **kwargs):
        new_slices = []
        for slice in self:
            new_slices.extend(BratSlice.from_base_with_split_pattern(base=self.base[slice.start:slice.end],
                                                                     offset=slice.start, **kwargs))
        # filter empty slices and convert to tuple
        new_slices = tuple(s for s in new_slices if s.start != s.end)
        return replace(self, slices=new_slices)

    def __repr__(self):
        return f"BratMultiSlice(slices={[(s.start, s.end) for s in self.slices]}, base={self.base[:10]}{'...' if len(self.base) > 0 else ''})"


@dataclass(frozen=True, eq=False)
class BratSpanAnnotation(SpanAnnotation):

    @staticmethod
    def from_string(annotation_string: str, base: str, id_separator: str = '\t', text_separator: str = '\t',
                    label_separator: str = ' '):
        id, annotation_string = annotation_string.split(id_separator, maxsplit=1)
        ann_label, ann_target = annotation_string.split(label_separator, maxsplit=1)
        target = BratMultiSlice.from_string(ann_target, base=base, text_separator=text_separator)
        return BratSpanAnnotation(_name=id, label=ann_label, target=target)

    @staticmethod
    def from_arrow(type: str, id: Optional[str] = None, **kwargs):
        target = BratMultiSlice.from_arrow(**kwargs)
        return BratSpanAnnotation(_name=id, label=type, target=target)

    def split_target_slices(self, **kwargs):
        target = cast(BratMultiSlice, self.target).split_slices(**kwargs)
        return replace(self, target=target)

    def get_name(self):
        return self._name or f'T{self.hash()}'

    def to_string(self):
        target = cast(BratMultiSlice, self.target)
        res = f'{self.get_name()}\t{self.label} {target.to_string()}'
        return res


@dataclass(frozen=True)
class BratRelation(Relation):

    @staticmethod
    def from_dict(**kwargs):
        # use sorted keys to produce same hash for equal Arguments objects
        Arguments = namedtuple('Arguments', sorted(kwargs.keys()))
        return BratRelation(arguments=Arguments(**kwargs))

    @staticmethod
    def from_string(relation_string: str, target_mapping: Dict[str, BratSpanAnnotation], relation_separator: str = ' ',
                    label_separator: str = ':'):
        rel_args = [rel_arg.split(label_separator) for rel_arg in relation_string.split(relation_separator)]
        # argument labels should be unique
        assert len(rel_args) == len(set([rel_arg[0] for rel_arg in rel_args])), \
            f'argument labels for relation are not unique: {rel_args}'
        return BratRelation.from_dict(**{arg_name: target_mapping[arg_id] for arg_name, arg_id in rel_args})


@dataclass(frozen=True, eq=False)
class BratRelationAnnotation(RelationAnnotation):

    @staticmethod
    def from_string(annotation_string: str, target_mapping: Dict[str, BratSpanAnnotation], id_separator: str = '\t',
                    label_separator: str = ' '):
        id, annotation_string = annotation_string.split(id_separator, maxsplit=1)
        ann = annotation_string.strip()
        label, ann_target = ann.split(label_separator, maxsplit=1)
        # assert len(text) == 0, f'text has to be empty for relations, but is "{text}" for "{ann_id}"'
        target = BratRelation.from_string(ann_target, target_mapping)
        return BratRelationAnnotation(_name=id, label=label, target=target)

    @staticmethod
    def from_arrow(arguments: Dict[str, List[str]], type: str, target_mapping: Dict[str, BratSpanAnnotation],
                   id: Optional[str] = None):
        arguments_list = dl_to_ld(arguments)
        arg_dict = {arg['type']: target_mapping[arg['target']] for arg in arguments_list}
        return BratRelationAnnotation(_name=id, label=type, target=BratRelation.from_dict(**arg_dict))

    def get_name(self):
        return self._name or f'R{self.hash()}'

    def to_string(self):
        res = f'{self.get_name()}\t{self.label}'
        args_dict = self.target.arguments._asdict()
        # normalize by sorting by argument names
        for arg_name in sorted(args_dict):
            res += f' {arg_name}:{args_dict[arg_name].get_name()}'
        return res

BratAnnotation: TypeAlias = Union[BratRelationAnnotation, BratSpanAnnotation]

@dataclass(frozen=True, eq=False)
class BratAnnotationLayer(AnnotationLayer, ABC):

    def to_string(self):
        return '\n'.join([annotation.to_string() for annotation in self])


BRAT_SPAN_LAYER_NAME = 'T'
BRAT_RELATION_LAYER_NAME = 'R'


@dataclass(frozen=True, eq=False)
class BratSpanAnnotationLayer(BratAnnotationLayer, SpanAnnotationLayer):
    _name: Optional[str] = BRAT_SPAN_LAYER_NAME

    def __post_init__(self):
        super().__post_init__()
        for annotation in self:
            assert annotation.target.base == self.base(), \
                'BratSpanAnnotationLayer contains annotations for different bases'

    @staticmethod
    def annotation_type():
        return BratSpanAnnotation


@dataclass(frozen=True, eq=False)
class BratRelationAnnotationLayer(BratAnnotationLayer):
    _name: Optional[str] = BRAT_RELATION_LAYER_NAME

    @staticmethod
    def annotation_type():
        return BratRelationAnnotation

    def select_subset_with_valid_targets_via_names(self, span_annotations: BratSpanAnnotationLayer, link: bool = False,
                                                   ) -> 'BratRelationAnnotationLayer':
        keep: List[BratRelationAnnotation] = []
        span_dict = span_annotations.as_name_dict()
        for annot in self:
            args_dict = annot.target.arguments._asdict()
            if all(t._name in span_dict for t in args_dict.values()):
                if link:
                    new_args_dict = {arg_name: span_dict[arg_value._name] for arg_name, arg_value in
                                     args_dict.items()}
                    annot = replace(annot, target=BratRelation.from_dict(**new_args_dict))
                keep.append(annot)
        return replace(self, annotations=tuple(keep))

    def link_targets_via_hashes(self, target_layer: BratSpanAnnotationLayer):
        spans_hash_dict = target_layer.as_hash_dict()
        new_annots = []
        for annot in self:
            annot = cast(BratRelationAnnotation, annot)
            args_dict = annot.target.arguments._asdict()
            new_args_dict = {arg_name: spans_hash_dict[arg_value.hash()] for arg_name, arg_value in args_dict.items()}
            new_annots.append(replace(annot, target=BratRelation.from_dict(**new_args_dict)))
        return replace(self, annotations=tuple(new_annots))

    def check_argument_names(self, argument_names_dict, default_key=None):
        for annot in self:
            annot.check_argument_names(argument_names_dict=argument_names_dict, default_key=default_key)


@dataclass
class BratAnnotationCollection(AnnotationCollection):

    @staticmethod
    def from_annotations(annotations: Iterable[Union[BratSpanAnnotation, BratRelationAnnotation]], base: str):

        span_annots = []
        rel_annots = []
        hashes = set()
        for annotation in annotations:
            h = annotation.hash()
            if isinstance(annotation, BratSpanAnnotationLayer.annotation_type()):
                if h not in hashes:
                    span_annots.append(annotation)
            elif isinstance(annotation, BratRelationAnnotationLayer.annotation_type()):
                if h not in hashes:
                    rel_annots.append(annotation)
            else:
                raise TypeError(f'unknown annotation type: {type(annotation)}. expected: {BratSpanAnnotationLayer.annotation_type()}, '
                                f'{BratRelationAnnotationLayer.annotation_type()} or subclass')
            hashes.add(h)
        spans = BratSpanAnnotationLayer(annotations=tuple(span_annots))
        rels = BratRelationAnnotationLayer(annotations=tuple(rel_annots))

        # The objects that are targets might be equal to span entries, but they are not the same.
        # We have to re-link them.
        rels = rels.link_targets_via_hashes(spans)
        return BratAnnotationCollection(base=base, layers=[spans, rels])

    @staticmethod
    def from_annotation_dicts(annotation_dicts: List[Dict], base: str, json_store: Optional[Dict] = None):
        return BratAnnotationCollection.from_annotations(
            annotations=(FrozenDataclass.from_typeddict(ad, json_store=json_store) for ad in annotation_dicts),
            base=base
        )

    @staticmethod
    def from_file(file_path: str):
        ann_file_name = file_path + '.ann'
        txt_file_name = file_path + '.txt'

        with open(txt_file_name) as f:
            base_text = f.read()

        annot_lines_dict = defaultdict(list)
        with open(ann_file_name, "r") as f:
            for line in f:
                # skip empty lines and comments
                if len(line) <= 1 or line[0] == '#':
                    continue
                # remove newline character
                if line.endswith('\n'):
                    line = line[:-1]
                annot_lines_dict[line[0]].append(line)

        span_annots = list(BratSpanAnnotation.from_string(l, base=base_text) for l in annot_lines_dict['T'])
        spans = BratSpanAnnotationLayer(annotations=span_annots)

        rel_annots = list(BratRelationAnnotation.from_string(l, target_mapping=spans.as_name_dict()) for l in annot_lines_dict['R'])
        rels = BratRelationAnnotationLayer(annotations=rel_annots)

        return BratAnnotationCollection(layers=[spans, rels], base=base_text)

    def to_files(self, base_path):
        # join layer annotations using new line and add new line at the end of the file
        annotations_string = '\n'.join([layer.to_string() for layer in self]) + '\n'
        with open(base_path + '.txt', 'w') as f:
            f.write(self.base)
        with open(base_path + '.ann', 'w') as f:
            f.write(annotations_string)

    @staticmethod
    def from_arrow(instance: Dict,
                   relation_argument_names_dict: Optional[Dict[str, Tuple[str, str]]] = None,
                   relation_argument_names_default_key: Optional[str] = None):
        base_text = instance['context']

        spans_list = dl_to_ld(instance['spans'])
        spans = BratSpanAnnotationLayer(annotations=tuple(
            BratSpanAnnotation.from_arrow(base=base_text, **span) for span in spans_list
        ))

        relations_list = dl_to_ld(instance['relations'])
        rels = BratRelationAnnotationLayer(annotations=tuple(
            BratRelationAnnotation.from_arrow(**rel, target_mapping=spans.as_name_dict()) for rel in relations_list
        ))
        if relation_argument_names_dict is not None:
            rels.check_argument_names(argument_names_dict=relation_argument_names_dict,
                                      default_key=relation_argument_names_default_key)

        return BratAnnotationCollection(layers=[spans, rels], base=base_text)


def split_string(pattern: str, string: str) -> List[str]:
    """
    differs from re.split if delete_pattern contains brackets (this function allows it)!
    :param string: text to split
    :param pattern: pattern to split with
    :return: sequence of strings that form the input `text` if concatenated and every second element
        matches with `pattern`
    """
    start = 0
    splits = []
    for mo in re.finditer(pattern, string):
        splits.append(string[start:mo.start()])
        splits.append(string[mo.start():mo.end()])
        start = mo.end()
    splits.append(string[start:])
    assert ''.join(splits) == string, f'some input is lost during split by pattern={pattern}'
    return splits



