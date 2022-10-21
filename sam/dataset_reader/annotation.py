from __future__ import annotations

import logging
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Union, cast, Tuple, NamedTuple, Callable

from sam.dataset_reader.dataclass_tools import FrozenDataclass

logger = logging.getLogger(__name__)


def pos_subsequence(needle, haystack):
    """
    Finds if a list is a subsequence of another.

    * args
        needle: the candidate subsequence
        haystack: the parent list

    * returns
        boolean

    is_subsequence([1, 2, 3, 4], [1, 2, 3, 4, 5, 6])
    True
    is_subsequence([1, 2, 3, 4], [1, 2, 3, 5, 6])
    False
    is_subsequence([6], [1, 2, 3, 5, 6])
    True
    is_subsequence([5, 6], [1, 2, 3, 5, 6])
    True
    is_subsequence([[5, 6], 7], [1, 2, 3, [5, 6], 7])
    True
    is_subsequence([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, [5, 6], 7])
    False
    """
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i+len(needle)] == needle:
            return i
    return None


@dataclass(frozen=True)
class AnnotationTarget(ABC, FrozenDataclass):
    pass


@dataclass(frozen=True)
class _AnnotationBase:
    label: str
    target: AnnotationTarget


@dataclass(frozen=True)
class _AnnotationDefaultsBase:
    _name: Optional[str] = None


@dataclass(frozen=True)
class Annotation(_AnnotationDefaultsBase, _AnnotationBase, FrozenDataclass):
    pass

    def __eq__(self, other):
        return self.hash() == other.hash()


@dataclass(frozen=True)
class Slice(FrozenDataclass):
    """
    A SpanPosition identifies a slice of a Sequential.
    """
    start: int
    end: int

    def shift(self, offset) -> Slice:
        return replace(self, start=self.start+offset, end=self.end+offset)

    def shift_start(self, offset) -> Slice:
        return replace(self, start=self.start+offset)

    def shift_end(self, offset) -> Slice:
        return replace(self, end=self.end+offset)

    def __len__(self):
        return self.end - self.start

    def __contains__(self, item):
        return self.start <= item < self.end


@dataclass(frozen=True)
class MultiSlice(AnnotationTarget):
    """
    A Span is identified by one or multiple slices (see SpanPosition).
    """
    slices: Tuple[Slice]
    base: Sequence

    def get_spans(self) -> Tuple[Sequence]:
        return tuple(self.base[pos.start:pos.end] for pos in self.slices)

    # per default (if glue_sequence=='') merge all contiguous positions
    def merge_slices(self, glue_sequence: Sequence = '') -> MultiSlice:
        assert isinstance(self.base, type(glue_sequence)) or isinstance(glue_sequence, type(self.base)), \
            f'type of glue_sequence [{type(glue_sequence)}] does not match type of self.base [{self.base}]'
        modified = False
        new_slices: List[Slice] = []
        if len(self.slices) > 1:
            new_slices.append(self.slices[0])
            for s in self.slices[1:]:
                # merge span annotations over newline
                if s.start - new_slices[-1].end == len(glue_sequence) \
                        and self.base[new_slices[-1].end:s.start] == glue_sequence:
                    new_slices[-1] = replace(new_slices[-1], end=s.end)
                    # NOTE: if merged, do not check if the annotation text matches
                    modified = True
                else:
                    new_slices.append(s)
        return replace(self, slices=tuple(new_slices)) if modified else self

    def split_slices_DEP(self, pattern: Sequence) -> Tuple[MultiSlice, bool]:
        assert isinstance(self.base, type(pattern)) or isinstance(pattern, type(self.base)), \
            f'type of remove_sequence [{type(pattern)}] does not match type of self.base [{self.base}]'
        modified = False
        new_slices: List[Slice] = []
        slices = list(self.slices)
        while len(slices) > 0:
            s = slices.pop(0)
            pos_in_slice = pos_subsequence(pattern, self.base[s.start:s.end])
            if pos_in_slice is not None:
                new_slices.append(replace(s, end=s.start + pos_in_slice))
                slices.insert(0, replace(s, start=s.start + pos_in_slice + len(pattern)))
                modified = True
            else:
                new_slices.append(s)
        return replace(self, slices=tuple(new_slices)) if modified else self, modified

    def is_in_range(self, start: int, end: int) -> bool:
        _in_range = True
        for s in self.slices:
            if s.start < start or end < s.end:
                _in_range = False
                break
        return _in_range

    def has_overlap(self, start: int, end: int) -> bool:
        _overlaps = False
        for s in self.slices:
            if start <= s.start < end or start < s.end <= end:
                _overlaps = True
                break
        return _overlaps

    def shift(self, offset: int) -> MultiSlice:
        return replace(self, slices=tuple(s.shift(offset=offset) for s in self.slices))

    def remap(self, mapping: Sequence[Slice], base: Sequence) -> MultiSlice:
        """

        :param mapping: a list of positions (dict with start and end entries) that point to same space as previous
                        self.positions and indices of the elements will be the new positions. The mapping may be
                        obtained from start and end offsets from tokens after tokenization (switching from character
                        base to token base).
        :param base: the new base
        """

        new_indices = []
        for i, new_slice in enumerate(mapping):
            if new_slice.start is None or new_slice.end is None:
                continue
            # if pos is contained in any self.positions entry -> add i to new_indices
            for old_slice in self.slices:
                if old_slice.start <= new_slice.start < old_slice.end \
                        and old_slice.start < new_slice.end <= old_slice.end:
                    new_indices.append(i)
                    break
        assert len(new_indices) > 0, f'could not find any new slice that fits into old one'
        # create continuous elements from new_indices
        prev_idx = new_indices[0]
        new_slices = []
        new_slice = {'start': new_indices[0]}
        for idx in new_indices[1:]:
            if prev_idx + 1 != idx:
                new_slice['end'] = prev_idx + 1
                new_slices.append(Slice(**new_slice))
                new_slice = {'start': idx}
            prev_idx = idx
        new_slice['end'] = prev_idx + 1
        new_slices.append(Slice(**new_slice))
        return replace(self, slices=tuple(new_slices), base=base)

    def first_start(self):
        return self.slices[0].start

    def last_end(self):
        return self.slices[-1].end

    def is_before(self, other):
        return self.last_end() <= other.first_start()

    def is_after(self, other):
        return other.last_end() <= self.first_start()

    def center_position(self) -> float:
        assert len(self.slices) > 0, f'can not get center position if number if slices is zero'
        return self.slices[0].start + (self.slices[-1].end - self.slices[0].start) / 2

    @staticmethod
    def distance(ms1, ms2):
        if ms1.is_before(ms2):
            d = ms2.first_start() - ms1.last_end()
        elif ms2.is_before(ms1):
            d = ms1.first_start() - ms2.last_end()
        else:
            d = -1
        return d

    def __iter__(self):
        return iter(self.slices)

    def __len__(self):
        return len(self.slices)


@dataclass(frozen=True)
class Relation(AnnotationTarget):
    arguments: NamedTuple


@dataclass(frozen=True)
class _SpanAnnotationBase(_AnnotationBase):
    target: MultiSlice  # overwrite with more specific type


@dataclass(frozen=True)
class _RelationAnnotationBase(_AnnotationBase):
    target: Relation  # overwrite with more specific type


@dataclass(frozen=True, eq=False)
class SpanAnnotation(Annotation, _AnnotationDefaultsBase, _SpanAnnotationBase):
    pass


@dataclass(frozen=True, eq=False)
class RelationAnnotation(Annotation, _RelationAnnotationBase):

    def check_argument_names(self, argument_names_dict, default_key=None):
        if self.label in argument_names_dict:
            key = self.label
        else:
            assert default_key is not None, f'default key should not be None if label not in argument_names'
            key = default_key
        assert argument_names_dict[key] == self.target.arguments._fields, \
            f'expected argument names {argument_names_dict[key]} do not match existing argument names ' \
            f'{self.target.arguments._fields}'


@dataclass(frozen=True)
class AnnotationLayer(ABC, FrozenDataclass):
    """
    An AnnotationLayer holds annotation objects of the same type.
    """
    annotations: Tuple[Annotation]
    _name: Optional[str] = None

    def __post_init__(self):
        self._check_types(self.annotations)

    @staticmethod
    def annotation_type():
        raise NotImplementedError

    def as_name_dict(self) -> Dict[str, Annotation]:
        res = {}
        for annot in self:
            assert annot._name is not None, f'annotation name is required for dict creation'
            assert annot._name not in res, f'layer contains mutliple annotations with same name: {annot._name}'
            res[annot._name] = annot
        assert len(res) == len(self), f'annotation layer contains annotations with same ids'
        return res

    def as_hash_dict(self) -> Union[Dict[str, SpanAnnotation], Dict[str, SpanAnnotation]]:
        return {annot.hash(): annot for annot in self}

    def _check_type(self, other):
        assert isinstance(other, self.annotation_type()), \
            f'annotation to add has wrong type: {type(other)}. Expected: {self.annotation_type()} or subclass.'

    def _check_types(self, others):
        for o in others:
            self._check_type(o)

    def annotation_names(self):
        return [annot._name for annot in self.annotations]

    def hashes(self):
        return [annot.hash() for annot in self.annotations]

    def count_tp_fp_fn(self, gold: AnnotationLayer,
                       exclude_annotation_filter: Optional[Callable[[Annotation],bool]]=None) -> Dict[Tuple, int]:
        counts = defaultdict(lambda: 0)
        matched_gold = []
        for annotation in self:
            if annotation in gold:
                matched_gold.append(annotation)
                if exclude_annotation_filter is None or not exclude_annotation_filter(annotation):
                    counts[('true_positives', annotation.label)] += 1
            else:
                if exclude_annotation_filter is None or not exclude_annotation_filter(annotation):
                    counts[('false_positives', annotation.label)] += 1
        for gold_annotation in gold:
            if gold_annotation not in matched_gold \
                    and (exclude_annotation_filter is None
                         or not exclude_annotation_filter(gold_annotation)):
                counts[('false_negatives', gold_annotation.label)] += 1
        return counts

    def __contains__(self, item):
        # Have to compare hashes because default "in" operator checks object equality not the hashes and equality
        # operator is not overwritable in FrozenDataClass or Annotation
        return item.hash() in self.hashes()

    def __iter__(self):
        return iter(self.annotations)

    def __len__(self):
        return len(self.annotations)

    def __eq__(self, other):
        if self._name != other._name:
            return False
        return set(self.hashes()) == set(other.hashes())


@dataclass(frozen=True)
class SpanAnnotationLayer(AnnotationLayer):
    """
    A SpanAnnotationLayer is a layer of SpanAnnotations that share the same base.
    """

    @staticmethod
    def annotation_type():
        return SpanAnnotation

    def base(self):
        if len(self) == 0:
            return None
        else:
            return cast(self.annotation_type(), self.annotations[0]).target.base

    def select_subset_with_slice(self, slice: Slice) -> Tuple[SpanAnnotationLayer, Dict[str, str]]:
        annots = []
        annot_backrefs = {}
        for annot in self:
            annot = cast(self.annotation_type(), annot)
            if annot.target.is_in_range(start=slice.start, end=slice.end):
                new_base = annot.target.base[slice.start:slice.end]
                annot_target = replace(annot.target, base=new_base).shift(-slice.start)
                new_annot = replace(annot, target=annot_target)
                annot_backrefs[annot.hash()] = new_annot.hash()
                annots.append(new_annot)
        new_layer = replace(self, annotations=tuple(annots))
        return new_layer, annot_backrefs

    def partition_with_slices(self, slices: List[Slice]) -> Tuple[Tuple[SpanAnnotationLayer], Tuple[Dict[str, str]]]:
        res, annot_backrefs_tuple = zip(*[self.select_subset_with_slice(slice) for slice in slices])
        self_keys = set(self.as_hash_dict().keys())
        sliced_keys = set()
        for s in res:
            sliced_keys.update(s.as_hash_dict().keys())
        assert len(self_keys) == len(sliced_keys), \
            f'number of new annotations [{len(sliced_keys)}] does not match number of original annotations ' \
            f'[{len(self_keys)}]'

        return res, annot_backrefs_tuple

    @classmethod
    def merge(cls, layers: List, bases: List[Sequence], delimiter: Sequence = '') \
            -> Tuple[SpanAnnotationLayer, Dict[str, str]]:
        merged_base = None
        annots = []
        names = set()
        annot_backrefs = {}
        for _layer, _base in zip(layers, bases):
            names.add(_layer._name)
            if merged_base is not None:
                merged_base += delimiter
            else:
                merged_base = ''
            offset = len(merged_base)
            # sanity check
            layer_base = _layer.base()
            if layer_base is not None:
                assert layer_base == _base, 'layer base does not match expected base'
            merged_base += _base
            for annot in _layer:
                annot_target = replace(annot.target, base=merged_base).shift(offset)
                new_annot = replace(annot, target=annot_target)
                annot_backrefs[annot.hash()] = new_annot.hash()
                annots.append(new_annot)
        assert len(names) == 1, \
            f'got different names for layers to merge, but they have to have the same (or None): {names}'
        res = cls(annotations=tuple(annots), _name=tuple(names)[0])
        return res, annot_backrefs

    def remap(self, mapping: Sequence[Slice], base: Sequence) \
            -> Tuple[AnnotationLayer, Dict[str, Tuple[SpanAnnotation, AssertionError]]]:
        annots_removed = dict()
        annots: List[SpanAnnotation] = []
        for h, annot in self.as_hash_dict().items():
            # remap span annotations to tokens
            try:
                annot_target = annot.target.remap(mapping, base)
                new_annot = replace(annot, target=annot_target)
                annots.append(new_annot)
            except AssertionError as e:
                annots_removed[h] = (annot, e)

        return replace(self, annotations=tuple(annots)), annots_removed

    def __eq__(self, other):
        return super().__eq__(other)


@dataclass
class AnnotationCollection:
    base: str
    layers: List[AnnotationLayer] = field(default_factory=list)

    def as_name_dict(self):
        d = {layer._name: layer for layer in self.layers}
        assert len(d) == len(self.layers), \
            f'annotation collection contains layers with duplicated names: {[layer._name for layer in self.layers]}'
        return d

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, key):
        return self.as_name_dict()[key]

    def layer_names(self):
        return (layer._name for layer in self.layers)

    def __iter__(self):
        return iter(self.layers)
