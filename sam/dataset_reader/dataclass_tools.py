import copy
import hashlib
import importlib
import json
import logging
from collections import namedtuple

from dataclasses import _is_dataclass_instance, fields, dataclass, FrozenInstanceError
from typing import Optional

logger = logging.getLogger(__name__)

HASH_PREFIX = '_hash:'
TYPE_KEY = '_type'
FIXED_SIZE_TYPES = (int, float, bool, type(None))
VAR_SIZE_TYPES = (tuple, list, dict, str)
IMMUTABLE_TYPES = FIXED_SIZE_TYPES + (tuple, str)
MUTABLE_TYPES = (list, dict)


def tuple_wrapper(*args):
    return tuple(args)


def isnamedtupleclass(t):
    if not hasattr(t, '__bases__'):
        return False
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


NAMED_TUPLE_MODULE_NAME = 'NamedTuple'


def class_to_str(cls):
    if isnamedtupleclass(cls):
        return f'{NAMED_TUPLE_MODULE_NAME}.{cls.__name__}({",".join(cls._fields)})'

    assert cls.__module__ != NAMED_TUPLE_MODULE_NAME, f'classed from module with name "{NAMED_TUPLE_MODULE_NAME}" not allowed'
    return f'{cls.__module__}.{cls.__name__}'


def str_to_class(class_str):
    """Return a class instance from a string reference"""
    module_name, class_name = class_str.rsplit(".", 1)
    if module_name == NAMED_TUPLE_MODULE_NAME:
        assert '(' in class_name and class_name[-1] == ')', \
            f'malformed named named tuple class string, it has to start with a class name followed by comma separated ' \
            f'argument labels encapsulated in brackets, but is: {class_name}'
        class_name, fields_string = class_name.split('(', maxsplit=1)
        field_names = fields_string[:-1].split(',')
        class_ = namedtuple(class_name, field_names=field_names)
    else:
        module_ = importlib.import_module(module_name)
        class_ = getattr(module_, class_name)
    return class_


def typeddict_to_dataclass(typeddict, type_key=TYPE_KEY, json_store=None):
    if isinstance(typeddict, dict):
        cls = None
        if type_key in typeddict:
            # do not pop! we may re-use this dict
            type_str = typeddict[type_key]
            cls = str_to_class(type_str)
        res_dict = {typeddict_to_dataclass(k, type_key, json_store): typeddict_to_dataclass(v, type_key, json_store=json_store)
                    for k, v in typeddict.items() if k != type_key}
        if cls is not None:
            res = cls(**res_dict)
        else:
            res = res_dict
        return res
    elif isinstance(typeddict, (list, tuple)):
        res = type(typeddict)((typeddict_to_dataclass(e, type_key, json_store=json_store) for e in typeddict))
        # named tuples contain the class marker TYPE_KEY as first element directly followed by the class string
        if len(res) > 0 and res[0] == TYPE_KEY:
            assert len(res) > 1, f'found class marker [{TYPE_KEY}] in list/tuple, but now class string is available'
            cls = str_to_class(res[1])
            res = cls(*res[2:])
        return res
    elif isinstance(typeddict, str) and typeddict.startswith(HASH_PREFIX):
        stored = json_store[typeddict]
        return typeddict_to_dataclass(stored, type_key, json_store=json_store)
    else:
        return typeddict


def dc_as_str(obj, **kwargs):
    return json.dumps(dataclass_as_typeddict(obj, **kwargs), sort_keys=True)


def dc_hash(obj, omit_fields_prefix='_'):
    to_hash = dc_as_str(obj, omit_fields_prefix=omit_fields_prefix)
    h = hashlib.md5(to_hash.encode('utf-8')).hexdigest()
    return h


def dataclass_as_typeddict(obj, *, dict_factory=dict, type_key=TYPE_KEY, omit_fields_prefix: Optional[str] = None,
                           json_store=None, hash_func=dc_hash, hash_prefix=HASH_PREFIX):
    """Return the fields of a dataclass instance as a new dictionary mapping
    field names to field values.

    Example usage:

      @dataclass
      class C:
          x: int
          y: int

      c = C(1, 2)
      assert asdict(c) == {'x': 1, 'y': 2}

    If given, 'dict_factory' will be used instead of built-in dict.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts.
    """
    if not _is_dataclass_instance(obj) and not isinstance(obj, VAR_SIZE_TYPES):# and not isinstance(obj, (dict, tuple, list)):
        raise TypeError("asdict() should be called on dataclass instances")
    return _astypeddict_inner(obj, dict_factory, type_key, omit_fields_prefix, json_store, hash_func, hash_prefix)


def _astypeddict_inner(obj, dict_factory, type_key, omit_fields_prefix, json_store, hash_func, hash_prefix):
    h = None
    if json_store is not None:
        if not isinstance(obj, FIXED_SIZE_TYPES):
            h_value = obj.hash() if isinstance(obj, FrozenDataclass) else hash_func(obj)
            h = f'{hash_prefix}{h_value}'
            if h in json_store:
                return h

    if _is_dataclass_instance(obj):
        result = []
        for f in fields(obj):
            if omit_fields_prefix is None or not f.name.startswith(omit_fields_prefix):
                value = _astypeddict_inner(getattr(obj, f.name), dict_factory, type_key, omit_fields_prefix, json_store,
                                           hash_func, hash_prefix)
                assert f.name != type_key, f'field name does not be the type_key={type_key}'
                result.append((f.name, value))
        result.append((type_key, class_to_str(type(obj))))
        res = dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        #res = type(obj)(*[_astypeddict_inner(v, dict_factory, type_key, omit_fields_prefix, json_store, hash_func, hash_prefix) for v in obj])
        res = [type_key, class_to_str(type(obj))]
        for f in obj._fields:
            v = getattr(obj, f)
            if omit_fields_prefix is None or not f.startswith(omit_fields_prefix):
                assert f != type_key, f'field name does not be the type_key={type_key}'
                value = _astypeddict_inner(v, dict_factory, type_key, omit_fields_prefix, json_store,
                                           hash_func, hash_prefix)
                res.append(value)
        res = tuple(res)
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        res = type(obj)(_astypeddict_inner(v, dict_factory, type_key, omit_fields_prefix, json_store, hash_func, hash_prefix) for v in obj)
        if isinstance(obj, tuple):
            cls_str = class_to_str(tuple_wrapper)
            res = type(obj)(v for v in [TYPE_KEY, cls_str]) + res
    elif isinstance(obj, dict):
        res = type(obj)((_astypeddict_inner(k, dict_factory, type_key, omit_fields_prefix, json_store, hash_func, hash_prefix),
                          _astypeddict_inner(v, dict_factory, type_key, omit_fields_prefix, json_store, hash_func, hash_prefix))
                         for k, v in obj.items())
    else:
        if json_store is not None and isinstance(obj, str):
            assert not obj.startswith(hash_prefix), \
                f'strings with prefix hash prefix={hash_prefix} not allowed if obj_store is used'
        res = copy.deepcopy(obj)

    if json_store is not None and h is not None:
        json_store[h] = res
        return h
    else:
        return res


def check_frozen(obj):
    if isinstance(obj, FIXED_SIZE_TYPES + (str,)):
        pass
    elif isinstance(obj, FrozenDataclass):
        for f in fields(obj):
            check_frozen(getattr(obj, f.name))
    elif isinstance(obj, MUTABLE_TYPES):
        raise FrozenInstanceError(f'is not frozen, but a mutable object: {obj}')
    elif isinstance(obj, tuple):
        # named tuple
        if hasattr(obj, '_fields'):
            for v in obj._asdict().values():
                check_frozen(v)
        # default tuple
        else:
            for e in obj:
                check_frozen(e)
    else:
        raise FrozenInstanceError(f'can not assume that object of unknown type is frozen (use FrozenDataclass type): {obj}')


FROZEN_DATACLASS_HASH_KEY = '__hash'


@dataclass(frozen=True)
class FrozenDataclass:

    def __post_init__(self):
        check_frozen(self)
        assert FROZEN_DATACLASS_HASH_KEY not in self.__dict__, \
            f'{FROZEN_DATACLASS_HASH_KEY} not allowed in self.__dict__'

    def hash(self):
        # bypass frozen to cache own hash
        if FROZEN_DATACLASS_HASH_KEY not in self.__dict__:
            self.__dict__[FROZEN_DATACLASS_HASH_KEY] = dc_hash(self)
        return self.__dict__[FROZEN_DATACLASS_HASH_KEY]

    @staticmethod
    def from_typeddict(typeddict, **kwarks):
        return typeddict_to_dataclass(typeddict, **kwarks)

    def as_typeddict(self, **kwargs):
        return dataclass_as_typeddict(self, **kwargs)




