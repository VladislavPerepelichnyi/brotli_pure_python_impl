"""Encoder-side view of the static dictionary (port of the parts of
c/enc/encoder_dict.h / encoder_dict.c that are used with the built-in
dictionary; custom / compound dictionaries are not supported).

Hierarchy (as in C):
  SharedEncoderDictionary
    .compound   : CompoundDictionary (always empty here)
    .contextual : ContextualEncoderDictionary
                    .context_based = False
                    .dict[0..63]   -> the single BrotliEncoderDictionary
    .max_quality = 11
"""

from ..common import dictionary as _common_dictionary
from .dictionary_hash import (kStaticDictionaryHashWords,
                              kStaticDictionaryHashLengths)

SHARED_BROTLI_NUM_DICTIONARY_CONTEXTS = 64
BROTLI_NUM_TRANSFORMS = 121  # BrotliGetTransforms()->num_transforms (RFC 7932)

kCutoffTransformsCount = 10
#   0,  12,   27,    23,    42,    63,    56,    48,    59,    64
# 0+0, 4+8, 8+19, 12+11, 16+26, 20+43, 24+32, 28+20, 32+27, 36+28
kCutoffTransforms = (0x071B520A << 32) | 0xDA2D3200


class BrotliDictionary(object):
    """BrotliDictionary (c/common/dictionary.h)."""

    __slots__ = ('size_bits_by_length', 'offsets_by_length', 'data_size',
                 'data')

    def __init__(self):
        self.size_bits_by_length = _common_dictionary.SIZE_BITS_BY_LENGTH
        self.offsets_by_length = _common_dictionary.OFFSETS_BY_LENGTH
        self.data = _common_dictionary.DATA
        self.data_size = len(self.data)


_DICTIONARY = None


def brotli_get_dictionary():
    """BrotliGetDictionary."""
    global _DICTIONARY
    if _DICTIONARY is None:
        _DICTIONARY = BrotliDictionary()
    return _DICTIONARY


class BrotliEncoderDictionary(object):
    """Dictionary data (words and transforms) for 1 possible context.

    ``buckets`` / ``dict_words`` (kStaticDictionaryBuckets /
    kStaticDictionaryWords from static_dict_lut.h) are owned by the quality-11
    static dictionary module and are left as None here; that module should
    use its own LUT when it finds None."""

    __slots__ = ('words', 'num_transforms', 'cutoff_transforms_count',
                 'cutoff_transforms', 'hash_table_words',
                 'hash_table_lengths', 'buckets', 'dict_words',
                 'has_words_heavy', 'parent')

    def __init__(self):
        # InitEncoderDictionary
        self.words = brotli_get_dictionary()
        self.num_transforms = BROTLI_NUM_TRANSFORMS
        self.hash_table_words = kStaticDictionaryHashWords
        self.hash_table_lengths = kStaticDictionaryHashLengths
        self.buckets = None
        self.dict_words = None
        self.cutoff_transforms_count = kCutoffTransformsCount
        self.cutoff_transforms = kCutoffTransforms
        self.parent = None
        self.has_words_heavy = False


class CompoundDictionary(object):
    """CompoundDictionary (compound_dictionary.h); always empty here."""

    __slots__ = ('num_chunks', 'total_size', 'chunk_offsets', 'chunk_source',
                 'chunks', 'num_prepared_instances_')

    def __init__(self):
        self.num_chunks = 0
        self.total_size = 0
        self.chunk_offsets = [0] * 16
        self.chunk_source = []
        self.chunks = []
        self.num_prepared_instances_ = 0


class ContextualEncoderDictionary(object):
    __slots__ = ('context_based', 'num_dictionaries', 'context_map', 'dict',
                 'num_instances_', 'instance_')

    def __init__(self):
        self.context_based = False
        self.num_dictionaries = 1
        self.context_map = bytes(SHARED_BROTLI_NUM_DICTIONARY_CONTEXTS)
        self.instance_ = BrotliEncoderDictionary()
        self.instance_.parent = self
        self.num_instances_ = 1
        self.dict = [self.instance_] * SHARED_BROTLI_NUM_DICTIONARY_CONTEXTS


class SharedEncoderDictionary(object):
    __slots__ = ('magic', 'compound', 'contextual', 'max_quality')

    def __init__(self):
        self.magic = 0xDEBCEDE0  # kSharedDictionaryMagic
        self.compound = CompoundDictionary()
        self.contextual = ContextualEncoderDictionary()
        self.max_quality = 11  # BROTLI_MAX_QUALITY


_SHARED = None


def brotli_init_shared_encoder_dictionary():
    """BrotliInitSharedEncoderDictionary: returns the (shared, immutable)
    default dictionary object."""
    global _SHARED
    if _SHARED is None:
        _SHARED = SharedEncoderDictionary()
    return _SHARED
