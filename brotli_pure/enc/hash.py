"""A (forgetful) hash table to the data seen by the compressor, to help create
backward references to previous data (port of c/enc/hash.h).

Hasher objects
--------------
Every hasher type N is a class ``HN`` (in the module porting its
``hash_*_inc.h`` template) constructed as ``HN(common, params)`` (C:
``InitializeHN``) and exposing the C template functions as methods::

    hash_type_length()          store_lookahead()
    prepare(one_shot, input_size, data)
    store(data, mask, ix)       store_range(data, mask, ix_start, ix_end)
    stitch_to_previous_block(num_bytes, position, ringbuffer, ringbuffer_mask)
    prepare_distance_cache(distance_cache)
    find_longest_match(dictionary, data, ring_buffer_mask, distance_cache,
                       cur_ix, max_length, max_backward, dictionary_distance,
                       max_distance, out)      # out: HasherSearchResult
    common                      # HasherCommon

``Hasher`` mirrors the C ``Hasher`` struct: ``hasher.common`` and
``hasher.privat`` (the HN instance, created lazily by ``hasher_setup``).
"""

from .hash_base import hash14, kHashMul32, kHashMul64  # noqa: F401 (re-export)
from .find_match_length import find_match_length_with_limit
from .quality import choose_hasher


class HasherCommon(object):
    """HasherCommon (hash.h).  ``extra`` (raw memory) is not modelled; the
    hasher objects own their tables."""

    __slots__ = ('is_setup', 'dict_num_lookups', 'dict_num_matches', 'params',
                 'is_prepared')

    def __init__(self):
        self.is_setup = False
        self.dict_num_lookups = 0
        self.dict_num_matches = 0
        self.params = None
        self.is_prepared = False


kCutoffTransformsCount = 10
kCutoffTransforms = (0x071B520A << 32) | 0xDA2D3200


class HasherSearchResult(object):
    """HasherSearchResult (hash.h)."""

    __slots__ = ('len', 'distance', 'score', 'len_code_delta')

    def __init__(self, len_=0, distance=0, score=0, len_code_delta=0):
        self.len = len_
        self.distance = distance
        self.score = score
        self.len_code_delta = len_code_delta

    def assign(self, other):
        self.len = other.len
        self.distance = other.distance
        self.score = other.score
        self.len_code_delta = other.len_code_delta


def prepare_distance_cache(distance_cache, num_distances):
    """PrepareDistanceCache: extends distance_cache (a list of >= 16 ints)."""
    if num_distances > 4:
        last_distance = distance_cache[0]
        distance_cache[4] = last_distance - 1
        distance_cache[5] = last_distance + 1
        distance_cache[6] = last_distance - 2
        distance_cache[7] = last_distance + 2
        distance_cache[8] = last_distance - 3
        distance_cache[9] = last_distance + 3
        if num_distances > 10:
            next_last_distance = distance_cache[1]
            distance_cache[10] = next_last_distance - 1
            distance_cache[11] = next_last_distance + 1
            distance_cache[12] = next_last_distance - 2
            distance_cache[13] = next_last_distance + 2
            distance_cache[14] = next_last_distance - 3
            distance_cache[15] = next_last_distance + 3


BROTLI_LITERAL_BYTE_SCORE = 135
BROTLI_DISTANCE_BIT_PENALTY = 30
# Score must be positive after applying maximal penalty.
BROTLI_SCORE_BASE = BROTLI_DISTANCE_BIT_PENALTY * 8 * 8  # * sizeof(size_t)


def backward_reference_score(copy_length, backward_reference_offset):
    """BackwardReferenceScore."""
    return (BROTLI_SCORE_BASE + BROTLI_LITERAL_BYTE_SCORE * copy_length -
            BROTLI_DISTANCE_BIT_PENALTY *
            (backward_reference_offset.bit_length() - 1))


def backward_reference_score_using_last_distance(copy_length):
    """BackwardReferenceScoreUsingLastDistance."""
    return BROTLI_LITERAL_BYTE_SCORE * copy_length + BROTLI_SCORE_BASE + 15


def backward_reference_penalty_using_last_distance(distance_short_code):
    """BackwardReferencePenaltyUsingLastDistance."""
    return 39 + ((0x1CA10 >> (distance_short_code & 0xE)) & 0xE)


def test_static_dictionary_item(dictionary, len_, word_idx, data, data_off,
                                max_length, max_backward, max_distance, out):
    """TestStaticDictionaryItem; ``data`` is (buffer, offset)."""
    words = dictionary.words
    offset = words.offsets_by_length[len_] + len_ * word_idx
    if len_ > max_length:
        return False
    matchlen = find_match_length_with_limit(data, data_off, words.data, offset,
                                            len_)
    if matchlen + dictionary.cutoff_transforms_count <= len_ or matchlen == 0:
        return False
    cut = len_ - matchlen
    transform_id = (cut << 2) + ((dictionary.cutoff_transforms >> (cut * 6)) & 0x3F)
    backward = (max_backward + 1 + word_idx +
                (transform_id << words.size_bits_by_length[len_]))
    if backward > max_distance:
        return False
    score = backward_reference_score(matchlen, backward)
    if score < out.score:
        return False
    out.len = matchlen
    out.len_code_delta = len_ - matchlen
    out.distance = backward
    out.score = score
    return True


def search_in_static_dictionary(dictionary, common, data, data_off, max_length,
                                max_backward, max_distance, out, shallow):
    """SearchInStaticDictionary."""
    if common.dict_num_matches < (common.dict_num_lookups >> 7):
        return
    key = hash14(data, data_off) << 1
    lengths = dictionary.hash_table_lengths
    words = dictionary.hash_table_words
    for _ in range(1 if shallow else 2):
        common.dict_num_lookups += 1
        if lengths[key] != 0:
            item_matches = test_static_dictionary_item(
                dictionary, lengths[key], words[key], data, data_off,
                max_length, max_backward, max_distance, out)
            if item_matches:
                common.dict_num_matches += 1
        key += 1


class BackwardMatch(object):
    """BackwardMatch (hash.h): ``distance`` and ``length_and_code``."""

    __slots__ = ('distance', 'length_and_code')

    def __init__(self, distance=0, length_and_code=0):
        self.distance = distance
        self.length_and_code = length_and_code

    def __repr__(self):
        return 'BackwardMatch(distance=%d, len=%d, code=%d)' % (
            self.distance, self.length_and_code >> 5,
            backward_match_length_code(self))


def init_backward_match(self, dist, len_):
    """InitBackwardMatch."""
    self.distance = dist & 0xFFFFFFFF
    self.length_and_code = (len_ << 5) & 0xFFFFFFFF
    return self


def init_dictionary_backward_match(self, dist, len_, len_code):
    """InitDictionaryBackwardMatch."""
    self.distance = dist & 0xFFFFFFFF
    self.length_and_code = ((len_ << 5) |
                            (0 if len_ == len_code else len_code)) & 0xFFFFFFFF
    return self


def backward_match_length(self):
    """BackwardMatchLength."""
    return self.length_and_code >> 5


def backward_match_length_code(self):
    """BackwardMatchLengthCode."""
    code = self.length_and_code & 31
    return code if code else (self.length_and_code >> 5)


# MAX_NUM_MATCHES == 64 + MAX_TREE_SEARCH_DEPTH
MAX_NUM_MATCHES_H10 = 128


def _hasher_class(type_):
    """Maps a hasher type id to its class (lazy imports)."""
    if type_ == 10:
        from .hash_to_binary_tree import H10
        return H10
    if type_ == 5:
        from .hash_longest_match import H5
        return H5
    if type_ == 6:
        from .hash_longest_match64 import H6
        return H6
    if type_ == 58:
        from .hash_longest_match_simd import H58
        return H58
    if type_ == 68:
        from .hash_longest_match64_simd import H68
        return H68
    if type_ == 40:
        from .hash_forgetful_chain import H40
        return H40
    if type_ == 41:
        from .hash_forgetful_chain import H41
        return H41
    if type_ == 42:
        from .hash_forgetful_chain import H42
        return H42
    raise NotImplementedError('hasher type %d is not ported' % type_)


class Hasher(object):
    """Hasher (hash.h): ``common`` + ``privat`` (the concrete hasher).

    Attribute lookups that miss on the wrapper are forwarded to ``privat``,
    so ``hasher.find_all_matches(...)`` == ``hasher.privat.find_all_matches``
    (C: ``FN(FindAllMatches)(&hasher->privat._H10, ...)``)."""

    __slots__ = ('common', 'privat')

    def __init__(self):
        # HasherInit
        self.common = HasherCommon()
        self.privat = None

    def __getattr__(self, name):
        privat = object.__getattribute__(self, 'privat')
        if privat is None:
            raise AttributeError(name)
        return getattr(privat, name)


def hasher_init(hasher=None):
    """HasherInit: returns a fresh (not set up) Hasher."""
    return Hasher()


def destroy_hasher(hasher):
    """DestroyHasher."""
    hasher.privat = None


def hasher_reset(hasher):
    """HasherReset."""
    hasher.common.is_prepared = False


def hasher_setup(hasher, params, data, position, input_size, is_last):
    """HasherSetup: chooses and creates the hasher on first use (updating
    params.hasher) and prepares it if needed."""
    one_shot = (position == 0 and is_last)
    common = hasher.common
    if not common.is_setup:
        choose_hasher(params, params.hasher)
        common.params = params.hasher
        common.dict_num_lookups = 0
        common.dict_num_matches = 0
        cls = _hasher_class(common.params.type)
        # HasherSize + Initialize: the hasher allocates its own tables.
        hasher.privat = cls(common, params, one_shot, input_size)
        # HasherReset
        common.is_prepared = False
        common.is_setup = True
    if not common.is_prepared:
        hasher.privat.prepare(one_shot, input_size, data)
        common.is_prepared = True


def init_or_stitch_to_previous_block(hasher, data, mask, params, position,
                                     input_size, is_last):
    """InitOrStitchToPreviousBlock."""
    hasher_setup(hasher, params, data, position, input_size, is_last)
    hasher.privat.stitch_to_previous_block(input_size, position, data, mask)
