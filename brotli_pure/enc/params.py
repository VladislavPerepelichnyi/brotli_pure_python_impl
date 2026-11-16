"""Parameters for the Brotli encoder with chosen quality levels
(port of c/enc/params.h).  The constants of c/common/constants.h that the
encoder uses for distance parameters live here as well."""

from ..common.constants import MODE_GENERIC, DEFAULT_QUALITY, DEFAULT_WINDOW

BROTLI_NUM_DISTANCE_SHORT_CODES = 16
BROTLI_MAX_NPOSTFIX = 3
BROTLI_MAX_NDIRECT = 120
BROTLI_MAX_DISTANCE_BITS = 24
BROTLI_LARGE_MAX_DISTANCE_BITS = 62
BROTLI_MAX_DISTANCE = 0x3FFFFFC
BROTLI_MAX_ALLOWED_DISTANCE = 0x7FFFFFFC
BROTLI_WINDOW_GAP = 16


def brotli_distance_alphabet_size(npostfix, ndirect, maxnbits):
    """BROTLI_DISTANCE_ALPHABET_SIZE(NPOSTFIX, NDIRECT, MAXNBITS)."""
    return BROTLI_NUM_DISTANCE_SHORT_CODES + ndirect + (maxnbits << (npostfix + 1))


def brotli_max_backward_limit(w):
    """BROTLI_MAX_BACKWARD_LIMIT(W)."""
    return (1 << w) - BROTLI_WINDOW_GAP


class BrotliHasherParams(object):
    __slots__ = ('type', 'bucket_bits', 'block_bits',
                 'num_last_distances_to_check')

    def __init__(self):
        self.type = 0
        self.bucket_bits = 0
        self.block_bits = 0
        self.num_last_distances_to_check = 0

    def copy(self):
        h = BrotliHasherParams()
        h.type = self.type
        h.bucket_bits = self.bucket_bits
        h.block_bits = self.block_bits
        h.num_last_distances_to_check = self.num_last_distances_to_check
        return h

    def __repr__(self):
        return 'BrotliHasherParams(type=%d, bucket_bits=%d, block_bits=%d, nld=%d)' % (
            self.type, self.bucket_bits, self.block_bits,
            self.num_last_distances_to_check)


class BrotliDistanceParams(object):
    __slots__ = ('distance_postfix_bits', 'num_direct_distance_codes',
                 'alphabet_size_max', 'alphabet_size_limit', 'max_distance')

    def __init__(self):
        self.distance_postfix_bits = 0
        self.num_direct_distance_codes = 0
        self.alphabet_size_max = brotli_distance_alphabet_size(
            0, 0, BROTLI_MAX_DISTANCE_BITS)
        self.alphabet_size_limit = self.alphabet_size_max
        self.max_distance = BROTLI_MAX_DISTANCE

    def copy(self):
        d = BrotliDistanceParams()
        d.distance_postfix_bits = self.distance_postfix_bits
        d.num_direct_distance_codes = self.num_direct_distance_codes
        d.alphabet_size_max = self.alphabet_size_max
        d.alphabet_size_limit = self.alphabet_size_limit
        d.max_distance = self.max_distance
        return d

    def __repr__(self):
        return ('BrotliDistanceParams(npostfix=%d, ndirect=%d, max=%d, '
                'limit=%d, max_distance=%d)' % (
                    self.distance_postfix_bits, self.num_direct_distance_codes,
                    self.alphabet_size_max, self.alphabet_size_limit,
                    self.max_distance))


class BrotliEncoderParams(object):
    """BrotliEncoderParams; ``dictionary`` is a SharedEncoderDictionary
    (encoder_dict.py)."""

    __slots__ = ('mode', 'quality', 'lgwin', 'lgblock', 'stream_offset',
                 'size_hint', 'disable_literal_context_modeling',
                 'large_window', 'hasher', 'dist', 'dictionary')

    def __init__(self):
        # BrotliEncoderInitParams (encode.c)
        from .encoder_dict import brotli_init_shared_encoder_dictionary
        self.mode = MODE_GENERIC
        self.large_window = False
        self.quality = DEFAULT_QUALITY
        self.lgwin = DEFAULT_WINDOW
        self.lgblock = 0
        self.stream_offset = 0
        self.size_hint = 0
        self.disable_literal_context_modeling = False
        self.hasher = BrotliHasherParams()
        self.dictionary = brotli_init_shared_encoder_dictionary()
        self.dist = BrotliDistanceParams()

    def copy(self):
        """Struct copy (``BrotliEncoderParams block_params = *params``).
        The dictionary is shared (it is read-only during encoding)."""
        p = BrotliEncoderParams.__new__(BrotliEncoderParams)
        p.mode = self.mode
        p.quality = self.quality
        p.lgwin = self.lgwin
        p.lgblock = self.lgblock
        p.stream_offset = self.stream_offset
        p.size_hint = self.size_hint
        p.disable_literal_context_modeling = self.disable_literal_context_modeling
        p.large_window = self.large_window
        p.hasher = self.hasher.copy()
        p.dist = self.dist.copy()
        p.dictionary = self.dictionary
        return p
