"""Constants and formulas that affect speed-ratio trade-offs and thus define
quality levels (port of c/enc/quality.h)."""

from ..common.constants import (MIN_QUALITY as BROTLI_MIN_QUALITY,
                                MAX_QUALITY as BROTLI_MAX_QUALITY,
                                MIN_WINDOW_BITS as BROTLI_MIN_WINDOW_BITS,
                                MAX_WINDOW_BITS as BROTLI_MAX_WINDOW_BITS,
                                LARGE_MAX_WINDOW_BITS as BROTLI_LARGE_MAX_WINDOW_BITS,
                                MIN_INPUT_BLOCK_BITS as BROTLI_MIN_INPUT_BLOCK_BITS,
                                MAX_INPUT_BLOCK_BITS as BROTLI_MAX_INPUT_BLOCK_BITS)

FAST_ONE_PASS_COMPRESSION_QUALITY = 0
FAST_TWO_PASS_COMPRESSION_QUALITY = 1
ZOPFLIFICATION_QUALITY = 10
HQ_ZOPFLIFICATION_QUALITY = 11

MAX_QUALITY_FOR_STATIC_ENTROPY_CODES = 2
MIN_QUALITY_FOR_BLOCK_SPLIT = 4
MIN_QUALITY_FOR_NONZERO_DISTANCE_PARAMS = 4
MIN_QUALITY_FOR_OPTIMIZE_HISTOGRAMS = 4
MIN_QUALITY_FOR_EXTENSIVE_REFERENCE_SEARCH = 5
MIN_QUALITY_FOR_CONTEXT_MODELING = 5
MIN_QUALITY_FOR_HQ_CONTEXT_MODELING = 7
MIN_QUALITY_FOR_HQ_BLOCK_SPLITTING = 10

# For quality below MIN_QUALITY_FOR_BLOCK_SPLIT there is no block splitting,
# so we buffer at most this much literals and commands.
MAX_NUM_DELAYED_SYMBOLS = 0x2FFF

# The maximum length for which the zopflification uses distinct distances.
MAX_ZOPFLI_LEN_QUALITY_10 = 150
MAX_ZOPFLI_LEN_QUALITY_11 = 325

# Do not thoroughly search when a long copy is found.
BROTLI_LONG_COPY_QUICK_STEP = 16384

# platform.h: on x86-64 Linux/GCC (the reference build used as oracle)
# BROTLI_TZCNT64 is __builtin_ctzll and _M_X64 is not defined, hence
# BROTLI_MAX_SIMD_QUALITY == 6 and the SIMD hashers H58/H68 are compiled in.
# Set to None to emulate a build without SIMD hashers (H5/H6 are then used,
# which are not ported here).
BROTLI_MAX_SIMD_QUALITY = 6


def max_hash_table_size(quality):
    """MaxHashTableSize."""
    return 1 << 15 if quality == FAST_ONE_PASS_COMPRESSION_QUALITY else 1 << 17


def max_zopfli_len(params):
    """MaxZopfliLen."""
    return (MAX_ZOPFLI_LEN_QUALITY_10 if params.quality <= 10
            else MAX_ZOPFLI_LEN_QUALITY_11)


def max_zopfli_candidates(params):
    """MaxZopfliCandidates."""
    return 1 if params.quality <= 10 else 5


def sanitize_params(params):
    """SanitizeParams."""
    params.quality = min(BROTLI_MAX_QUALITY,
                         max(BROTLI_MIN_QUALITY, params.quality))
    if params.quality <= MAX_QUALITY_FOR_STATIC_ENTROPY_CODES:
        params.large_window = False
    if params.lgwin < BROTLI_MIN_WINDOW_BITS:
        params.lgwin = BROTLI_MIN_WINDOW_BITS
    else:
        max_lgwin = (BROTLI_LARGE_MAX_WINDOW_BITS if params.large_window
                     else BROTLI_MAX_WINDOW_BITS)
        if params.lgwin > max_lgwin:
            params.lgwin = max_lgwin


def compute_lg_block(params):
    """ComputeLgBlock: returns optimized lg_block value."""
    lgblock = params.lgblock
    if (params.quality == FAST_ONE_PASS_COMPRESSION_QUALITY or
            params.quality == FAST_TWO_PASS_COMPRESSION_QUALITY):
        lgblock = params.lgwin
    elif params.quality < MIN_QUALITY_FOR_BLOCK_SPLIT:
        lgblock = 14
    elif lgblock == 0:
        lgblock = 16
        if params.quality >= 9 and params.lgwin > lgblock:
            lgblock = min(18, params.lgwin)
    else:
        lgblock = min(BROTLI_MAX_INPUT_BLOCK_BITS,
                      max(BROTLI_MIN_INPUT_BLOCK_BITS, lgblock))
    return lgblock


def compute_rb_bits(params):
    """ComputeRbBits: log2 of the size of main ring buffer area."""
    return 1 + max(params.lgwin, params.lgblock)


def max_metablock_size(params):
    """MaxMetablockSize."""
    bits = min(compute_rb_bits(params), BROTLI_MAX_INPUT_BLOCK_BITS)
    return 1 << bits


def literal_spree_length_for_sparse_search(params):
    """LiteralSpreeLengthForSparseSearch."""
    return 64 if params.quality < 9 else 512


def choose_hasher(params, hparams):
    """ChooseHasher: fills BrotliHasherParams ``hparams`` in place."""
    if params.quality > 9:
        hparams.type = 10
    elif params.quality == 4 and params.size_hint >= (1 << 20):
        hparams.type = 54
    elif params.quality < 5:
        hparams.type = params.quality
    elif params.lgwin <= 16:
        hparams.type = (40 if params.quality < 7 else
                        41 if params.quality < 9 else 42)
    elif params.size_hint >= (1 << 20) and params.lgwin >= 19:
        if BROTLI_MAX_SIMD_QUALITY is not None:
            hparams.type = 68 if params.quality <= BROTLI_MAX_SIMD_QUALITY else 6
        else:
            hparams.type = 6
        hparams.block_bits = params.quality - 1
        hparams.bucket_bits = 15
        hparams.num_last_distances_to_check = (
            4 if params.quality < 7 else 10 if params.quality < 9 else 16)
    else:
        if BROTLI_MAX_SIMD_QUALITY is not None:
            hparams.type = 58 if params.quality <= BROTLI_MAX_SIMD_QUALITY else 5
        else:
            hparams.type = 5
        hparams.block_bits = params.quality - 1
        hparams.bucket_bits = 14 if params.quality < 7 else 15
        hparams.num_last_distances_to_check = (
            4 if params.quality < 7 else 10 if params.quality < 9 else 16)
    if params.lgwin > 24:
        # Different hashers for large window brotli.
        if hparams.type == 3:
            hparams.type = 35
        if hparams.type == 54:
            hparams.type = 55
        if hparams.type == 6 or hparams.type == 68:
            hparams.type = 65
