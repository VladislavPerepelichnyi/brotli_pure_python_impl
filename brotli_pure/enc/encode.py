"""Implementation of Brotli compressor (port of c/enc/encode.c, state.h).

``BrotliEncoderState`` ports the streaming state machine of encode.c
(``BrotliEncoderCompressStream`` with PROCESS / FLUSH / FINISH; metadata
blocks, stream offsets and attached dictionaries are not supported).
``compress()`` reproduces the C Python binding: ``Compressor.process(data) +
Compressor.finish()``, i.e. one PROCESS call with the whole input followed by
one FINISH call.

Supported qualities: all of 0-11.  Qualities 0 and 1 use the fragment
compressors (compress_fragment.py, compress_fragment_two_pass.py), 2-9 the
generic "hasher + backward references" path with the hashers that the x86-64
reference build selects for them (H2/H3/H4/H54, H40/H41/H42, H5/H6, H58/H68),
and 10 / 11 the zopfli modules (backward_references_hq.py, metablock_hq.py,
hash_to_binary_tree.py, ...; developed on the enc-q11 branch), which are
imported lazily.  ``UnsupportedQualityError`` (a subclass of both
``brotli_pure.error`` and ``NotImplementedError``) is kept for code paths
that are not ported (none of the qualities reachable through ``compress``).
"""

from ..common.constants import (MODE_GENERIC, MODE_TEXT, MODE_FONT,
                                MIN_WINDOW_BITS, MAX_WINDOW_BITS,
                                MIN_INPUT_BLOCK_BITS, MAX_INPUT_BLOCK_BITS)
from ..common.context import CONTEXT_LOOKUP, CONTEXT_UTF8, CONTEXT_SIGNED
from ..common.errors import BrotliError
from .backward_references import brotli_create_backward_references
from .bit_cost import brotli_bits_entropy
from .brotli_bit_stream import (brotli_store_meta_block,
                                brotli_store_meta_block_fast,
                                brotli_store_meta_block_trivial,
                                brotli_store_uncompressed_meta_block)
from .command import (init_insert_command, command_restore_distance_code,
                      get_length_code)
from .fast_log import fast_log2
from .hash import Hasher, hasher_reset, init_or_stitch_to_previous_block
from .metablock import (MetaBlockSplit, brotli_build_meta_block_greedy,
                        brotli_optimize_histograms, brotli_init_distance_params,
                        BROTLI_MAX_STATIC_CONTEXTS)
from .params import (BrotliEncoderParams, BROTLI_MAX_NPOSTFIX,
                     BROTLI_MAX_NDIRECT, BROTLI_WINDOW_GAP,
                     BROTLI_NUM_DISTANCE_SHORT_CODES, brotli_max_backward_limit)
from .quality import (sanitize_params, compute_lg_block, max_metablock_size,
                      FAST_ONE_PASS_COMPRESSION_QUALITY,
                      FAST_TWO_PASS_COMPRESSION_QUALITY,
                      ZOPFLIFICATION_QUALITY, HQ_ZOPFLIFICATION_QUALITY,
                      MAX_QUALITY_FOR_STATIC_ENTROPY_CODES,
                      MIN_QUALITY_FOR_BLOCK_SPLIT,
                      MIN_QUALITY_FOR_NONZERO_DISTANCE_PARAMS,
                      MIN_QUALITY_FOR_OPTIMIZE_HISTOGRAMS,
                      MIN_QUALITY_FOR_CONTEXT_MODELING,
                      MIN_QUALITY_FOR_HQ_CONTEXT_MODELING,
                      MIN_QUALITY_FOR_HQ_BLOCK_SPLITTING,
                      MAX_NUM_DELAYED_SYMBOLS)
from .quality import max_hash_table_size
from .ringbuffer import RingBuffer
from .write_bits import BitWriter

# BrotliEncoderOperation
BROTLI_OPERATION_PROCESS = 0
BROTLI_OPERATION_FLUSH = 1
BROTLI_OPERATION_FINISH = 2
BROTLI_OPERATION_EMIT_METADATA = 3

# BrotliEncoderStreamState
BROTLI_STREAM_PROCESSING = 0
BROTLI_STREAM_FLUSH_REQUESTED = 1
BROTLI_STREAM_FINISHED = 2
BROTLI_STREAM_METADATA_HEAD = 3
BROTLI_STREAM_METADATA_BODY = 4

# BrotliEncoderFlintState
BROTLI_FLINT_NEEDS_2_BYTES = 2
BROTLI_FLINT_NEEDS_1_BYTE = 1
BROTLI_FLINT_WAITING_FOR_PROCESSING = 0
BROTLI_FLINT_WAITING_FOR_FLUSHING = -1
BROTLI_FLINT_DONE = -2

BROTLI_UINT32_MAX = 0xFFFFFFFF

#: Qualities whose code path is fully ported.
SUPPORTED_QUALITIES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

kMinUTF8Ratio = 0.75


class UnsupportedQualityError(BrotliError, NotImplementedError):
    """Raised for valid quality levels whose code path is not ported."""


# ---------------------------------------------------------------------------
# Small helpers


def wrap_position(position):
    """WrapPosition: wraps 64-bit input position to 32-bit ring-buffer
    position preserving "not-a-first-lap" feature."""
    result = position & 0xFFFFFFFF
    gb = position >> 30
    if gb > 2:
        # Wrap every 2GiB; The first 3GB are continuous.
        result = (result & ((1 << 30) - 1)) | ((((gb - 1) & 1) + 1) << 30)
    return result


def encode_window_bits(lgwin, large_window):
    """EncodeWindowBits; returns (last_bytes, last_bytes_bits)."""
    if large_window:
        return ((lgwin & 0x3F) << 8) | 0x11, 14
    if lgwin == 16:
        return 0, 1
    elif lgwin == 17:
        return 1, 7
    elif lgwin > 17:
        return ((lgwin - 17) << 1) | 0x01, 4
    else:
        return ((lgwin - 8) << 4) | 0x01, 7


def hash_table_size(max_table_size, input_size):
    """HashTableSize."""
    htsize = 256
    while htsize < max_table_size and htsize < input_size:
        htsize <<= 1
    return htsize


def get_hash_table(quality, input_size):
    """GetHashTable: returns a zeroed hash table (list) for qualities 0/1.
    Use smaller hash table when input.size() is smaller, since we fill the
    table, incurring O(hash table size) overhead for compression, and if the
    input is short, we won't need that many hash table entries anyway."""
    max_table_size = max_hash_table_size(quality)
    htsize = hash_table_size(max_table_size, input_size)
    if quality == FAST_ONE_PASS_COMPRESSION_QUALITY:
        # Only odd shifts are supported by fast-one-pass.
        if (htsize & 0xAAAAA) == 0:
            htsize <<= 1
    return [0] * htsize


def estimate_entropy(population, start, size):
    """EstimateEntropy (over population[start:start+size])."""
    total = 0
    result = 0.0
    for i in range(start, start + size):
        p = population[i]
        total += p
        result += p * fast_log2(p)
    result = total * fast_log2(total) - result
    return result


kStaticContextMapContinuation = (
    1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
kStaticContextMapSimpleUTF8 = (
    0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
kStaticContextMapComplexUTF8 = (
    11, 11, 12, 12,  # 0 special
    0, 0, 0, 0,      # 4 lf
    1, 1, 9, 9,      # 8 space
    2, 2, 2, 2,      # !, first after space/lf and after something else.
    1, 1, 1, 1,      # "
    8, 3, 3, 3,      # %
    1, 1, 1, 1,      # ({[
    2, 2, 2, 2,      # }])
    8, 4, 4, 4,      # :;
    8, 7, 4, 4,      # .
    8, 0, 0, 0,      # >
    3, 3, 3, 3,      # [0..9]
    5, 5, 10, 5,     # [A-Z]
    5, 5, 10, 5,
    6, 6, 6, 6,      # [a-z]
    6, 6, 6, 6,
)


def choose_context_map(quality, bigram_histo):
    """ChooseContextMap: decides about the context map based on the ability
    of the prediction ability of the previous byte UTF8-prefix on the next
    byte.  Returns (num_literal_contexts, literal_context_map or None);
    None means "unchanged" (the C function leaves the map untouched)."""
    monogram_histo = [0, 0, 0]
    two_prefix_histo = [0] * 6
    for i in range(9):
        monogram_histo[i % 3] += bigram_histo[i]
        two_prefix_histo[i % 6] += bigram_histo[i]
    entropy1 = estimate_entropy(monogram_histo, 0, 3)
    entropy2 = (estimate_entropy(two_prefix_histo, 0, 3) +
                estimate_entropy(two_prefix_histo, 3, 3))
    entropy3 = 0.0
    for i in range(3):
        entropy3 += estimate_entropy(bigram_histo, 3 * i, 3)

    total = monogram_histo[0] + monogram_histo[1] + monogram_histo[2]
    entropy0 = 1.0 / total
    entropy1 *= entropy0
    entropy2 *= entropy0
    entropy3 *= entropy0

    if quality < MIN_QUALITY_FOR_HQ_CONTEXT_MODELING:
        # 3 context models is a bit slower, don't use it at lower qualities.
        entropy3 = entropy1 * 10
    # If expected savings by symbol are less than 0.2 bits, skip the
    # context modeling -- in exchange for faster decoding speed.
    if entropy1 - entropy2 < 0.2 and entropy1 - entropy3 < 0.2:
        return 1, None
    elif entropy2 - entropy3 < 0.02:
        return 2, kStaticContextMapSimpleUTF8
    else:
        return 3, kStaticContextMapContinuation


def should_use_complex_static_context_map(input_, start_pos, length, mask,
                                          quality, size_hint):
    """ShouldUseComplexStaticContextMap: returns
    (True, num_literal_contexts, literal_context_map) or (False, None, None).
    """
    # Try the more complex static context map only for long data.
    if size_hint < (1 << 20):
        return False, None, None
    end_pos = start_pos + length
    # To make entropy calculations faster, we collect histograms
    # over the 5 most significant bits of literals. One histogram
    # without context and 13 additional histograms for each context value.
    combined_histo = [0] * 32
    context_histo = [0] * (32 * BROTLI_MAX_STATIC_CONTEXTS)
    total = 0
    lut = CONTEXT_LOOKUP
    utf8_off = CONTEXT_UTF8 << 9
    cmap = kStaticContextMapComplexUTF8
    while start_pos + 64 <= end_pos:
        stride_end_pos = start_pos + 64
        prev2 = input_[start_pos & mask]
        prev1 = input_[(start_pos + 1) & mask]
        # To make the analysis of the data faster we only examine 64 byte
        # long strides at every 4kB intervals.
        for pos in range(start_pos + 2, stride_end_pos):
            literal = input_[pos & mask]
            context = cmap[lut[utf8_off + prev1] | lut[utf8_off + 256 + prev2]]
            total += 1
            combined_histo[literal >> 3] += 1
            context_histo[(context << 5) + (literal >> 3)] += 1
            prev2 = prev1
            prev1 = literal
        start_pos += 4096
    entropy1 = estimate_entropy(combined_histo, 0, 32)
    entropy2 = 0.0
    for i in range(BROTLI_MAX_STATIC_CONTEXTS):
        entropy2 += estimate_entropy(context_histo, i << 5, 32)
    entropy0 = 1.0 / total
    entropy1 *= entropy0
    entropy2 *= entropy0
    # The triggering heuristics below were tuned by compressing the
    # individual files of the silesia corpus. If we skip this kind of context
    # modeling for not very well compressible input (i.e. entropy using
    # context modeling is 60% of maximal entropy) or if expected savings by
    # symbol are less than 0.2 bits, then in every case when it triggers, the
    # final compression ratio is improved. Note however that this heuristics
    # might be too strict for some cases and could be tuned further.
    if entropy2 > 3.0 or entropy1 - entropy2 < 0.2:
        return False, None, None
    return True, BROTLI_MAX_STATIC_CONTEXTS, kStaticContextMapComplexUTF8


_BIGRAM_LUT = (0, 0, 1, 2)


def decide_over_literal_context_modeling(input_, start_pos, length, mask,
                                         quality, size_hint,
                                         num_literal_contexts,
                                         literal_context_map):
    """DecideOverLiteralContextModeling; returns the (possibly updated)
    (num_literal_contexts, literal_context_map)."""
    if quality < MIN_QUALITY_FOR_CONTEXT_MODELING or length < 64:
        return num_literal_contexts, literal_context_map
    ok, n, cmap = should_use_complex_static_context_map(
        input_, start_pos, length, mask, quality, size_hint)
    if ok:
        # Context map was already set, nothing else to do.
        return n, cmap
    # Gather bi-gram data of the UTF8 byte prefixes. To make the analysis of
    # UTF8 data faster we only examine 64 byte long strides at every 4kB
    # intervals.
    end_pos = start_pos + length
    bigram_prefix_histo = [0] * 9
    lut = _BIGRAM_LUT
    while start_pos + 64 <= end_pos:
        stride_end_pos = start_pos + 64
        prev = lut[input_[start_pos & mask] >> 6] * 3
        for pos in range(start_pos + 1, stride_end_pos):
            literal = input_[pos & mask]
            bigram_prefix_histo[prev + lut[literal >> 6]] += 1
            prev = lut[literal >> 6] * 3
        start_pos += 4096
    n, cmap = choose_context_map(quality, bigram_prefix_histo)
    if cmap is None:
        cmap = literal_context_map
    return n, cmap


def should_compress(data, mask, last_flush_pos, bytes_, num_literals,
                    num_commands):
    """ShouldCompress."""
    # TODO(eustas): find more precise minimal block overhead.
    if bytes_ <= 2:
        return False
    if num_commands < (bytes_ >> 8) + 2:
        if float(num_literals) > 0.99 * float(bytes_):
            literal_histo = [0] * 256
            kSampleRate = 13
            kInvSampleRate = 1.0 / 13.0
            kMinEntropy = 7.92
            bit_cost_threshold = float(bytes_) * kMinEntropy * kInvSampleRate
            t = (bytes_ + kSampleRate - 1) // kSampleRate
            pos = last_flush_pos & 0xFFFFFFFF
            for _ in range(t):
                literal_histo[data[pos & mask]] += 1
                pos = (pos + kSampleRate) & 0xFFFFFFFF
            if brotli_bits_entropy(literal_histo, 256) > bit_cost_threshold:
                return False
    return True


def choose_context_mode(params, data, pos, mask, length):
    """ChooseContextMode: chooses the literal context mode for a
    metablock."""
    # We only do the computation for the option of something else than
    # CONTEXT_UTF8 for the highest qualities
    if params.quality >= MIN_QUALITY_FOR_HQ_BLOCK_SPLITTING:
        from .utf8_util import brotli_is_mostly_utf8
        if not brotli_is_mostly_utf8(data, pos, mask, length, kMinUTF8Ratio):
            return CONTEXT_SIGNED
    return CONTEXT_UTF8


def write_meta_block_internal(data, mask, last_flush_pos, bytes_, is_last,
                              literal_context_mode, params, prev_byte,
                              prev_byte2, num_literals, num_commands, commands,
                              saved_dist_cache, dist_cache, w):
    """WriteMetaBlockInternal: writes into BitWriter ``w`` (which holds the
    pending last_bytes as its initial bits) and returns the writer to use
    afterwards (a fresh one if the output was rolled back)."""
    wrapped_last_flush_pos = wrap_position(last_flush_pos)
    literal_context_lut = literal_context_mode << 9
    block_params = params.copy()

    if bytes_ == 0:
        # Write the ISLAST and ISEMPTY bits.
        w.write_bits(2, 3)
        w.jump_to_byte_boundary()
        return w

    if not should_compress(data, mask, last_flush_pos, bytes_, num_literals,
                           num_commands):
        # Restore the distance cache, as its last update by
        # CreateBackwardReferences is now unused.
        dist_cache[0:4] = saved_dist_cache[0:4]
        brotli_store_uncompressed_meta_block(is_last, data,
                                             wrapped_last_flush_pos, mask,
                                             bytes_, w)
        return w

    # BROTLI_DCHECK(*storage_ix <= 14)
    last_bytes = w.acc
    last_bytes_bits = w.nbits
    if params.quality <= MAX_QUALITY_FOR_STATIC_ENTROPY_CODES:
        brotli_store_meta_block_fast(data, wrapped_last_flush_pos, bytes_,
                                     mask, is_last, params, commands,
                                     num_commands, w)
    elif params.quality < MIN_QUALITY_FOR_BLOCK_SPLIT:
        brotli_store_meta_block_trivial(data, wrapped_last_flush_pos, bytes_,
                                        mask, is_last, params, commands,
                                        num_commands, w)
    else:
        mb = MetaBlockSplit()
        if params.quality < MIN_QUALITY_FOR_HQ_BLOCK_SPLITTING:
            num_literal_contexts = 1
            literal_context_map = None
            if not params.disable_literal_context_modeling:
                # TODO(eustas): pull to higher level and reuse.
                num_literal_contexts, literal_context_map = \
                    decide_over_literal_context_modeling(
                        data, wrapped_last_flush_pos, bytes_, mask,
                        params.quality, params.size_hint,
                        num_literal_contexts, literal_context_map)
            brotli_build_meta_block_greedy(
                data, wrapped_last_flush_pos, mask, prev_byte, prev_byte2,
                literal_context_lut, num_literal_contexts,
                literal_context_map, commands, num_commands, mb)
        else:
            from .metablock_hq import brotli_build_meta_block
            brotli_build_meta_block(data, wrapped_last_flush_pos, mask,
                                    block_params, prev_byte, prev_byte2,
                                    commands, num_commands,
                                    literal_context_mode, mb)
        if params.quality >= MIN_QUALITY_FOR_OPTIMIZE_HISTOGRAMS:
            # The number of distance symbols effectively used for distance
            # histograms. It might be less than distance alphabet size
            # for "Large Window Brotli" (32-bit).
            brotli_optimize_histograms(block_params.dist.alphabet_size_limit, mb)
        brotli_store_meta_block(data, wrapped_last_flush_pos, bytes_, mask,
                                prev_byte, prev_byte2, is_last, block_params,
                                literal_context_mode, commands, num_commands,
                                mb, w)
    if bytes_ + 4 < (w.storage_ix >> 3):
        # Restore the distance cache and last byte.
        dist_cache[0:4] = saved_dist_cache[0:4]
        w = BitWriter(last_bytes, last_bytes_bits)
        brotli_store_uncompressed_meta_block(is_last, data,
                                             wrapped_last_flush_pos, mask,
                                             bytes_, w)
    return w


def choose_distance_params(params):
    """ChooseDistanceParams."""
    distance_postfix_bits = 0
    num_direct_distance_codes = 0
    if params.quality >= MIN_QUALITY_FOR_NONZERO_DISTANCE_PARAMS:
        if params.mode == MODE_FONT:
            distance_postfix_bits = 1
            num_direct_distance_codes = 12
        else:
            distance_postfix_bits = params.dist.distance_postfix_bits
            num_direct_distance_codes = params.dist.num_direct_distance_codes
        ndirect_msb = (num_direct_distance_codes >> distance_postfix_bits) & 0x0F
        if (distance_postfix_bits > BROTLI_MAX_NPOSTFIX or
                num_direct_distance_codes > BROTLI_MAX_NDIRECT or
                (ndirect_msb << distance_postfix_bits) !=
                num_direct_distance_codes):
            distance_postfix_bits = 0
            num_direct_distance_codes = 0
    brotli_init_distance_params(params.dist, distance_postfix_bits,
                                num_direct_distance_codes, params.large_window)


# ---------------------------------------------------------------------------
# Encoder state


class BrotliEncoderState(object):
    """BrotliEncoderStateStruct (state.h) + the streaming functions of
    encode.c."""

    def __init__(self):
        # BrotliEncoderInitState
        self.params = BrotliEncoderParams()
        self.input_pos = 0
        self.commands = []
        self.num_literals = 0
        self.last_insert_len = 0
        self.last_flush_pos = 0
        self.last_processed_pos = 0
        self.prev_byte = 0
        self.prev_byte2 = 0
        self.hasher = Hasher()
        self.total_in = 0
        self.total_out = 0
        self.stream_state = BROTLI_STREAM_PROCESSING
        self.is_last_block_emitted = False
        self.is_initialized = False
        self.ringbuffer = RingBuffer()
        self.last_bytes = 0
        self.last_bytes_bits = 0
        self.flint = BROTLI_FLINT_DONE
        self.remaining_metadata_bytes = BROTLI_UINT32_MAX
        # Pending output (C: next_out_ / available_out_).
        self.pending = b''
        self.one_pass_arena = None
        self.two_pass_arena = None
        # Initialize distance cache.
        self.dist_cache = [4, 11, 15, 16] + [0] * 12
        # Save the state of the distance cache in case we need to restore it
        # for emitting an uncompressed block.
        self.saved_dist_cache = self.dist_cache[0:4]

    # --- parameters -------------------------------------------------------

    def set_parameter(self, name, value):
        """BrotliEncoderSetParameter (subset)."""
        # Changing parameters on the fly is not implemented yet.
        if self.is_initialized:
            return False
        p = self.params
        if name == 'mode':
            p.mode = value
        elif name == 'quality':
            p.quality = value
        elif name == 'lgwin':
            p.lgwin = value
        elif name == 'lgblock':
            p.lgblock = value
        elif name == 'disable_literal_context_modeling':
            if value not in (0, 1):
                return False
            p.disable_literal_context_modeling = bool(value)
        elif name == 'size_hint':
            p.size_hint = value
        elif name == 'large_window':
            p.large_window = bool(value)
        elif name == 'npostfix':
            p.dist.distance_postfix_bits = value
        elif name == 'ndirect':
            p.dist.num_direct_distance_codes = value
        else:
            return False
        return True

    def ensure_initialized(self):
        """EnsureInitialized."""
        if self.is_initialized:
            return True
        self.last_bytes_bits = 0
        self.last_bytes = 0
        self.flint = BROTLI_FLINT_DONE
        self.remaining_metadata_bytes = BROTLI_UINT32_MAX

        p = self.params
        sanitize_params(p)
        p.lgblock = compute_lg_block(p)
        choose_distance_params(p)

        if p.quality not in SUPPORTED_QUALITIES:
            raise UnsupportedQualityError(
                'brotli_pure: quality %d is not supported (supported: %s)' %
                (p.quality, ', '.join(str(q) for q in SUPPORTED_QUALITIES)))

        self.ringbuffer.setup(p)

        # Initialize last byte with stream header.
        lgwin = p.lgwin
        if (p.quality == FAST_ONE_PASS_COMPRESSION_QUALITY or
                p.quality == FAST_TWO_PASS_COMPRESSION_QUALITY):
            lgwin = max(lgwin, 18)
        self.last_bytes, self.last_bytes_bits = encode_window_bits(
            lgwin, p.large_window)

        if p.quality == FAST_ONE_PASS_COMPRESSION_QUALITY:
            from .compress_fragment import (BrotliOnePassArena,
                                            init_command_prefix_codes)
            self.one_pass_arena = BrotliOnePassArena()
            init_command_prefix_codes(self.one_pass_arena)
        elif p.quality == FAST_TWO_PASS_COMPRESSION_QUALITY:
            from .compress_fragment_two_pass import BrotliTwoPassArena
            self.two_pass_arena = BrotliTwoPassArena()

        self.is_initialized = True
        return True

    # --- input ------------------------------------------------------------

    def input_block_size(self):
        """InputBlockSize."""
        return 1 << self.params.lgblock

    def unprocessed_input_size(self):
        """UnprocessedInputSize."""
        return self.input_pos - self.last_processed_pos

    def remaining_input_block_size(self):
        """RemainingInputBlockSize."""
        delta = self.unprocessed_input_size()
        block_size = self.input_block_size()
        if delta >= block_size:
            return 0
        return block_size - delta

    def copy_input_to_ring_buffer(self, input_size, input_buffer, start):
        """CopyInputToRingBuffer: copies input_buffer[start:start+input_size]
        to the internal ring buffer."""
        rb = self.ringbuffer
        rb.write(input_buffer, start, input_size)
        self.input_pos += input_size
        # If needed, initialize 7 more bytes in the ring buffer to make the
        # hashing not depend on uninitialized data (see encode.c).  Only
        # clear during the first round of ring-buffer writes. On subsequent
        # rounds data in the ring-buffer would be affected.
        if rb.pos <= rb.mask:
            buf = rb.buffer
            for i in range(rb.pos, rb.pos + 7):
                buf[i] = 0

    def update_last_processed_pos(self):
        """UpdateLastProcessedPos: marks all input as processed.  Returns
        True if position wrapping occurs."""
        wrapped_last_processed_pos = wrap_position(self.last_processed_pos)
        wrapped_input_pos = wrap_position(self.input_pos)
        self.last_processed_pos = self.input_pos
        return wrapped_input_pos < wrapped_last_processed_pos

    def extend_last_command(self, bytes_, wrapped_last_processed_pos):
        """ExtendLastCommand; returns the updated
        (bytes, wrapped_last_processed_pos)."""
        last_command = self.commands[-1]
        data = self.ringbuffer.buffer
        mask = self.ringbuffer.mask
        max_backward_distance = (1 << self.params.lgwin) - BROTLI_WINDOW_GAP
        last_copy_len = last_command.copy_len & 0x1FFFFFF
        last_processed_pos = self.last_processed_pos - last_copy_len
        max_distance = (last_processed_pos
                        if last_processed_pos < max_backward_distance
                        else max_backward_distance)
        cmd_dist = self.dist_cache[0]
        distance_code = command_restore_distance_code(last_command,
                                                      self.params.dist)
        compound_dictionary_size = self.params.dictionary.compound.total_size
        if (distance_code < BROTLI_NUM_DISTANCE_SHORT_CODES or
                distance_code - (BROTLI_NUM_DISTANCE_SHORT_CODES - 1) == cmd_dist):
            if cmd_dist <= max_distance:
                while (bytes_ != 0 and
                       data[wrapped_last_processed_pos & mask] ==
                       data[(wrapped_last_processed_pos - cmd_dist) & mask]):
                    last_command.copy_len += 1
                    bytes_ -= 1
                    wrapped_last_processed_pos += 1
            elif ((cmd_dist - max_distance - 1) < compound_dictionary_size and
                  last_copy_len < cmd_dist - max_distance):
                raise NotImplementedError('compound dictionaries')
            # The copy length is at most the metablock size, and thus
            # expressible.
            # NB: the C code adds the raw 7-bit delta (copy_len_ >> 25)
            # without sign extension, see docs/notes/enc-q5.md.
            last_command.cmd_prefix = get_length_code(
                last_command.insert_len,
                (last_command.copy_len & 0x1FFFFFF) +
                (last_command.copy_len >> 25),
                (last_command.dist_prefix & 0x3FF) == 0)
        return bytes_, wrapped_last_processed_pos

    # --- EncodeData ---------------------------------------------------------

    def encode_data(self, is_last, force_flush):
        """EncodeData: processes the accumulated input data and returns the
        bytes of the new output meta-block (b'' if the processed input data
        is buffered internally).  Until is_last is True the encoder may
        retain up to 7 bits of the last byte of output (last_bytes)."""
        p = self.params
        delta = self.unprocessed_input_size()
        bytes_ = delta & 0xFFFFFFFF
        wrapped_last_processed_pos = wrap_position(self.last_processed_pos)
        rb = self.ringbuffer
        data = rb.buffer
        mask = rb.mask
        fast_compress = (p.quality == FAST_ONE_PASS_COMPRESSION_QUALITY or
                         p.quality == FAST_TWO_PASS_COMPRESSION_QUALITY)

        if delta == 0:  # No new input; still might want to flush or finish.
            if data is None:  # No input has been processed so far.
                if is_last:  # Emit complete finalized stream.
                    self.last_bytes |= 3 << self.last_bytes_bits
                    self.last_bytes_bits += 2
                    out = bytes([self.last_bytes & 0xFF,
                                 (self.last_bytes >> 8) & 0xFF])
                    return out[:(self.last_bytes_bits + 7) >> 3]
                else:  # No data, not last -> no-op.
                    return b''
            else:
                # Fast compress performs flush every block -> flush is no-op.
                if not is_last and (not force_flush or fast_compress):
                    return b''  # Another no-op.

        if p.quality > p.dictionary.max_quality:
            raise BrotliError('brotli: encoder failed')
        # Adding more blocks after "last" block is forbidden.
        if self.is_last_block_emitted:
            raise BrotliError('brotli: encoder failed')
        if is_last:
            self.is_last_block_emitted = True

        if delta > self.input_block_size():
            raise BrotliError('brotli: encoder failed')
        if fast_compress:
            w = BitWriter(self.last_bytes, self.last_bytes_bits)
            table = get_hash_table(p.quality, bytes_)
            start = wrapped_last_processed_pos & mask
            if p.quality == FAST_ONE_PASS_COMPRESSION_QUALITY:
                from .compress_fragment import brotli_compress_fragment_fast
                brotli_compress_fragment_fast(
                    self.one_pass_arena, data, start, bytes_, is_last, table,
                    len(table), w)
            else:
                from .compress_fragment_two_pass import \
                    brotli_compress_fragment_two_pass
                brotli_compress_fragment_two_pass(
                    self.two_pass_arena, data, start, bytes_, is_last, table,
                    len(table), w)
            out, self.last_bytes, self.last_bytes_bits = w.get_output()
            self.update_last_processed_pos()
            return out

        init_or_stitch_to_previous_block(self.hasher, data, mask, p,
                                         wrapped_last_processed_pos, bytes_,
                                         is_last)

        literal_context_mode = choose_context_mode(
            p, data, wrap_position(self.last_flush_pos), mask,
            self.input_pos - self.last_flush_pos)
        literal_context_lut = literal_context_mode << 9

        if self.commands and self.last_insert_len == 0:
            bytes_, wrapped_last_processed_pos = self.extend_last_command(
                bytes_, wrapped_last_processed_pos)

        if p.quality == ZOPFLIFICATION_QUALITY:
            from .backward_references_hq import \
                brotli_create_zopfli_backward_references as create_refs
        elif p.quality == HQ_ZOPFLIFICATION_QUALITY:
            from .backward_references_hq import \
                brotli_create_hq_zopfli_backward_references as create_refs
        else:
            create_refs = None
        if create_refs is not None:
            # (num_bytes, position, ringbuffer, ringbuffer_mask,
            #  literal_context_lut, params, hasher, dist_cache,
            #  last_insert_len, commands, num_commands, num_literals)
            #  -> (last_insert_len, num_commands, num_literals)
            result = create_refs(
                bytes_, wrapped_last_processed_pos, data, mask,
                literal_context_lut, p, self.hasher, self.dist_cache,
                self.last_insert_len, self.commands, len(self.commands),
                self.num_literals)
            self.last_insert_len = result[0]
            self.num_literals = result[-1]
        else:
            self.last_insert_len, self.num_literals = \
                brotli_create_backward_references(
                    bytes_, wrapped_last_processed_pos, data, mask,
                    literal_context_lut, p, self.hasher, self.dist_cache,
                    self.last_insert_len, self.commands, self.num_literals)

        max_length = max_metablock_size(p)
        max_literals = max_length // 8
        max_commands = max_length // 8
        processed_bytes = self.input_pos - self.last_flush_pos
        # If maximal possible additional block doesn't fit metablock, flush
        # now.
        # TODO(eustas): Postpone decision until next block arrives?
        next_input_fits_metablock = (
            processed_bytes + self.input_block_size() <= max_length)
        # If block splitting is not used, then flush as soon as there is some
        # amount of commands / literals produced.
        should_flush = (p.quality < MIN_QUALITY_FOR_BLOCK_SPLIT and
                        self.num_literals + len(self.commands) >=
                        MAX_NUM_DELAYED_SYMBOLS)
        if (not is_last and not force_flush and not should_flush and
                next_input_fits_metablock and
                self.num_literals < max_literals and
                len(self.commands) < max_commands):
            # Merge with next input block. Everything will happen later.
            if self.update_last_processed_pos():
                hasher_reset(self.hasher)
            return b''

        # Create the last insert-only command.
        if self.last_insert_len > 0:
            self.commands.append(init_insert_command(self.last_insert_len))
            self.num_literals += self.last_insert_len
            self.last_insert_len = 0

        if not is_last and self.input_pos == self.last_flush_pos:
            # We have no new input data and we don't have to finish the
            # stream, so nothing to do.
            return b''

        metablock_size = self.input_pos - self.last_flush_pos
        w = BitWriter(self.last_bytes, self.last_bytes_bits)
        w = write_meta_block_internal(
            data, mask, self.last_flush_pos, metablock_size, is_last,
            literal_context_mode, p, self.prev_byte, self.prev_byte2,
            self.num_literals, len(self.commands), self.commands,
            self.saved_dist_cache, self.dist_cache, w)
        out, self.last_bytes, self.last_bytes_bits = w.get_output()
        self.last_flush_pos = self.input_pos
        if self.update_last_processed_pos():
            hasher_reset(self.hasher)
        if self.last_flush_pos > 0:
            self.prev_byte = data[(self.last_flush_pos - 1) & mask]
        if self.last_flush_pos > 1:
            self.prev_byte2 = data[(self.last_flush_pos - 2) & mask]
        self.commands = []
        self.num_literals = 0
        # Save the state of the distance cache in case we need to restore it
        # for emitting an uncompressed block.
        self.saved_dist_cache = self.dist_cache[0:4]
        return out

    # --- streaming ----------------------------------------------------------

    def update_size_hint(self, available_in):
        """UpdateSizeHint."""
        if self.params.size_hint == 0:
            delta = self.unprocessed_input_size()
            tail = available_in
            limit = 1 << 30
            if delta >= limit or tail >= limit or (delta + tail) >= limit:
                total = limit
            else:
                total = delta + tail
            self.params.size_hint = total

    def _inject_byte_padding_block(self):
        """InjectBytePaddingBlock: appends an empty metadata block that
        byte-aligns the stream to the pending output."""
        seal = self.last_bytes
        seal_bits = self.last_bytes_bits
        self.last_bytes = 0
        self.last_bytes_bits = 0
        # is_last = 0, data_nibbles = 11, reserved = 0, meta_nibbles = 00
        seal |= 0x6 << seal_bits
        seal_bits += 6
        out = bytearray([seal & 0xFF])
        if seal_bits > 8:
            out.append((seal >> 8) & 0xFF)
        if seal_bits > 16:
            out.append((seal >> 16) & 0xFF)
        self.pending += bytes(out)

    def compress_stream_fast(self, op, data):
        """BrotliEncoderCompressStreamFast (qualities 0 and 1): compresses
        the caller's input directly, in blocks of at most 1 << lgwin bytes."""
        p = self.params
        block_size_limit = 1 << p.lgwin
        next_in = 0
        available_in = len(data)
        output = []
        if p.quality == FAST_ONE_PASS_COMPRESSION_QUALITY:
            from .compress_fragment import brotli_compress_fragment_fast \
                as fragment
            arena = self.one_pass_arena
        else:
            from .compress_fragment_two_pass import \
                brotli_compress_fragment_two_pass as fragment
            arena = self.two_pass_arena
        while True:
            # InjectFlushOrPushOutput
            if (self.stream_state == BROTLI_STREAM_FLUSH_REQUESTED and
                    self.last_bytes_bits != 0):
                self._inject_byte_padding_block()
                continue
            if self.pending:
                output.append(self.pending)
                self.total_out += len(self.pending)
                self.pending = b''
                continue
            # Compress block only when internal output buffer is empty, stream
            # is not finished, there is no pending flush request, and there is
            # either additional input or pending operation.
            if (self.stream_state == BROTLI_STREAM_PROCESSING and
                    (available_in != 0 or op != BROTLI_OPERATION_PROCESS)):
                block_size = min(block_size_limit, available_in)
                is_last = (available_in == block_size and
                           op == BROTLI_OPERATION_FINISH)
                force_flush = (available_in == block_size and
                               op == BROTLI_OPERATION_FLUSH)
                if force_flush and block_size == 0:
                    self.stream_state = BROTLI_STREAM_FLUSH_REQUESTED
                    continue
                w = BitWriter(self.last_bytes, self.last_bytes_bits)
                table = get_hash_table(p.quality, block_size)
                fragment(arena, data, next_in, block_size, is_last, table,
                         len(table), w)
                if block_size != 0:
                    next_in += block_size
                    available_in -= block_size
                    self.total_in += block_size
                out, self.last_bytes, self.last_bytes_bits = w.get_output()
                self.pending = out
                if force_flush:
                    self.stream_state = BROTLI_STREAM_FLUSH_REQUESTED
                if is_last:
                    self.stream_state = BROTLI_STREAM_FINISHED
                continue
            break
        # CheckFlushComplete
        if (self.stream_state == BROTLI_STREAM_FLUSH_REQUESTED and
                not self.pending):
            self.stream_state = BROTLI_STREAM_PROCESSING
        return b''.join(output)

    def compress_stream(self, op, data=b''):
        """BrotliEncoderCompressStream for one call with all of ``data``
        available and an unbounded output buffer; returns the produced
        bytes."""
        self.ensure_initialized()
        if op not in (BROTLI_OPERATION_PROCESS, BROTLI_OPERATION_FLUSH,
                      BROTLI_OPERATION_FINISH):
            raise BrotliError('brotli: unsupported operation')
        if not isinstance(data, (bytes, bytearray)):
            data = bytes(memoryview(data))
        next_in = 0
        available_in = len(data)
        if self.stream_state != BROTLI_STREAM_PROCESSING and available_in != 0:
            raise BrotliError('brotli: encoder failed')
        if (self.params.quality == FAST_ONE_PASS_COMPRESSION_QUALITY or
                self.params.quality == FAST_TWO_PASS_COMPRESSION_QUALITY):
            return self.compress_stream_fast(op, data)
        output = []
        while True:
            remaining_block_size = self.remaining_input_block_size()
            # Shorten input to flint size.
            if self.flint >= 0 and remaining_block_size > self.flint:
                remaining_block_size = self.flint

            if remaining_block_size != 0 and available_in != 0:
                copy_input_size = min(remaining_block_size, available_in)
                self.copy_input_to_ring_buffer(copy_input_size, data, next_in)
                next_in += copy_input_size
                available_in -= copy_input_size
                self.total_in += copy_input_size
                if self.flint > 0:
                    self.flint -= copy_input_size
                continue

            # InjectFlushOrPushOutput
            if (self.stream_state == BROTLI_STREAM_FLUSH_REQUESTED and
                    self.last_bytes_bits != 0):
                self._inject_byte_padding_block()
                continue
            if self.pending:
                output.append(self.pending)
                self.total_out += len(self.pending)
                self.pending = b''
                continue

            # Compress data only when internal output buffer is empty, stream
            # is not finished and there is no pending flush request.
            if self.stream_state == BROTLI_STREAM_PROCESSING:
                if remaining_block_size == 0 or op != BROTLI_OPERATION_PROCESS:
                    is_last = (available_in == 0 and
                               op == BROTLI_OPERATION_FINISH)
                    force_flush = (available_in == 0 and
                                   op == BROTLI_OPERATION_FLUSH)
                    self.update_size_hint(available_in)
                    self.pending = self.encode_data(is_last, force_flush)
                    if force_flush:
                        self.stream_state = BROTLI_STREAM_FLUSH_REQUESTED
                    if is_last:
                        self.stream_state = BROTLI_STREAM_FINISHED
                    continue
            break
        # CheckFlushComplete
        if (self.stream_state == BROTLI_STREAM_FLUSH_REQUESTED and
                not self.pending):
            self.stream_state = BROTLI_STREAM_PROCESSING
        return b''.join(output)

    def is_finished(self):
        """BrotliEncoderIsFinished."""
        return self.stream_state == BROTLI_STREAM_FINISHED and not self.pending


# ---------------------------------------------------------------------------
# Python-binding compatible API (python/_brotli.c, python/brotli.py)


def _check_byte_param(name, value):
    # PyArg_ParseTuple "b" format: unsigned char.
    if isinstance(value, bool) or not hasattr(value, '__index__'):
        raise TypeError("'%s' must be an integer, not %s" %
                        (name, type(value).__name__))
    value = value.__index__()
    if value < 0:
        raise OverflowError('unsigned byte integer is less than minimum')
    if value > 255:
        raise OverflowError('unsigned byte integer is greater than maximum')
    return value


class Compressor(object):
    """Port of brotli.Compressor (python/_brotli.c)."""

    def __init__(self, mode=MODE_GENERIC, quality=11, lgwin=22, lgblock=0):
        mode = _check_byte_param('mode', mode)
        quality = _check_byte_param('quality', quality)
        lgwin = _check_byte_param('lgwin', lgwin)
        lgblock = _check_byte_param('lgblock', lgblock)
        self._state = BrotliEncoderState()
        if mode not in (MODE_GENERIC, MODE_TEXT, MODE_FONT):
            raise BrotliError('brotli: invalid mode')
        self._state.set_parameter('mode', mode)
        if quality > 11:
            raise BrotliError('brotli: invalid quality; range is 0 to 11')
        self._state.set_parameter('quality', quality)
        if not (MIN_WINDOW_BITS <= lgwin <= MAX_WINDOW_BITS):
            raise BrotliError('brotli: invalid lgwin; range is 10 to 24')
        self._state.set_parameter('lgwin', lgwin)
        if not (lgblock == 0 or
                MIN_INPUT_BLOCK_BITS <= lgblock <= MAX_INPUT_BLOCK_BITS):
            raise BrotliError(
                'brotli: invalid lgblock; range is 16 to 24, or 0')
        self._state.set_parameter('lgblock', lgblock)
        if quality not in SUPPORTED_QUALITIES:
            raise UnsupportedQualityError(
                'brotli_pure: quality %d is not supported (supported: %s)' %
                (quality, ', '.join(str(q) for q in SUPPORTED_QUALITIES)))

    def process(self, data):
        return self._state.compress_stream(BROTLI_OPERATION_PROCESS, data)

    def flush(self):
        return self._state.compress_stream(BROTLI_OPERATION_FLUSH)

    def finish(self):
        out = self._state.compress_stream(BROTLI_OPERATION_FINISH)
        assert self._state.is_finished()
        return out


def compress(data, mode=MODE_GENERIC, quality=11, lgwin=22, lgblock=0):
    """brotli.compress: ``Compressor(...).process(data) + .finish()``."""
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(memoryview(data))
    c = Compressor(mode=mode, quality=quality, lgwin=lgwin, lgblock=lgblock)
    return c.process(data) + c.finish()
