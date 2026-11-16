"""Brotli decoder (port of c/dec/decode.c, google/brotli v1.2.1).

Entry point: ``decompress(data) -> bytes``.

Structure
---------
``BrotliDecoderDecompressStream`` in C is a resumable state machine that can
stop whenever input or output runs out.  Since the whole input is available
here, every stage is run to completion, so the state machine becomes
straight-line code (``_decompress_stream``).  The decoding algorithms
themselves (Huffman table construction, code length decoding, context maps
with RLE + inverse move-to-front, block switching, distance ring buffer,
static dictionary + transforms, metadata/uncompressed meta-blocks, padding
checks, ...) follow the C code step by step; functions are named after
their C counterparts.

Ring buffer
-----------
C decodes into a ring buffer of at most ``1 << window_bits`` bytes.  All
reads from the ring buffer (back-references, literal context bytes) are
restricted to distances <= ``max_distance`` = min(bytes produced so far,
window size - 16), so the ring buffer always holds exactly the bytes a
linear output buffer would hold at those positions.  The C decoder
initialises the two bytes "before" the stream start with zeros (the last two
ring buffer bytes), which is reproduced by treating missing context bytes
as 0.  We therefore decode straight into one growing ``bytearray``.

Error semantics (python/_brotli.c ``brotli_decompress``)
-------------------------------------------------------
The oracle fails unless ``BrotliDecoderDecompressStream`` returns SUCCESS
*and* consumed all input.  So: every format error, any truncation (C:
NEEDS_MORE_INPUT with no more input), and any trailing byte after the last
meta-block raise ``BrotliError``.  Large-window streams are rejected
(the binding never enables BROTLI_DECODER_PARAM_LARGE_WINDOW).

Two deliberate short-cuts that do not change results:

* When an insert or copy would make ``meta_block_remaining_len`` negative,
  C keeps decoding until the next check (WriteRingBuffer /
  METABLOCK_DONE), which always fails.  We fail immediately.
* A command that consumes no input bits and emits a zero-length
  transformed dictionary word leaves the decoder state unchanged except
  for the block counters, so C repeats it until a block counter runs out
  (up to 2**24 times).  We skip those identical iterations in one step
  (see docs/notes/decoder.md).
"""

import os

from ..common.constants import MAX_WINDOW_BITS  # noqa: F401
from ..common.context import CONTEXT_LOOKUP, context_lut
from ..common.dictionary import (DATA as _DICT_DATA, OFFSETS_BY_LENGTH,
                                 SIZE_BITS_BY_LENGTH)
from ..common.errors import BrotliError
from ..common.transforms import RFC_TRANSFORMS, transform_dictionary_word
from .bit_reader import BIT_MASK, BrotliBitReader, _overrun  # noqa: F401
from .huffman import (HUFFMAN_MAX_CODE_LENGTH, HUFFMAN_MAX_SIZE_26,
                      HUFFMAN_MAX_SIZE_258, HUFFMAN_MAX_SIZE_272,
                      build_code_lengths_huffman_table, build_huffman_table,
                      build_simple_huffman_table)
from .prefix import (BLOCK_LENGTH_NBITS, BLOCK_LENGTH_OFFSET, CMD_LUT,
                     NUM_BLOCK_LEN_SYMBOLS, NUM_COMMAND_SYMBOLS)
from .state import BrotliDecoderState, HuffmanTreeGroup

# Debug switch: when set, internal Python exceptions are not converted into
# BrotliError (used by the test-suite to make sure none ever happen).
_STRICT = bool(os.environ.get("BROTLI_PURE_STRICT"))

HUFFMAN_TABLE_BITS = 8
HUFFMAN_TABLE_MASK = 0xFF

NUM_LITERAL_SYMBOLS = 256
LITERAL_CONTEXT_BITS = 6
DISTANCE_CONTEXT_BITS = 2
NUM_DISTANCE_SHORT_CODES = 16
MAX_DISTANCE_BITS = 24
MAX_ALLOWED_DISTANCE = 0x7FFFFFFC
WINDOW_GAP = 16
REPEAT_PREVIOUS_CODE_LENGTH = 16
REPEAT_ZERO_CODE_LENGTH = 17
CODE_LENGTH_CODES = REPEAT_ZERO_CODE_LENGTH + 1
INITIAL_REPEATED_CODE_LENGTH = 8
SHARED_BROTLI_MIN_DICTIONARY_WORD_LENGTH = 4
SHARED_BROTLI_MAX_DICTIONARY_WORD_LENGTH = 31

kCodeLengthCodeOrder = (
    1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15)

# Static prefix code for the complex code length code lengths.
kCodeLengthPrefixLength = (2, 2, 2, 3, 2, 2, 2, 4, 2, 2, 2, 3, 2, 2, 2, 4)
kCodeLengthPrefixValue = (0, 4, 3, 2, 0, 4, 3, 1, 0, 4, 3, 2, 0, 4, 3, 5)

_TRANSFORMS = RFC_TRANSFORMS
_NUM_TRANSFORMS = RFC_TRANSFORMS.num_transforms
_IDENTITY_TRANSFORM = RFC_TRANSFORMS.cut_off_transforms[0]

# Per context mode: the two 256-entry halves of the context lookup table.
_CONTEXT_LUT_P1 = tuple(bytes(CONTEXT_LOOKUP[context_lut(m):context_lut(m) + 256])
                        for m in range(4))
_CONTEXT_LUT_P2 = tuple(bytes(CONTEXT_LOOKUP[context_lut(m) + 256:context_lut(m) + 512])
                        for m in range(4))


def _fail(code):
    raise BrotliError("brotli: decoder failed (BROTLI_DECODER_ERROR_" + code + ")")


def _distance_alphabet_size(npostfix, ndirect, maxnbits):
    """BROTLI_DISTANCE_ALPHABET_SIZE (constants.h)."""
    return NUM_DISTANCE_SHORT_CODES + ndirect + (maxnbits << (npostfix + 1))


# ---------------------------------------------------------------------------
# Stream / meta-block headers
# ---------------------------------------------------------------------------

def decode_window_bits(s, br):
    """DecodeWindowBits: reads 1..7 bits (large-window mode is never
    enabled by the Python binding, so its signature is an error)."""
    large_window = s.large_window
    s.large_window = False
    n = br.read_bits(1)
    if n == 0:
        s.window_bits = 16
        return
    n = br.read_bits(3)
    if n != 0:
        s.window_bits = (17 + n) & 63
        return
    n = br.read_bits(3)
    if n == 1:
        if large_window:  # pragma: no cover - not reachable via public API
            if br.read_bits(1) == 1:
                _fail("FORMAT_WINDOW_BITS")
            s.large_window = True
            return
        _fail("FORMAT_WINDOW_BITS")
    if n != 0:
        s.window_bits = (8 + n) & 63
        return
    s.window_bits = 17


def decode_var_len_uint8(br):
    """DecodeVarLenUint8: decodes a number in [0..255] (1..11 bits)."""
    if br.read_bits(1) == 0:
        return 0
    n = br.read_bits(3)
    if n == 0:
        return 1
    return (1 << n) + br.read_bits(n)


def decode_meta_block_length(s, br):
    """DecodeMetaBlockLength: meta-block length and flags (2..31 bits)."""
    s.is_last_metablock = br.read_bits(1)
    s.meta_block_remaining_len = 0
    s.is_uncompressed = 0
    s.is_metadata = 0
    if s.is_last_metablock:
        if br.read_bits(1):  # ISEMPTY
            return
    bits = br.read_bits(2)
    s.size_nibbles = bits + 4
    if bits == 3:
        s.is_metadata = 1
        # BROTLI_STATE_METABLOCK_HEADER_RESERVED
        if br.read_bits(1) != 0:
            _fail("FORMAT_RESERVED")
        # BROTLI_STATE_METABLOCK_HEADER_BYTES
        size_bytes = br.read_bits(2)
        if size_bytes == 0:
            return
        s.size_nibbles = size_bytes
        # BROTLI_STATE_METABLOCK_HEADER_METADATA
        length = 0
        for i in range(size_bytes):
            bits = br.read_bits(8)
            if i + 1 == size_bytes and size_bytes > 1 and bits == 0:
                _fail("FORMAT_EXUBERANT_META_NIBBLE")
            length |= bits << (i * 8)
        s.meta_block_remaining_len = length + 1
        return
    # BROTLI_STATE_METABLOCK_HEADER_SIZE
    size_nibbles = s.size_nibbles
    length = 0
    for i in range(size_nibbles):
        bits = br.read_bits(4)
        if i + 1 == size_nibbles and size_nibbles > 4 and bits == 0:
            _fail("FORMAT_EXUBERANT_NIBBLE")
        length |= bits << (i * 4)
    # BROTLI_STATE_METABLOCK_HEADER_UNCOMPRESSED
    if not s.is_last_metablock:
        s.is_uncompressed = br.read_bits(1)
    s.meta_block_remaining_len = length + 1


# ---------------------------------------------------------------------------
# Huffman code reading
# ---------------------------------------------------------------------------

def _log2_floor(x):
    """Log2Floor (decode.c) -- actually the bit length of x."""
    result = 0
    while x:
        x >>= 1
        result += 1
    return result


def read_simple_huffman_symbols(alphabet_size_max, alphabet_size_limit,
                                num_symbols, br):
    """ReadSimpleHuffmanSymbols: reads num_symbols + 1 symbols (1..4)."""
    max_bits = _log2_floor(alphabet_size_max - 1)
    symbols = []
    for _ in range(num_symbols + 1):
        v = br.read_bits(max_bits)
        if v >= alphabet_size_limit:
            _fail("FORMAT_SIMPLE_HUFFMAN_ALPHABET")
        symbols.append(v)
    for i in range(num_symbols):
        for k in range(i + 1, num_symbols + 1):
            if symbols[i] == symbols[k]:
                _fail("FORMAT_SIMPLE_HUFFMAN_SAME")
    return symbols


def read_code_length_code_lengths(skip, br):
    """ReadCodeLengthCodeLengths: reads 15..18 codes using the static
    prefix code (each 2..4 bits).  Returns (code_length_code_lengths,
    code_length_histo)."""
    code_length_code_lengths = [0] * CODE_LENGTH_CODES
    code_length_histo = [0] * (HUFFMAN_MAX_CODE_LENGTH + 1)
    num_codes = 0
    space = 32
    for i in range(skip, CODE_LENGTH_CODES):
        code_len_idx = kCodeLengthCodeOrder[i]
        ix = br.get_bits(4)
        v = kCodeLengthPrefixValue[ix]
        br.drop_bits(kCodeLengthPrefixLength[ix])
        code_length_code_lengths[code_len_idx] = v
        if v != 0:
            space -= 32 >> v
            num_codes += 1
            code_length_histo[v] += 1
            if space <= 0:  # C: (space - 1U >= 32U) -- 0 or wrapped around.
                break
    if not (num_codes == 1 or space == 0):
        _fail("FORMAT_CL_SPACE")
    return code_length_code_lengths, code_length_histo


def read_symbol_code_lengths(alphabet_size, table, br):
    """ReadSymbolCodeLengths / SafeReadSymbolCodeLengths, together with
    ProcessSingleCodeLength / ProcessRepeatedCodeLength (inlined).

    Returns (space, symbol_lists, code_length_histo).  symbol_lists[16 + k]
    corresponds to C's symbol_lists[k] (heads of the per-length linked
    lists live at C indices -16..-1)."""
    off = HUFFMAN_MAX_CODE_LENGTH + 1
    symbol_lists = [0] * (off + alphabet_size)
    code_length_histo = [0] * (HUFFMAN_MAX_CODE_LENGTH + 1)
    next_symbol = [i - off for i in range(HUFFMAN_MAX_CODE_LENGTH + 1)]
    for i in range(HUFFMAN_MAX_CODE_LENGTH + 1):
        symbol_lists[off + next_symbol[i]] = 0xFFFF
    symbol = 0
    prev_code_len = INITIAL_REPEATED_CODE_LENGTH
    repeat = 0
    repeat_code_len = 0
    space = 32768
    while symbol < alphabet_size and space > 0:
        bits = br.get_bits(8)  # 5 bits code + up to 3 extra bits
        p = table[bits & 0x1F]
        nb = p & 15
        code_len = p >> 4  # 0..17
        if code_len < REPEAT_PREVIOUS_CODE_LENGTH:
            br.drop_bits(nb)
            # ProcessSingleCodeLength
            repeat = 0
            if code_len != 0:
                symbol_lists[off + next_symbol[code_len]] = symbol
                next_symbol[code_len] = symbol
                prev_code_len = code_len
                space -= 32768 >> code_len
                code_length_histo[code_len] += 1
            symbol += 1
        else:
            extra_bits = code_len - 14  # 2 or 3
            repeat_delta = (bits >> nb) & BIT_MASK[extra_bits]
            br.drop_bits(nb + extra_bits)
            # ProcessRepeatedCodeLength
            new_len = 0
            if code_len == REPEAT_PREVIOUS_CODE_LENGTH:
                new_len = prev_code_len
            if repeat_code_len != new_len:
                repeat = 0
                repeat_code_len = new_len
            old_repeat = repeat
            if repeat > 0:
                repeat -= 2
                repeat <<= extra_bits
            repeat += repeat_delta + 3
            repeat_delta = repeat - old_repeat
            if symbol + repeat_delta > alphabet_size:
                symbol = alphabet_size
                space = 0xFFFFF
                break
            if repeat_code_len != 0:
                last = symbol + repeat_delta
                nxt = next_symbol[repeat_code_len]
                while True:
                    symbol_lists[off + nxt] = symbol
                    nxt = symbol
                    symbol += 1
                    if symbol == last:
                        break
                next_symbol[repeat_code_len] = nxt
                space -= repeat_delta << (15 - repeat_code_len)
                code_length_histo[repeat_code_len] += repeat_delta
            else:
                symbol += repeat_delta
    return space, symbol_lists, code_length_histo


def read_huffman_code(alphabet_size_max, alphabet_size_limit, table_alloc, br):
    """ReadHuffmanCode: decodes one Huffman code and builds its decoding
    table.  Returns (table, table_size).

    A) "simple" codes with 1..4 symbols whose code lengths are implied;
    B) "complex" codes: a code length code (static prefix code) followed by
       the Huffman-coded code lengths of the symbols."""
    table = [0] * table_alloc
    hskip = br.read_bits(2)
    if hskip == 1:
        # BROTLI_STATE_HUFFMAN_SIMPLE_SIZE / _READ / _BUILD
        num_symbols = br.read_bits(2)
        symbols = read_simple_huffman_symbols(
            alphabet_size_max, alphabet_size_limit, num_symbols, br)
        if num_symbols == 3:
            num_symbols += br.read_bits(1)
        table_size = build_simple_huffman_table(
            table, HUFFMAN_TABLE_BITS, symbols, num_symbols)
        return table, table_size
    # BROTLI_STATE_HUFFMAN_COMPLEX
    code_length_code_lengths, code_length_histo = \
        read_code_length_code_lengths(hskip, br)
    cl_table = [0] * 32
    build_code_lengths_huffman_table(cl_table, code_length_code_lengths,
                                     code_length_histo)
    # BROTLI_STATE_HUFFMAN_LENGTH_SYMBOLS
    space, symbol_lists, code_length_histo = read_symbol_code_lengths(
        alphabet_size_limit, cl_table, br)
    if space != 0:
        _fail("FORMAT_HUFFMAN_SPACE")
    table_size = build_huffman_table(
        table, HUFFMAN_TABLE_BITS, symbol_lists, HUFFMAN_MAX_CODE_LENGTH + 1,
        code_length_histo)
    return table, table_size


def read_block_length(table, br):
    """ReadBlockLength / SafeReadBlockLength: 3..39 bits."""
    code = br.read_symbol(table)
    return BLOCK_LENGTH_OFFSET[code] + br.read_bits(BLOCK_LENGTH_NBITS[code])


def inverse_move_to_front_transform(v, s):
    """InverseMoveToFrontTransform.

    C re-initialises only the prefix of its 256-entry list that the previous
    call could have touched (mtf_upper_bound); the list content at entry is
    therefore always the identity permutation, which we use directly."""
    mtf = bytearray(range(256))
    upper_bound = 0
    for i in range(len(v)):
        index = v[i]
        value = mtf[index]
        upper_bound |= index
        v[i] = value
        if index:
            del mtf[index]
            mtf.insert(0, value)
    s.mtf_upper_bound = upper_bound >> 2


def huffman_tree_group_decode(group, br):
    """HuffmanTreeGroupDecode: decodes a series of Huffman tables."""
    # BrotliDecoderHuffmanTreeGroupInit: 376 = 256 (1st level table) +
    # 4 + 7 + 15 + 31 + 63 (2nd level mix-tables).
    table_alloc = group.alphabet_size_limit + 376
    htrees = group.htrees
    for i in range(group.num_htrees):
        htrees[i] = read_huffman_code(group.alphabet_size_max,
                                      group.alphabet_size_limit,
                                      table_alloc, br)[0]


def decode_context_map(context_map_size, s):
    """DecodeContextMap.  Returns (num_htrees, context_map).

    1) Read auxiliary information (6..16 bits);
    2) decode the Huffman table used for reading context map items;
    3) read context map items; "0" values could be run-length encoded;
    4) optionally, apply the inverse move-to-front transform."""
    br = s.br
    num_htrees = decode_var_len_uint8(br) + 1
    context_map = bytearray(context_map_size)
    if num_htrees <= 1:
        return num_htrees, context_map
    # BROTLI_STATE_CONTEXT_MAP_READ_PREFIX
    bits = br.get_bits(5)
    if bits & 1:  # Use RLE for zeros.
        max_run_length_prefix = (bits >> 1) + 1
        br.drop_bits(5)
    else:
        max_run_length_prefix = 0
        br.drop_bits(1)
    # BROTLI_STATE_CONTEXT_MAP_HUFFMAN
    alphabet_size = num_htrees + max_run_length_prefix
    table = read_huffman_code(alphabet_size, alphabet_size,
                              HUFFMAN_MAX_SIZE_272, br)[0]
    # BROTLI_STATE_CONTEXT_MAP_DECODE
    context_index = 0
    while context_index < context_map_size:
        code = br.read_symbol(table)
        if code == 0:
            context_index += 1
            continue
        if code > max_run_length_prefix:
            context_map[context_index] = code - max_run_length_prefix
            context_index += 1
            continue
        # RLE sub-stage.
        reps = br.read_bits(code) + (1 << code)
        if context_index + reps > context_map_size:
            _fail("FORMAT_CONTEXT_MAP_REPEAT")
        context_index += reps  # entries are already zero
    # BROTLI_STATE_CONTEXT_MAP_TRANSFORM
    if br.read_bits(1):
        inverse_move_to_front_transform(context_map, s)
    return num_htrees, context_map


# ---------------------------------------------------------------------------
# Block switching
# ---------------------------------------------------------------------------

def decode_block_type_and_length(s, tree_type):
    """DecodeBlockTypeAndLength: decodes a block type and length and
    updates the block type ring buffer (3..54 bits)."""
    max_block_type = s.num_block_types[tree_type]
    if max_block_type <= 1:
        _fail("FORMAT_BLOCK_SWITCH")
    br = s.br
    block_type = br.read_symbol(s.block_type_trees[tree_type])
    s.block_length[tree_type] = read_block_length(
        s.block_len_trees[tree_type], br)
    ringbuffer = s.block_type_rb
    rb = tree_type * 2
    if block_type == 1:
        block_type = ringbuffer[rb + 1] + 1
    elif block_type == 0:
        block_type = ringbuffer[rb]
    else:
        block_type -= 2
    if block_type >= max_block_type:
        block_type -= max_block_type
    ringbuffer[rb] = ringbuffer[rb + 1]
    ringbuffer[rb + 1] = block_type


def detect_trivial_literal_block_types(s):
    """DetectTrivialLiteralBlockTypes: marks block types whose 64 literal
    contexts all map to the same Huffman tree."""
    trivial = [0] * 8
    cm = s.context_map
    for i in range(s.num_block_types[0]):
        offset = i << LITERAL_CONTEXT_BITS
        sample = cm[offset]
        if cm.count(sample, offset, offset + (1 << LITERAL_CONTEXT_BITS)) == 64:
            trivial[i >> 5] |= 1 << (i & 31)
    s.trivial_literal_contexts = trivial


def prepare_literal_decoding(s):
    """PrepareLiteralDecoding."""
    block_type = s.block_type_rb[1]
    context_offset = block_type << LITERAL_CONTEXT_BITS
    s.context_map_slice = context_offset
    trivial = s.trivial_literal_contexts[block_type >> 5]
    s.trivial_literal_context = (trivial >> (block_type & 31)) & 1
    s.literal_htree = s.literal_hgroup.htrees[s.context_map[context_offset]]
    s.context_lookup = s.context_modes[block_type] & 3


def decode_literal_block_switch(s):
    """DecodeLiteralBlockSwitch."""
    decode_block_type_and_length(s, 0)
    prepare_literal_decoding(s)


def decode_command_block_switch(s):
    """DecodeCommandBlockSwitch."""
    decode_block_type_and_length(s, 1)
    s.htree_command = s.insert_copy_hgroup.htrees[s.block_type_rb[3]]


def decode_distance_block_switch(s):
    """DecodeDistanceBlockSwitch."""
    decode_block_type_and_length(s, 2)
    s.dist_context_map_slice = s.block_type_rb[5] << DISTANCE_CONTEXT_BITS
    s.dist_htree_index = s.dist_context_map[
        s.dist_context_map_slice + s.distance_context]


# ---------------------------------------------------------------------------
# Uncompressed and metadata meta-blocks
# ---------------------------------------------------------------------------

def skip_metadata_block(s):
    """SkipMetadataBlock (the binding installs no metadata callbacks)."""
    br = s.br
    n = s.meta_block_remaining_len
    if n == 0:
        return
    pos = br.byte_position()
    if pos + n > br.last_in:
        _overrun()
    br.set_byte_position(pos + n)
    s.meta_block_remaining_len = 0


def copy_uncompressed_block_to_output(s):
    """CopyUncompressedBlockToOutput."""
    br = s.br
    n = s.meta_block_remaining_len
    pos = br.byte_position()
    if pos + n > br.last_in:
        _overrun()
    s.output += br.buf[pos:pos + n]
    br.set_byte_position(pos + n)
    s.meta_block_remaining_len = 0


# ---------------------------------------------------------------------------
# Compressed meta-block header
# ---------------------------------------------------------------------------

def read_compressed_metablock_header(s):
    """BROTLI_STATE_HUFFMAN_CODE_0 .. BROTLI_STATE_TREE_GROUP of
    BrotliDecoderDecompressStream."""
    br = s.br
    for i in range(3):
        # BROTLI_STATE_HUFFMAN_CODE_0
        nbt = decode_var_len_uint8(br) + 1
        s.num_block_types[i] = nbt
        if nbt < 2:
            continue
        # BROTLI_STATE_HUFFMAN_CODE_1
        alphabet_size = nbt + 2
        s.block_type_trees[i] = read_huffman_code(
            alphabet_size, alphabet_size, HUFFMAN_MAX_SIZE_258, br)[0]
        # BROTLI_STATE_HUFFMAN_CODE_2
        s.block_len_trees[i] = read_huffman_code(
            NUM_BLOCK_LEN_SYMBOLS, NUM_BLOCK_LEN_SYMBOLS,
            HUFFMAN_MAX_SIZE_26, br)[0]
        # BROTLI_STATE_HUFFMAN_CODE_3
        s.block_length[i] = read_block_length(s.block_len_trees[i], br)

    # BROTLI_STATE_METABLOCK_HEADER_2
    bits = br.read_bits(6)
    s.distance_postfix_bits = bits & BIT_MASK[2]
    bits >>= 2
    s.num_direct_distance_codes = bits << s.distance_postfix_bits

    # BROTLI_STATE_CONTEXT_MODES (ReadContextModes)
    s.context_modes = bytes(br.read_bits(2)
                            for _ in range(s.num_block_types[0]))

    # BROTLI_STATE_CONTEXT_MAP_1
    s.num_literal_htrees, s.context_map = decode_context_map(
        s.num_block_types[0] << LITERAL_CONTEXT_BITS, s)
    detect_trivial_literal_block_types(s)

    # BROTLI_STATE_CONTEXT_MAP_2
    npostfix = s.distance_postfix_bits
    ndirect = s.num_direct_distance_codes
    distance_alphabet_size_max = _distance_alphabet_size(
        npostfix, ndirect, MAX_DISTANCE_BITS)
    distance_alphabet_size_limit = distance_alphabet_size_max
    s.num_dist_htrees, s.dist_context_map = decode_context_map(
        s.num_block_types[2] << DISTANCE_CONTEXT_BITS, s)
    s.literal_hgroup = HuffmanTreeGroup(
        NUM_LITERAL_SYMBOLS, NUM_LITERAL_SYMBOLS, s.num_literal_htrees)
    s.insert_copy_hgroup = HuffmanTreeGroup(
        NUM_COMMAND_SYMBOLS, NUM_COMMAND_SYMBOLS, s.num_block_types[1])
    s.distance_hgroup = HuffmanTreeGroup(
        distance_alphabet_size_max, distance_alphabet_size_limit,
        s.num_dist_htrees)

    # BROTLI_STATE_TREE_GROUP
    huffman_tree_group_decode(s.literal_hgroup, br)
    huffman_tree_group_decode(s.insert_copy_hgroup, br)
    huffman_tree_group_decode(s.distance_hgroup, br)

    # BROTLI_STATE_BEFORE_COMPRESSED_METABLOCK_BODY
    prepare_literal_decoding(s)
    s.dist_context_map_slice = 0
    s.htree_command = s.insert_copy_hgroup.htrees[0]
    calculate_distance_lut(s)


def calculate_distance_lut(s):
    """CalculateDistanceLut: per distance symbol, the number of extra bits
    and the base distance (RFC 7932 section 4)."""
    npostfix = s.distance_postfix_bits
    ndirect = s.num_direct_distance_codes
    alphabet_size_limit = s.distance_hgroup.alphabet_size_limit
    postfix = 1 << npostfix
    dist_extra_bits = [0] * alphabet_size_limit
    dist_offset = [0] * alphabet_size_limit
    bits = 1
    half = 0
    # Skip short codes.
    i = NUM_DISTANCE_SHORT_CODES
    # Fill direct codes.
    for j in range(ndirect):
        dist_extra_bits[i] = 0
        dist_offset[i] = j + 1
        i += 1
    # Fill regular distance codes.
    while i < alphabet_size_limit:
        base = ndirect + ((((2 + half) << bits) - 4) << npostfix) + 1
        # Always fill the complete group.
        for j in range(postfix):
            dist_extra_bits[i] = bits
            dist_offset[i] = base + j
            i += 1
        bits = bits + half
        half = half ^ 1
    s.dist_extra_bits = dist_extra_bits
    s.dist_offset = dist_offset


# ---------------------------------------------------------------------------
# Compressed meta-block body
# ---------------------------------------------------------------------------

def process_commands(s):
    """ProcessCommandsInternal: decodes the commands of one compressed
    meta-block (insert literals, then copy a back-reference or a static
    dictionary word) until the meta-block length is exhausted.

    This is the hot loop: the bit reader state is kept in local variables and
    the C helpers (ReadSymbol, BrotliReadBits24, ReadCommand, ReadDistance,
    TakeDistanceFromRingBuffer, ...) are inlined."""
    br = s.br
    buf = br.buf
    val = br.val
    bit_pos = br.bit_pos
    next_in = br.next_in
    last_bits = br.last_in << 3
    from_bytes = int.from_bytes
    bit_mask = BIT_MASK

    out = s.output
    append = out.append
    remaining = s.meta_block_remaining_len
    max_backward_distance = s.max_backward_distance
    dist_rb = s.dist_rb
    dist_rb_idx = s.dist_rb_idx
    npostfix = s.distance_postfix_bits
    dist_extra_bits = s.dist_extra_bits
    dist_offset = s.dist_offset
    literal_htrees = s.literal_hgroup.htrees
    dist_htrees = s.distance_hgroup.htrees
    dist_context_map = s.dist_context_map
    context_map = s.context_map
    block_length = s.block_length
    bl0, bl1, bl2 = block_length
    cmd_lut = CMD_LUT
    dict_data = _DICT_DATA
    transforms = _TRANSFORMS

    htree_command = s.htree_command
    dist_cmap_slice = s.dist_context_map_slice

    # Literal decoding state (PrepareLiteralDecoding).
    trivial = s.trivial_literal_context
    literal_htree = s.literal_htree
    cm_slice = s.context_map_slice
    ctx_trees = [literal_htrees[context_map[cm_slice + c]] for c in range(64)]
    lut1 = _CONTEXT_LUT_P1[s.context_lookup]
    lut2 = _CONTEXT_LUT_P2[s.context_lookup]

    while True:
        # CommandBegin
        cmd_start_bits = (next_in << 3) - bit_pos
        if bl1 == 0:
            br.val, br.bit_pos, br.next_in = val, bit_pos, next_in
            decode_command_block_switch(s)
            val, bit_pos, next_in = br.val, br.bit_pos, br.next_in
            bl1 = block_length[1]
            htree_command = s.htree_command

        # ReadCommand
        if bit_pos < 15:
            if (next_in << 3) - bit_pos > last_bits:
                _overrun()
            val |= from_bytes(buf[next_in:next_in + 4], "little") << bit_pos
            next_in += 4
            bit_pos += 32
        e = htree_command[val & 0xFF]
        nb = e & 15
        if nb > 8:
            e = htree_command[(val & 0xFF) + (e >> 4) +
                              ((val >> 8) & bit_mask[nb - 8])]
            nb = (e & 15) + 8
        val >>= nb
        bit_pos -= nb
        (insert_len_extra_bits, copy_len_extra_bits, distance_code,
         distance_context, insert_length, copy_length) = cmd_lut[e >> 4]
        dist_htree_index = dist_context_map[dist_cmap_slice + distance_context]
        if insert_len_extra_bits:
            if bit_pos < insert_len_extra_bits:
                if (next_in << 3) - bit_pos > last_bits:
                    _overrun()
                val |= from_bytes(buf[next_in:next_in + 4], "little") << bit_pos
                next_in += 4
                bit_pos += 32
            insert_length += val & bit_mask[insert_len_extra_bits]
            val >>= insert_len_extra_bits
            bit_pos -= insert_len_extra_bits
        if copy_len_extra_bits:
            if bit_pos < copy_len_extra_bits:
                if (next_in << 3) - bit_pos > last_bits:
                    _overrun()
                val |= from_bytes(buf[next_in:next_in + 4], "little") << bit_pos
                next_in += 4
                bit_pos += 32
            copy_length += val & bit_mask[copy_len_extra_bits]
            val >>= copy_len_extra_bits
            bit_pos -= copy_len_extra_bits
        bl1 -= 1

        if insert_length:
            remaining -= insert_length
            if remaining < 0:
                # C decodes the literals first, then fails in
                # WriteRingBuffer / METABLOCK_DONE (BLOCK_LENGTH_1/2).
                _fail("FORMAT_BLOCK_LENGTH_1")
            # CommandInner: read the literals in the command.
            while insert_length:
                if bl0 == 0:
                    # NextLiteralBlock
                    br.val, br.bit_pos, br.next_in = val, bit_pos, next_in
                    decode_literal_block_switch(s)
                    val, bit_pos, next_in = br.val, br.bit_pos, br.next_in
                    bl0 = block_length[0]
                    trivial = s.trivial_literal_context
                    literal_htree = s.literal_htree
                    cm_slice = s.context_map_slice
                    ctx_trees = [literal_htrees[context_map[cm_slice + c]]
                                 for c in range(64)]
                    lut1 = _CONTEXT_LUT_P1[s.context_lookup]
                    lut2 = _CONTEXT_LUT_P2[s.context_lookup]
                    continue
                n = insert_length if insert_length < bl0 else bl0
                insert_length -= n
                bl0 -= n
                if trivial:
                    tbl = literal_htree
                    for _ in range(n):
                        if bit_pos < 15:
                            if (next_in << 3) - bit_pos > last_bits:
                                _overrun()
                            val |= from_bytes(buf[next_in:next_in + 4],
                                              "little") << bit_pos
                            next_in += 4
                            bit_pos += 32
                        e = tbl[val & 0xFF]
                        nb = e & 15
                        if nb > 8:
                            e = tbl[(val & 0xFF) + (e >> 4) +
                                    ((val >> 8) & bit_mask[nb - 8])]
                            nb = (e & 15) + 8
                        val >>= nb
                        bit_pos -= nb
                        append(e >> 4)
                else:
                    olen = len(out)
                    p1 = out[-1] if olen else 0
                    p2 = out[-2] if olen > 1 else 0
                    for _ in range(n):
                        if bit_pos < 15:
                            if (next_in << 3) - bit_pos > last_bits:
                                _overrun()
                            val |= from_bytes(buf[next_in:next_in + 4],
                                              "little") << bit_pos
                            next_in += 4
                            bit_pos += 32
                        tbl = ctx_trees[lut1[p1] | lut2[p2]]
                        e = tbl[val & 0xFF]
                        nb = e & 15
                        if nb > 8:
                            e = tbl[(val & 0xFF) + (e >> 4) +
                                    ((val >> 8) & bit_mask[nb - 8])]
                            nb = (e & 15) + 8
                        val >>= nb
                        bit_pos -= nb
                        p2 = p1
                        p1 = e >> 4
                        append(p1)
            if remaining <= 0:
                break  # BROTLI_STATE_METABLOCK_DONE

        # CommandPostDecodeLiterals
        if distance_code >= 0:
            # Implicit distance case.
            explicit_distance = False
            distance_context = 0 if distance_code else 1
            dist_rb_idx -= 1
            distance_code = dist_rb[dist_rb_idx & 3]
        else:
            # Read distance code in the command, unless it was implicitly zero.
            explicit_distance = True
            if bl2 == 0:
                br.val, br.bit_pos, br.next_in = val, bit_pos, next_in
                s.distance_context = distance_context
                decode_distance_block_switch(s)
                val, bit_pos, next_in = br.val, br.bit_pos, br.next_in
                bl2 = block_length[2]
                dist_cmap_slice = s.dist_context_map_slice
                dist_htree_index = s.dist_htree_index
            # ReadDistance
            tbl = dist_htrees[dist_htree_index]
            if bit_pos < 15:
                if (next_in << 3) - bit_pos > last_bits:
                    _overrun()
                val |= from_bytes(buf[next_in:next_in + 4], "little") << bit_pos
                next_in += 4
                bit_pos += 32
            e = tbl[val & 0xFF]
            nb = e & 15
            if nb > 8:
                e = tbl[(val & 0xFF) + (e >> 4) + ((val >> 8) & bit_mask[nb - 8])]
                nb = (e & 15) + 8
            val >>= nb
            bit_pos -= nb
            code = e >> 4
            bl2 -= 1
            distance_context = 0
            if code < NUM_DISTANCE_SHORT_CODES:
                # TakeDistanceFromRingBuffer
                if code <= 3:
                    # Compensate double distance-ring-buffer roll for
                    # dictionary items.
                    distance_context = 1 >> code
                    distance_code = dist_rb[(dist_rb_idx - (code - 3)) & 3]
                    dist_rb_idx -= distance_context
                else:
                    if code < 10:
                        base = code - 4
                        index_delta = 3
                    else:
                        base = code - 10
                        index_delta = 2
                    # Unpack one of six 4-bit values.
                    delta = ((0x605142 >> (4 * base)) & 0xF) - 3
                    distance_code = dist_rb[(dist_rb_idx + index_delta) & 3] + delta
                    if distance_code <= 0:
                        # A huge distance will cause a BROTLI_FAILURE() soon.
                        distance_code = 0x7FFFFFFF
            else:
                nb = dist_extra_bits[code]
                if bit_pos < nb:
                    if (next_in << 3) - bit_pos > last_bits:
                        _overrun()
                    val |= from_bytes(buf[next_in:next_in + 4], "little") << bit_pos
                    next_in += 4
                    bit_pos += 32
                distance_code = dist_offset[code] + (
                    (val & bit_mask[nb]) << npostfix)
                val >>= nb
                bit_pos -= nb

        olen = len(out)
        max_distance = olen if olen < max_backward_distance else max_backward_distance
        i = copy_length
        # Apply copy of LZ77 back-reference, or static dictionary reference if
        # the distance is larger than the max LZ77 distance.
        if distance_code > max_distance:
            if distance_code > MAX_ALLOWED_DISTANCE:
                _fail("FORMAT_DISTANCE")
            if (SHARED_BROTLI_MIN_DICTIONARY_WORD_LENGTH <= i <=
                    SHARED_BROTLI_MAX_DICTIONARY_WORD_LENGTH):
                offset = OFFSETS_BY_LENGTH[i]
                shift = SIZE_BITS_BY_LENGTH[i]
                address = distance_code - max_distance - 1
                word_idx = address & bit_mask[shift]
                transform_idx = address >> shift
                # Compensate double distance-ring-buffer roll.
                dist_rb_idx += distance_context
                offset += word_idx * i
                if shift == 0:
                    _fail("FORMAT_DICTIONARY")
                if transform_idx < _NUM_TRANSFORMS:
                    if transform_idx == _IDENTITY_TRANSFORM:
                        out += dict_data[offset:offset + i]
                        length = i
                    else:
                        length = transform_dictionary_word(
                            out, dict_data[offset:offset + i], i, transforms,
                            transform_idx)
                        if length == 0 and distance_code <= 120:
                            _fail("FORMAT_TRANSFORM")
                    remaining -= length
                    if remaining < 0:
                        _fail("FORMAT_BLOCK_LENGTH_1")
                    if (length == 0 and insert_length == 0 and
                            (next_in << 3) - bit_pos == cmd_start_bits):
                        # No input consumed, no output produced, and the
                        # state is unchanged except for the block counters:
                        # the following commands up to the next block switch
                        # are identical no-ops.  Skip them all at once.
                        k = bl1
                        if explicit_distance:
                            if bl2 < k:
                                k = bl2
                            bl2 -= k
                        bl1 -= k
                else:
                    _fail("FORMAT_TRANSFORM")
            else:
                _fail("FORMAT_DICTIONARY")
        else:
            # Update the recent distances cache.
            dist_rb[dist_rb_idx & 3] = distance_code
            dist_rb_idx += 1
            remaining -= i
            if remaining < 0:
                _fail("FORMAT_BLOCK_LENGTH_1")
            src = olen - distance_code
            if i <= distance_code:
                out += out[src:src + i]
            else:
                # Overlapping copy: the output repeats with period distance.
                pattern = out[src:]
                reps, rem = divmod(i, distance_code)
                out += pattern * reps
                if rem:
                    out += pattern[:rem]
        if remaining <= 0:
            break  # BROTLI_STATE_METABLOCK_DONE

    br.val, br.bit_pos, br.next_in = val, bit_pos, next_in
    block_length[0] = bl0
    block_length[1] = bl1
    block_length[2] = bl2
    s.dist_rb_idx = dist_rb_idx
    s.meta_block_remaining_len = remaining


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _decompress_stream(s):
    """BrotliDecoderDecompressStream, run to completion on the whole input.

    Returns the decoded bytes or raises BrotliError."""
    br = s.br
    # BROTLI_STATE_UNINITED: prepare to the first read.
    br.warmup()
    # Decode window size.
    decode_window_bits(s, br)
    # BROTLI_STATE_INITIALIZE: maximum distance, see section 9.1. of the spec.
    s.max_backward_distance = (1 << s.window_bits) - WINDOW_GAP
    while True:
        # BROTLI_STATE_METABLOCK_BEGIN
        s.metablock_begin()
        # BROTLI_STATE_METABLOCK_HEADER
        decode_meta_block_length(s, br)
        if s.is_metadata or s.is_uncompressed:
            if not br.jump_to_byte_boundary():
                _fail("FORMAT_PADDING_1")
        if s.is_metadata:
            # BROTLI_STATE_METADATA
            skip_metadata_block(s)
        elif s.meta_block_remaining_len == 0:
            pass
        elif s.is_uncompressed:
            # BROTLI_STATE_UNCOMPRESSED
            copy_uncompressed_block_to_output(s)
        else:
            read_compressed_metablock_header(s)
            # BROTLI_STATE_COMMAND_*
            process_commands(s)
        # BROTLI_STATE_METABLOCK_DONE
        br.check_overrun()
        if s.meta_block_remaining_len < 0:
            _fail("FORMAT_BLOCK_LENGTH_2")
        s.cleanup_after_metablock()
        if not s.is_last_metablock:
            continue
        if not br.jump_to_byte_boundary():
            _fail("FORMAT_PADDING_2")
        # BROTLI_STATE_DONE.  The Python binding additionally requires that
        # the whole input was consumed (available_in == 0).
        if br.byte_position() != br.last_in:
            raise BrotliError("brotli: decoder failed (unused input after "
                              "the end of the stream)")
        return bytes(s.output)


def decompress(data):
    """Decompresses a complete Brotli stream (``brotli.decompress``).

    ``data`` may be any C-contiguous bytes-like object.  Raises BrotliError
    if the stream is invalid, truncated or followed by extra data.
    """
    view = memoryview(data)  # TypeError for non-buffer objects, like _brotli
    if not view.c_contiguous:
        raise TypeError("object supporting the buffer API required")
    s = BrotliDecoderState(view.tobytes())
    if _STRICT:
        return _decompress_stream(s)
    try:
        return _decompress_stream(s)
    except (IndexError, KeyError, ValueError, OverflowError,
            TypeError, AttributeError) as exc:  # pragma: no cover
        # Safety net: never leak an internal exception type for bad input.
        raise BrotliError("brotli: decoder failed (internal: %r)" % (exc,))
