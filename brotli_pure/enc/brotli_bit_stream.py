"""Brotli bit stream functions to support the low level format
(port of c/enc/brotli_bit_stream.h / .c and block_encoder_inc.h).

There are no compression algorithms here, just the right ordering of bits to
match the specs.  All functions take a ``BitWriter`` (write_bits.py) in place
of the C ``(size_t* storage_ix, uint8_t* storage)`` pair.

Not ported (only used by qualities 0-3): BrotliBuildAndStoreHuffmanTreeFast,
BrotliStoreMetaBlockFast, BrotliStoreMetaBlockTrivial.
"""

from ..common.context import CONTEXT_LOOKUP
from .command import (get_insert_length_code, get_copy_length_code,
                      kBrotliInsBase, kBrotliInsExtra, kBrotliCopyBase,
                      kBrotliCopyExtra, command_copy_len_code,
                      command_distance_context)
from .entropy_encode import (brotli_create_huffman_tree,
                             brotli_convert_bit_depths_to_symbols,
                             brotli_write_huffman_tree,
                             BROTLI_CODE_LENGTH_CODES,
                             BROTLI_REPEAT_PREVIOUS_CODE_LENGTH,
                             BROTLI_REPEAT_ZERO_CODE_LENGTH)
from .histogram import (BROTLI_NUM_LITERAL_SYMBOLS, BROTLI_NUM_COMMAND_SYMBOLS,
                        BROTLI_LITERAL_CONTEXT_BITS,
                        BROTLI_DISTANCE_CONTEXT_BITS)

BROTLI_NUM_BLOCK_LEN_SYMBOLS = 26
BROTLI_MAX_NUMBER_OF_BLOCK_TYPES = 256
BROTLI_MAX_BLOCK_TYPE_SYMBOLS = BROTLI_MAX_NUMBER_OF_BLOCK_TYPES + 2
BROTLI_MAX_CONTEXT_MAP_SYMBOLS = BROTLI_MAX_NUMBER_OF_BLOCK_TYPES + 16

MAX_HUFFMAN_TREE_SIZE = 2 * BROTLI_NUM_COMMAND_SYMBOLS + 1

# _kBrotliPrefixCodeRanges (c/common/constants.c): (offset, nbits)
kBrotliPrefixCodeRanges = (
    (1, 2), (5, 2), (9, 2), (13, 2), (17, 3), (25, 3),
    (33, 3), (41, 3), (49, 4), (65, 4), (81, 4), (97, 4),
    (113, 5), (145, 5), (177, 5), (209, 5), (241, 6), (305, 6),
    (369, 7), (497, 8), (753, 9), (1265, 10), (2289, 11), (4337, 12),
    (8433, 13), (16625, 24))


def block_length_prefix_code(len_):
    """BlockLengthPrefixCode."""
    code = (20 if len_ >= 753 else 14) if len_ >= 177 else (7 if len_ >= 41 else 0)
    while (code < (BROTLI_NUM_BLOCK_LEN_SYMBOLS - 1) and
           len_ >= kBrotliPrefixCodeRanges[code + 1][0]):
        code += 1
    return code


def get_block_length_prefix_code(len_):
    """GetBlockLengthPrefixCode; returns (code, n_extra, extra)."""
    code = block_length_prefix_code(len_)
    offset, nbits = kBrotliPrefixCodeRanges[code]
    return code, nbits, len_ - offset


class BlockTypeCodeCalculator(object):
    __slots__ = ('last_type', 'second_last_type')

    def __init__(self):
        # InitBlockTypeCodeCalculator
        self.last_type = 1
        self.second_last_type = 0

    def next_block_type_code(self, type_):
        """NextBlockTypeCode."""
        if type_ == self.last_type + 1:
            type_code = 1
        elif type_ == self.second_last_type:
            type_code = 0
        else:
            type_code = type_ + 2
        self.second_last_type = self.last_type
        self.last_type = type_
        return type_code


def brotli_encode_mlen(length):
    """BrotliEncodeMlen; returns (bits, numbits, nibblesbits).
    REQUIRES: 0 < length <= (1 << 24)."""
    lg = 1 if length == 1 else (length - 1).bit_length()
    mnibbles = (16 if lg < 16 else (lg + 3)) // 4
    return length - 1, mnibbles * 4, mnibbles - 4


def store_command_extra(cmd, w):
    """StoreCommandExtra."""
    copylen_code = command_copy_len_code(cmd)
    inscode = get_insert_length_code(cmd.insert_len)
    copycode = get_copy_length_code(copylen_code)
    insnumextra = kBrotliInsExtra[inscode]
    insextraval = cmd.insert_len - kBrotliInsBase[inscode]
    copyextraval = copylen_code - kBrotliCopyBase[copycode]
    bits = (copyextraval << insnumextra) | insextraval
    w.write_bits(insnumextra + kBrotliCopyExtra[copycode], bits)


class BlockSplitCode(object):
    """Data structure that stores almost everything that is needed to encode
    each block switch command."""

    __slots__ = ('type_code_calculator', 'type_depths', 'type_bits',
                 'length_depths', 'length_bits')

    def __init__(self):
        self.type_code_calculator = BlockTypeCodeCalculator()
        self.type_depths = [0] * BROTLI_MAX_BLOCK_TYPE_SYMBOLS
        self.type_bits = [0] * BROTLI_MAX_BLOCK_TYPE_SYMBOLS
        self.length_depths = [0] * BROTLI_NUM_BLOCK_LEN_SYMBOLS
        self.length_bits = [0] * BROTLI_NUM_BLOCK_LEN_SYMBOLS


def store_var_len_uint8(n, w):
    """StoreVarLenUint8: stores a number between 0 and 255."""
    if n == 0:
        w.write_bits(1, 0)
    else:
        nbits = n.bit_length() - 1
        w.write_bits(1, 1)
        w.write_bits(3, nbits)
        w.write_bits(nbits, n - (1 << nbits))


def store_compressed_meta_block_header(is_final_block, length, w):
    """StoreCompressedMetaBlockHeader.
    REQUIRES: 0 < length <= (1 << 24)."""
    # Write ISLAST bit.
    w.write_bits(1, 1 if is_final_block else 0)
    # Write ISEMPTY bit.
    if is_final_block:
        w.write_bits(1, 0)
    lenbits, nlenbits, nibblesbits = brotli_encode_mlen(length)
    w.write_bits(2, nibblesbits)
    w.write_bits(nlenbits, lenbits)
    if not is_final_block:
        # Write ISUNCOMPRESSED bit.
        w.write_bits(1, 0)


def brotli_store_uncompressed_meta_block_header(length, w):
    """BrotliStoreUncompressedMetaBlockHeader.
    REQUIRES: 0 < length <= (1 << 24)."""
    # Write ISLAST bit.
    # Uncompressed block cannot be the last one, so set to 0.
    w.write_bits(1, 0)
    lenbits, nlenbits, nibblesbits = brotli_encode_mlen(length)
    w.write_bits(2, nibblesbits)
    w.write_bits(nlenbits, lenbits)
    # Write ISUNCOMPRESSED bit.
    w.write_bits(1, 1)


_kStorageOrder = (1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# The bit lengths of the Huffman code over the code length alphabet
# are compressed with the following static Huffman code:
#   Symbol   Code
#   ------   ----
#   0          00
#   1        1110
#   2         110
#   3          01
#   4          10
#   5        1111
_kHuffmanBitLengthHuffmanCodeSymbols = (0, 7, 3, 2, 1, 15)
_kHuffmanBitLengthHuffmanCodeBitLengths = (2, 4, 3, 2, 2, 4)


def brotli_store_huffman_tree_of_huffman_tree_to_bit_mask(
        num_codes, code_length_bitdepth, w):
    """BrotliStoreHuffmanTreeOfHuffmanTreeToBitMask."""
    skip_some = 0  # skips none.
    # Throw away trailing zeros:
    codes_to_store = BROTLI_CODE_LENGTH_CODES
    if num_codes > 1:
        while codes_to_store > 0:
            if code_length_bitdepth[_kStorageOrder[codes_to_store - 1]] != 0:
                break
            codes_to_store -= 1
    if (code_length_bitdepth[_kStorageOrder[0]] == 0 and
            code_length_bitdepth[_kStorageOrder[1]] == 0):
        skip_some = 2  # skips two.
        if code_length_bitdepth[_kStorageOrder[2]] == 0:
            skip_some = 3  # skips three.
    w.write_bits(2, skip_some)
    for i in range(skip_some, codes_to_store):
        l_ = code_length_bitdepth[_kStorageOrder[i]]
        w.write_bits(_kHuffmanBitLengthHuffmanCodeBitLengths[l_],
                     _kHuffmanBitLengthHuffmanCodeSymbols[l_])


def brotli_store_huffman_tree_to_bit_mask(huffman_tree_size, huffman_tree,
                                          huffman_tree_extra_bits,
                                          code_length_bitdepth,
                                          code_length_bitdepth_symbols, w):
    """BrotliStoreHuffmanTreeToBitMask."""
    for i in range(huffman_tree_size):
        ix = huffman_tree[i]
        w.write_bits(code_length_bitdepth[ix], code_length_bitdepth_symbols[ix])
        # Extra bits
        if ix == BROTLI_REPEAT_PREVIOUS_CODE_LENGTH:
            w.write_bits(2, huffman_tree_extra_bits[i])
        elif ix == BROTLI_REPEAT_ZERO_CODE_LENGTH:
            w.write_bits(3, huffman_tree_extra_bits[i])


def store_simple_huffman_tree(depths, symbols, num_symbols, max_bits, w):
    """StoreSimpleHuffmanTree (``symbols`` is sorted in place)."""
    # value of 1 indicates a simple Huffman code
    w.write_bits(2, 1)
    w.write_bits(2, num_symbols - 1)  # NSYM - 1
    # Sort
    for i in range(num_symbols):
        for j in range(i + 1, num_symbols):
            if depths[symbols[j]] < depths[symbols[i]]:
                symbols[j], symbols[i] = symbols[i], symbols[j]
    if num_symbols == 2:
        w.write_bits(max_bits, symbols[0])
        w.write_bits(max_bits, symbols[1])
    elif num_symbols == 3:
        w.write_bits(max_bits, symbols[0])
        w.write_bits(max_bits, symbols[1])
        w.write_bits(max_bits, symbols[2])
    else:
        w.write_bits(max_bits, symbols[0])
        w.write_bits(max_bits, symbols[1])
        w.write_bits(max_bits, symbols[2])
        w.write_bits(max_bits, symbols[3])
        # tree-select
        w.write_bits(1, 1 if depths[symbols[0]] == 1 else 0)


def brotli_store_huffman_tree(depths, num, tree, w):
    """BrotliStoreHuffmanTree: num = alphabet size, depths = symbol depths."""
    # Write the Huffman tree into the brotli-representation.
    huffman_tree = []
    huffman_tree_extra_bits = []
    code_length_bitdepth = [0] * BROTLI_CODE_LENGTH_CODES
    code_length_bitdepth_symbols = [0] * BROTLI_CODE_LENGTH_CODES
    huffman_tree_histogram = [0] * BROTLI_CODE_LENGTH_CODES
    num_codes = 0
    code = 0

    huffman_tree_size = brotli_write_huffman_tree(
        depths, num, huffman_tree, huffman_tree_extra_bits)

    # Calculate the statistics of the Huffman tree in brotli-representation.
    for i in range(huffman_tree_size):
        huffman_tree_histogram[huffman_tree[i]] += 1

    for i in range(BROTLI_CODE_LENGTH_CODES):
        if huffman_tree_histogram[i]:
            if num_codes == 0:
                code = i
                num_codes = 1
            elif num_codes == 1:
                num_codes = 2
                break

    # Calculate another Huffman tree to use for compressing both the
    # earlier Huffman tree with.
    brotli_create_huffman_tree(huffman_tree_histogram, BROTLI_CODE_LENGTH_CODES,
                               5, tree, code_length_bitdepth)
    brotli_convert_bit_depths_to_symbols(code_length_bitdepth,
                                         BROTLI_CODE_LENGTH_CODES,
                                         code_length_bitdepth_symbols)

    # Now, we have all the data, let's start storing it
    brotli_store_huffman_tree_of_huffman_tree_to_bit_mask(
        num_codes, code_length_bitdepth, w)

    if num_codes == 1:
        code_length_bitdepth[code] = 0

    # Store the real Huffman tree now.
    brotli_store_huffman_tree_to_bit_mask(
        huffman_tree_size, huffman_tree, huffman_tree_extra_bits,
        code_length_bitdepth, code_length_bitdepth_symbols, w)


def build_and_store_huffman_tree(histogram, histogram_length, alphabet_size,
                                 tree, depth, bits, w, offset=0):
    """BuildAndStoreHuffmanTree: builds a Huffman tree from
    histogram[0:length] into depth[offset:offset+length] and
    bits[offset:offset+length] and stores the encoded tree."""
    count = 0
    s4 = [0, 0, 0, 0]
    for i in range(histogram_length):
        if histogram[i]:
            if count < 4:
                s4[count] = i
            elif count > 4:
                break
            count += 1

    max_bits = 0
    max_bits_counter = alphabet_size - 1
    while max_bits_counter:
        max_bits_counter >>= 1
        max_bits += 1

    if count <= 1:
        w.write_bits(4, 1)
        w.write_bits(max_bits, s4[0])
        depth[offset + s4[0]] = 0
        bits[offset + s4[0]] = 0
        return

    d = [0] * histogram_length
    b = [0] * histogram_length
    brotli_create_huffman_tree(histogram, histogram_length, 15, tree, d)
    brotli_convert_bit_depths_to_symbols(d, histogram_length, b)
    depth[offset:offset + histogram_length] = d
    bits[offset:offset + histogram_length] = b

    if count <= 4:
        store_simple_huffman_tree(d, s4, count, max_bits, w)
    else:
        brotli_store_huffman_tree(d, histogram_length, tree, w)


def _move_to_front_transform(v_in, v_size):
    """MoveToFrontTransform; returns v_out."""
    if v_size == 0:
        return []
    max_value = max(v_in[:v_size])
    mtf = list(range(max_value + 1))
    v_out = [0] * v_size
    for i in range(v_size):
        value = v_in[i]
        index = mtf.index(value)  # IndexOf
        v_out[i] = index
        # MoveToFront
        if index:
            del mtf[index]
            mtf.insert(0, value)
    return v_out


def _run_length_code_zeros(in_size, v, max_run_length_prefix):
    """RunLengthCodeZeros: rewrites v in place; returns
    (out_size, max_run_length_prefix)."""
    max_reps = 0
    i = 0
    while i < in_size:
        reps = 0
        while i < in_size and v[i] != 0:
            i += 1
        while i < in_size and v[i] == 0:
            reps += 1
            i += 1
        if reps > max_reps:
            max_reps = reps
    max_prefix = (max_reps.bit_length() - 1) if max_reps > 0 else 0
    max_prefix = min(max_prefix, max_run_length_prefix)
    max_run_length_prefix = max_prefix
    out_size = 0
    i = 0
    while i < in_size:
        if v[i] != 0:
            v[out_size] = v[i] + max_run_length_prefix
            i += 1
            out_size += 1
        else:
            reps = 1
            k = i + 1
            while k < in_size and v[k] == 0:
                reps += 1
                k += 1
            i += reps
            while reps != 0:
                if reps < (2 << max_prefix):
                    run_length_prefix = reps.bit_length() - 1
                    extra_bits = reps - (1 << run_length_prefix)
                    v[out_size] = run_length_prefix + (extra_bits << 9)
                    out_size += 1
                    break
                else:
                    extra_bits = (1 << max_prefix) - 1
                    v[out_size] = max_prefix + (extra_bits << 9)
                    reps -= (2 << max_prefix) - 1
                    out_size += 1
    return out_size, max_run_length_prefix


SYMBOL_BITS = 9
_kSymbolMask = (1 << SYMBOL_BITS) - 1


def encode_context_map(context_map, context_map_size, num_clusters, tree, w):
    """EncodeContextMap."""
    max_run_length_prefix = 6
    store_var_len_uint8(num_clusters - 1, w)
    if num_clusters == 1:
        return
    rle_symbols = _move_to_front_transform(context_map, context_map_size)
    num_rle_symbols, max_run_length_prefix = _run_length_code_zeros(
        context_map_size, rle_symbols, max_run_length_prefix)
    histogram = [0] * BROTLI_MAX_CONTEXT_MAP_SYMBOLS
    for i in range(num_rle_symbols):
        histogram[rle_symbols[i] & _kSymbolMask] += 1
    use_rle = max_run_length_prefix > 0
    w.write_bits(1, 1 if use_rle else 0)
    if use_rle:
        w.write_bits(4, max_run_length_prefix - 1)
    depths = [0] * BROTLI_MAX_CONTEXT_MAP_SYMBOLS
    bits = [0] * BROTLI_MAX_CONTEXT_MAP_SYMBOLS
    build_and_store_huffman_tree(histogram, num_clusters + max_run_length_prefix,
                                 num_clusters + max_run_length_prefix,
                                 tree, depths, bits, w)
    for i in range(num_rle_symbols):
        rle_symbol = rle_symbols[i] & _kSymbolMask
        extra_bits_val = rle_symbols[i] >> SYMBOL_BITS
        w.write_bits(depths[rle_symbol], bits[rle_symbol])
        if 0 < rle_symbol <= max_run_length_prefix:
            w.write_bits(rle_symbol, extra_bits_val)
    w.write_bits(1, 1)  # use move-to-front


def store_block_switch(code, block_len, block_type, is_first_block, w):
    """StoreBlockSwitch: stores the block switch command with index block_ix
    to the bit stream."""
    typecode = code.type_code_calculator.next_block_type_code(block_type)
    if not is_first_block:
        w.write_bits(code.type_depths[typecode], code.type_bits[typecode])
    lencode, len_nextra, len_extra = get_block_length_prefix_code(block_len)
    w.write_bits(code.length_depths[lencode], code.length_bits[lencode])
    w.write_bits(len_nextra, len_extra)


def build_and_store_block_split_code(types, lengths, num_blocks, num_types,
                                     tree, code, w):
    """BuildAndStoreBlockSplitCode: builds a BlockSplitCode data structure
    from the block split given by the vector of block types and block lengths
    and stores it to the bit stream."""
    type_histo = [0] * BROTLI_MAX_BLOCK_TYPE_SYMBOLS
    length_histo = [0] * BROTLI_NUM_BLOCK_LEN_SYMBOLS
    type_code_calculator = BlockTypeCodeCalculator()
    for i in range(num_blocks):
        type_code = type_code_calculator.next_block_type_code(types[i])
        if i != 0:
            type_histo[type_code] += 1
        length_histo[block_length_prefix_code(lengths[i])] += 1
    store_var_len_uint8(num_types - 1, w)
    if num_types > 1:  # TODO(eustas): else? could StoreBlockSwitch occur?
        build_and_store_huffman_tree(type_histo, num_types + 2, num_types + 2,
                                     tree, code.type_depths, code.type_bits, w)
        build_and_store_huffman_tree(length_histo, BROTLI_NUM_BLOCK_LEN_SYMBOLS,
                                     BROTLI_NUM_BLOCK_LEN_SYMBOLS, tree,
                                     code.length_depths, code.length_bits, w)
        store_block_switch(code, lengths[0], types[0], True, w)


def store_trivial_context_map(num_types, context_bits, tree, w):
    """StoreTrivialContextMap: stores a context map where the histogram type
    is always the block type."""
    store_var_len_uint8(num_types - 1, w)
    if num_types > 1:
        repeat_code = context_bits - 1
        repeat_bits = (1 << repeat_code) - 1
        alphabet_size = num_types + repeat_code
        histogram = [0] * alphabet_size
        depths = [0] * BROTLI_MAX_CONTEXT_MAP_SYMBOLS
        bits = [0] * BROTLI_MAX_CONTEXT_MAP_SYMBOLS
        # Write RLEMAX.
        w.write_bits(1, 1)
        w.write_bits(4, repeat_code - 1)
        histogram[repeat_code] = num_types
        histogram[0] = 1
        for i in range(context_bits, alphabet_size):
            histogram[i] = 1
        build_and_store_huffman_tree(histogram, alphabet_size, alphabet_size,
                                     tree, depths, bits, w)
        for i in range(num_types):
            code = 0 if i == 0 else i + context_bits - 1
            w.write_bits(depths[code], bits[code])
            w.write_bits(depths[repeat_code], bits[repeat_code])
            w.write_bits(repeat_code, repeat_bits)
        # Write IMTF (inverse-move-to-front) bit.
        w.write_bits(1, 1)


class BlockEncoder(object):
    """Manages the encoding of one block category (literal, command or
    distance)."""

    __slots__ = ('histogram_length', 'num_block_types', 'block_types',
                 'block_lengths', 'num_blocks', 'block_split_code',
                 'block_ix', 'block_len', 'entropy_ix', 'depths', 'bits')

    def __init__(self, histogram_length, num_block_types, block_types,
                 block_lengths, num_blocks):
        # InitBlockEncoder
        self.histogram_length = histogram_length
        self.num_block_types = num_block_types
        self.block_types = block_types
        self.block_lengths = block_lengths
        self.num_blocks = num_blocks
        self.block_split_code = BlockSplitCode()
        self.block_ix = 0
        self.block_len = 0 if num_blocks == 0 else block_lengths[0]
        self.entropy_ix = 0
        self.depths = None
        self.bits = None

    def build_and_store_block_switch_entropy_codes(self, tree, w):
        """BuildAndStoreBlockSwitchEntropyCodes: creates entropy codes of
        block lengths and block types and stores them to the bit stream."""
        build_and_store_block_split_code(
            self.block_types, self.block_lengths, self.num_blocks,
            self.num_block_types, tree, self.block_split_code, w)

    def build_and_store_entropy_codes(self, histograms, histograms_size,
                                      alphabet_size, tree, w):
        """FN(BuildAndStoreEntropyCodes) (block_encoder_inc.h): creates
        entropy codes for all block types and stores them to the bit
        stream."""
        hl = self.histogram_length
        table_size = histograms_size * hl
        self.depths = [0] * table_size
        self.bits = [0] * table_size
        for i in range(histograms_size):
            ix = i * hl
            build_and_store_huffman_tree(histograms[i].data, hl, alphabet_size,
                                         tree, self.depths, self.bits, w, ix)

    def store_symbol(self, symbol, w):
        """StoreSymbol: stores the next symbol with the entropy code of the
        current block type.  Updates the block type and block length at block
        boundaries."""
        if self.block_len == 0:
            self.block_ix += 1
            block_ix = self.block_ix
            block_len = self.block_lengths[block_ix]
            block_type = self.block_types[block_ix]
            self.block_len = block_len
            self.entropy_ix = block_type * self.histogram_length
            store_block_switch(self.block_split_code, block_len, block_type,
                               False, w)
        self.block_len -= 1
        ix = self.entropy_ix + symbol
        w.write_bits(self.depths[ix], self.bits[ix])

    def store_symbol_with_context(self, symbol, context, context_map, w,
                                  context_bits):
        """StoreSymbolWithContext."""
        if self.block_len == 0:
            self.block_ix += 1
            block_ix = self.block_ix
            block_len = self.block_lengths[block_ix]
            block_type = self.block_types[block_ix]
            self.block_len = block_len
            self.entropy_ix = block_type << context_bits
            store_block_switch(self.block_split_code, block_len, block_type,
                               False, w)
        self.block_len -= 1
        histo_ix = context_map[self.entropy_ix + context]
        ix = histo_ix * self.histogram_length + symbol
        w.write_bits(self.depths[ix], self.bits[ix])


def brotli_store_meta_block(input_, start_pos, length, mask, prev_byte,
                            prev_byte2, is_last, params, literal_context_mode,
                            commands, n_commands, mb, w):
    """BrotliStoreMetaBlock: stores a meta-block with the given MetaBlockSplit
    (block splits, context maps, histograms) into BitWriter ``w``."""
    pos = start_pos
    dist = params.dist
    num_distance_symbols = dist.alphabet_size_max
    num_effective_distance_symbols = dist.alphabet_size_limit
    tree = None
    lut = CONTEXT_LOOKUP
    lut_off = literal_context_mode << 9

    store_compressed_meta_block_header(is_last, length, w)

    literal_enc = BlockEncoder(
        BROTLI_NUM_LITERAL_SYMBOLS, mb.literal_split.num_types,
        mb.literal_split.types, mb.literal_split.lengths,
        mb.literal_split.num_blocks)
    command_enc = BlockEncoder(
        BROTLI_NUM_COMMAND_SYMBOLS, mb.command_split.num_types,
        mb.command_split.types, mb.command_split.lengths,
        mb.command_split.num_blocks)
    distance_enc = BlockEncoder(
        num_effective_distance_symbols, mb.distance_split.num_types,
        mb.distance_split.types, mb.distance_split.lengths,
        mb.distance_split.num_blocks)

    literal_enc.build_and_store_block_switch_entropy_codes(tree, w)
    command_enc.build_and_store_block_switch_entropy_codes(tree, w)
    distance_enc.build_and_store_block_switch_entropy_codes(tree, w)

    w.write_bits(2, dist.distance_postfix_bits)
    w.write_bits(4, dist.num_direct_distance_codes >> dist.distance_postfix_bits)
    for i in range(mb.literal_split.num_types):
        w.write_bits(2, literal_context_mode)

    literal_context_map = mb.literal_context_map
    distance_context_map = mb.distance_context_map
    if mb.literal_context_map_size == 0:
        store_trivial_context_map(mb.literal_histograms_size,
                                  BROTLI_LITERAL_CONTEXT_BITS, tree, w)
    else:
        encode_context_map(literal_context_map, mb.literal_context_map_size,
                           mb.literal_histograms_size, tree, w)

    if mb.distance_context_map_size == 0:
        store_trivial_context_map(mb.distance_histograms_size,
                                  BROTLI_DISTANCE_CONTEXT_BITS, tree, w)
    else:
        encode_context_map(distance_context_map, mb.distance_context_map_size,
                           mb.distance_histograms_size, tree, w)

    literal_enc.build_and_store_entropy_codes(
        mb.literal_histograms, mb.literal_histograms_size,
        BROTLI_NUM_LITERAL_SYMBOLS, tree, w)
    command_enc.build_and_store_entropy_codes(
        mb.command_histograms, mb.command_histograms_size,
        BROTLI_NUM_COMMAND_SYMBOLS, tree, w)
    distance_enc.build_and_store_entropy_codes(
        mb.distance_histograms, mb.distance_histograms_size,
        num_distance_symbols, tree, w)

    write_bits = w.write_bits
    store_command_symbol = command_enc.store_symbol
    store_literal = literal_enc.store_symbol
    store_literal_ctx = literal_enc.store_symbol_with_context
    use_literal_context_map = mb.literal_context_map_size != 0
    use_distance_context_map = mb.distance_context_map_size != 0
    for i in range(n_commands):
        cmd = commands[i]
        cmd_code = cmd.cmd_prefix
        store_command_symbol(cmd_code, w)
        store_command_extra(cmd, w)
        if not use_literal_context_map:
            for _ in range(cmd.insert_len):
                store_literal(input_[pos & mask], w)
                pos += 1
        else:
            for _ in range(cmd.insert_len):
                context = lut[lut_off + prev_byte] | lut[lut_off + 256 + prev_byte2]
                literal = input_[pos & mask]
                store_literal_ctx(literal, context, literal_context_map, w,
                                  BROTLI_LITERAL_CONTEXT_BITS)
                prev_byte2 = prev_byte
                prev_byte = literal
                pos += 1
        copy_len = cmd.copy_len & 0x1FFFFFF
        pos += copy_len
        if copy_len:
            prev_byte2 = input_[(pos - 2) & mask]
            prev_byte = input_[(pos - 1) & mask]
            if cmd_code >= 128:
                dist_code = cmd.dist_prefix & 0x3FF
                distnumextra = cmd.dist_prefix >> 10
                distextra = cmd.dist_extra
                if not use_distance_context_map:
                    distance_enc.store_symbol(dist_code, w)
                else:
                    context = command_distance_context(cmd)
                    distance_enc.store_symbol_with_context(
                        dist_code, context, distance_context_map, w,
                        BROTLI_DISTANCE_CONTEXT_BITS)
                write_bits(distnumextra, distextra)
    if is_last:
        w.jump_to_byte_boundary()


def brotli_store_uncompressed_meta_block(is_final_block, input_, position,
                                         mask, len_, w):
    """BrotliStoreUncompressedMetaBlock: simple raw storage of
    bytes-as-bytes."""
    masked_pos = position & mask
    brotli_store_uncompressed_meta_block_header(len_, w)
    w.jump_to_byte_boundary()
    if masked_pos + len_ > mask + 1:
        len1 = mask + 1 - masked_pos
        w.append_bytes(input_[masked_pos:masked_pos + len1])
        len_ -= len1
        masked_pos = 0
    w.append_bytes(input_[masked_pos:masked_pos + len_])
    # Since the uncompressed block itself may not be the final block, add an
    # empty one after this.
    if is_final_block:
        w.write_bits(1, 1)  # islast
        w.write_bits(1, 1)  # isempty
        w.jump_to_byte_boundary()
