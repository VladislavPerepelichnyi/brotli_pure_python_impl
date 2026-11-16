"""Algorithms for distributing the literals and commands of a metablock
between block types and contexts (port of the greedy / shared parts of
c/enc/metablock.h, metablock.c, metablock_inc.h; BlockSplit from
block_splitter.h).

The quality >= 10 entry point ``BrotliBuildMetaBlock`` (with
ComputeDistanceCost / RecomputeDistancePrefixes) lives in metablock_hq.py.
"""

from ..common.context import CONTEXT_LOOKUP
from .bit_cost import brotli_bits_entropy
from .command import command_copy_len
from .entropy_encode import brotli_optimize_huffman_counts_for_rle
from .histogram import (Histogram, BROTLI_NUM_LITERAL_SYMBOLS,
                        BROTLI_NUM_COMMAND_SYMBOLS,
                        BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS,
                        BROTLI_LITERAL_CONTEXT_BITS)
from .params import (brotli_distance_alphabet_size, BROTLI_MAX_DISTANCE_BITS,
                     BROTLI_LARGE_MAX_DISTANCE_BITS,
                     BROTLI_MAX_ALLOWED_DISTANCE,
                     BROTLI_NUM_DISTANCE_SHORT_CODES)

BROTLI_MAX_STATIC_CONTEXTS = 13
BROTLI_MAX_NUMBER_OF_BLOCK_TYPES = 256


class BlockSplit(object):
    """BlockSplit (block_splitter.h).  ``types`` and ``lengths`` are Python
    lists; C's ``num_blocks`` is kept as an explicit field because C allocates
    the arrays larger than the number of valid entries."""

    __slots__ = ('num_types', 'num_blocks', 'types', 'lengths')

    def __init__(self):
        # BrotliInitBlockSplit
        self.num_types = 0
        self.num_blocks = 0
        self.types = []
        self.lengths = []

    def __repr__(self):
        return 'BlockSplit(num_types=%d, num_blocks=%d, types=%r, lengths=%r)' % (
            self.num_types, self.num_blocks, self.types[:self.num_blocks],
            self.lengths[:self.num_blocks])


def brotli_init_block_split(self=None):
    """BrotliInitBlockSplit."""
    return BlockSplit()


def _ensure_capacity(lst, n, fill=0):
    if len(lst) < n:
        lst.extend([fill] * (n - len(lst)))


class MetaBlockSplit(object):
    """MetaBlockSplit (metablock.h).  Context maps are lists of ints and
    histogram arrays are lists of Histogram; as in C the ``*_size`` fields
    are explicit and must be kept in sync by the producer (they normally
    equal the list lengths)."""

    __slots__ = ('literal_split', 'command_split', 'distance_split',
                 'literal_context_map', 'literal_context_map_size',
                 'distance_context_map', 'distance_context_map_size',
                 'literal_histograms', 'literal_histograms_size',
                 'command_histograms', 'command_histograms_size',
                 'distance_histograms', 'distance_histograms_size')

    def __init__(self):
        # InitMetaBlockSplit
        self.literal_split = BlockSplit()
        self.command_split = BlockSplit()
        self.distance_split = BlockSplit()
        self.literal_context_map = []
        self.literal_context_map_size = 0
        self.distance_context_map = []
        self.distance_context_map_size = 0
        self.literal_histograms = []
        self.literal_histograms_size = 0
        self.command_histograms = []
        self.command_histograms_size = 0
        self.distance_histograms = []
        self.distance_histograms_size = 0


def init_meta_block_split():
    """InitMetaBlockSplit."""
    return MetaBlockSplit()


def brotli_calculate_distance_code_limit(max_distance, npostfix, ndirect):
    """BrotliCalculateDistanceCodeLimit (constants.h); returns
    (max_alphabet_size, max_distance)."""
    if max_distance <= ndirect:
        # This case never happens / exists only for the sake of completeness.
        return max_distance + BROTLI_NUM_DISTANCE_SHORT_CODES, max_distance
    # The first prohibited value.
    forbidden_distance = max_distance + 1
    # Subtract "directly" encoded region.
    offset = forbidden_distance - ndirect - 1
    ndistbits = 0
    # Postfix for the last dcode in the group.
    postfix = (1 << npostfix) - 1
    # Remove postfix and "head-start".
    offset = (offset >> npostfix) + 4
    # Calculate the number of distance bits.
    tmp = offset // 2
    # Poor-man's log2floor, to avoid extra dependencies.
    while tmp != 0:
        ndistbits += 1
        tmp >>= 1
    # One bit is covered with subrange addressing ("half").
    ndistbits -= 1
    # Find subrange.
    half = (offset >> ndistbits) & 1
    # Calculate the "group" part of dcode.
    group = ((ndistbits - 1) << 1) | half
    # Calculated "group" covers the prohibited distance value.
    if group == 0:
        # This case is added for correctness; does not occur for limit > 128.
        return ndirect + BROTLI_NUM_DISTANCE_SHORT_CODES, ndirect
    # Decrement "group", so it is the last permitted "group".
    group -= 1
    # After group was decremented, ndistbits and half must be recalculated.
    ndistbits = (group >> 1) + 1
    # The last available distance in the subrange has all extra bits set.
    extra = (1 << ndistbits) - 1
    # Calculate region start. NB: ndistbits >= 1.
    start = (1 << (ndistbits + 1)) - 4
    # Move to subregion.
    start += (group & 1) << ndistbits
    # Calculate the alphabet size.
    max_alphabet_size = (((group << npostfix) | postfix) + ndirect +
                         BROTLI_NUM_DISTANCE_SHORT_CODES + 1)
    # Calculate the maximal distance representable by alphabet.
    max_dist = ((start + extra) << npostfix) + postfix + ndirect + 1
    return max_alphabet_size & 0xFFFFFFFF, max_dist & 0xFFFFFFFF


def brotli_init_distance_params(dist_params, npostfix, ndirect, large_window):
    """BrotliInitDistanceParams: fills BrotliDistanceParams in place."""
    dist_params.distance_postfix_bits = npostfix
    dist_params.num_direct_distance_codes = ndirect
    alphabet_size_max = brotli_distance_alphabet_size(
        npostfix, ndirect, BROTLI_MAX_DISTANCE_BITS)
    alphabet_size_limit = alphabet_size_max
    max_distance = (ndirect + (1 << (BROTLI_MAX_DISTANCE_BITS + npostfix + 2)) -
                    (1 << (npostfix + 2))) & 0xFFFFFFFF
    if large_window:
        limit_size, limit_dist = brotli_calculate_distance_code_limit(
            BROTLI_MAX_ALLOWED_DISTANCE, npostfix, ndirect)
        alphabet_size_max = brotli_distance_alphabet_size(
            npostfix, ndirect, BROTLI_LARGE_MAX_DISTANCE_BITS)
        alphabet_size_limit = limit_size
        max_distance = limit_dist
    dist_params.alphabet_size_max = alphabet_size_max
    dist_params.alphabet_size_limit = alphabet_size_limit
    dist_params.max_distance = max_distance


# ---------------------------------------------------------------------------
# Greedy block splitter for one block category (metablock_inc.h).

class BlockSplitter(object):
    """FN(BlockSplitter) (metablock_inc.h); ``histogram_size`` selects the
    Literal / Command / Distance instantiation."""

    __slots__ = ('alphabet_size', 'min_block_size', 'split_threshold',
                 'num_blocks', 'split', 'histograms', 'histogram_size',
                 'target_block_size', 'block_size', 'curr_histogram_ix',
                 'last_histogram_ix', 'last_entropy', 'merge_last_count')

    def __init__(self, alphabet_size, histogram_size, min_block_size,
                 split_threshold, num_symbols, split, histograms):
        # FN(InitBlockSplitter); ``histograms`` is the (empty) list that
        # receives the histograms.
        max_num_blocks = num_symbols // min_block_size + 1
        # We have to allocate one more histogram than the maximum number of
        # block types for the current histogram when the meta-block is too
        # big.
        max_num_types = min(max_num_blocks, BROTLI_MAX_NUMBER_OF_BLOCK_TYPES + 1)
        self.alphabet_size = alphabet_size
        self.histogram_size = histogram_size
        self.min_block_size = min_block_size
        self.split_threshold = split_threshold
        self.num_blocks = 0
        self.split = split
        self.target_block_size = min_block_size
        self.block_size = 0
        self.curr_histogram_ix = 0
        self.merge_last_count = 0
        _ensure_capacity(split.types, max_num_blocks)
        _ensure_capacity(split.lengths, max_num_blocks)
        split.num_blocks = max_num_blocks
        del histograms[:]
        histograms.extend(Histogram(histogram_size) for _ in range(max_num_types))
        self.histograms = histograms
        self.last_histogram_ix = [0, 0]
        self.last_entropy = [0.0, 0.0]

    def finish_block(self, is_final):
        """FN(BlockSplitterFinishBlock)."""
        split = self.split
        last_entropy = self.last_entropy
        histograms = self.histograms
        alphabet_size = self.alphabet_size
        if self.block_size < self.min_block_size:
            self.block_size = self.min_block_size
        if self.num_blocks == 0:
            # Create first block.
            split.lengths[0] = self.block_size
            split.types[0] = 0
            last_entropy[0] = brotli_bits_entropy(histograms[0].data, alphabet_size)
            last_entropy[1] = last_entropy[0]
            self.num_blocks += 1
            split.num_types += 1
            self.curr_histogram_ix += 1
            if self.curr_histogram_ix < len(histograms):
                histograms[self.curr_histogram_ix].clear()
            self.block_size = 0
        elif self.block_size > 0:
            curr = histograms[self.curr_histogram_ix]
            entropy = brotli_bits_entropy(curr.data, alphabet_size)
            combined_histo = [None, None]
            combined_entropy = [0.0, 0.0]
            diff = [0.0, 0.0]
            for j in range(2):
                last_histogram_ix = self.last_histogram_ix[j]
                c = curr.copy()
                c.add_histogram(histograms[last_histogram_ix])
                combined_histo[j] = c
                combined_entropy[j] = brotli_bits_entropy(c.data, alphabet_size)
                diff[j] = combined_entropy[j] - entropy - last_entropy[j]

            if (split.num_types < BROTLI_MAX_NUMBER_OF_BLOCK_TYPES and
                    diff[0] > self.split_threshold and
                    diff[1] > self.split_threshold):
                # Create new block.
                split.lengths[self.num_blocks] = self.block_size
                split.types[self.num_blocks] = split.num_types
                self.last_histogram_ix[1] = self.last_histogram_ix[0]
                self.last_histogram_ix[0] = split.num_types & 0xFF
                last_entropy[1] = last_entropy[0]
                last_entropy[0] = entropy
                self.num_blocks += 1
                split.num_types += 1
                self.curr_histogram_ix += 1
                if self.curr_histogram_ix < len(histograms):
                    histograms[self.curr_histogram_ix].clear()
                self.block_size = 0
                self.merge_last_count = 0
                self.target_block_size = self.min_block_size
            elif diff[1] < diff[0] - 20.0:
                # Combine this block with second last block.
                split.lengths[self.num_blocks] = self.block_size
                split.types[self.num_blocks] = split.types[self.num_blocks - 2]
                lhi = self.last_histogram_ix
                lhi[0], lhi[1] = lhi[1], lhi[0]
                histograms[lhi[0]] = combined_histo[1]
                last_entropy[1] = last_entropy[0]
                last_entropy[0] = combined_entropy[1]
                self.num_blocks += 1
                self.block_size = 0
                histograms[self.curr_histogram_ix].clear()
                self.merge_last_count = 0
                self.target_block_size = self.min_block_size
            else:
                # Combine this block with last block.
                split.lengths[self.num_blocks - 1] += self.block_size
                histograms[self.last_histogram_ix[0]] = combined_histo[0]
                last_entropy[0] = combined_entropy[0]
                if split.num_types == 1:
                    last_entropy[1] = last_entropy[0]
                self.block_size = 0
                histograms[self.curr_histogram_ix].clear()
                self.merge_last_count += 1
                if self.merge_last_count > 1:
                    self.target_block_size += self.min_block_size
        if is_final:
            del histograms[split.num_types:]
            split.num_blocks = self.num_blocks

    def add_symbol(self, symbol):
        """FN(BlockSplitterAddSymbol)."""
        h = self.histograms[self.curr_histogram_ix]
        h.data[symbol] += 1
        h.total_count += 1
        self.block_size += 1
        if self.block_size == self.target_block_size:
            self.finish_block(False)


class ContextBlockSplitter(object):
    """ContextBlockSplitter (metablock.c): greedy block splitter for literals
    with a static context map."""

    __slots__ = ('alphabet_size', 'num_contexts', 'max_block_types',
                 'min_block_size', 'split_threshold', 'num_blocks', 'split',
                 'histograms', 'target_block_size', 'block_size',
                 'curr_histogram_ix', 'last_histogram_ix', 'last_entropy',
                 'merge_last_count')

    def __init__(self, alphabet_size, num_contexts, min_block_size,
                 split_threshold, num_symbols, split, histograms):
        # InitContextBlockSplitter
        max_num_blocks = num_symbols // min_block_size + 1
        self.alphabet_size = alphabet_size
        self.num_contexts = num_contexts
        self.max_block_types = BROTLI_MAX_NUMBER_OF_BLOCK_TYPES // num_contexts
        self.min_block_size = min_block_size
        self.split_threshold = split_threshold
        self.num_blocks = 0
        self.split = split
        self.target_block_size = min_block_size
        self.block_size = 0
        self.curr_histogram_ix = 0
        self.merge_last_count = 0
        # We have to allocate one more histogram than the maximum number of
        # block types for the current histogram when the meta-block is too
        # big.
        max_num_types = min(max_num_blocks, self.max_block_types + 1)
        _ensure_capacity(split.types, max_num_blocks)
        _ensure_capacity(split.lengths, max_num_blocks)
        split.num_blocks = max_num_blocks
        del histograms[:]
        histograms.extend(Histogram(BROTLI_NUM_LITERAL_SYMBOLS)
                          for _ in range(max_num_types * num_contexts))
        self.histograms = histograms
        self.last_histogram_ix = [0, 0]
        self.last_entropy = [0.0] * (2 * BROTLI_MAX_STATIC_CONTEXTS)

    def finish_block(self, is_final):
        """ContextBlockSplitterFinishBlock."""
        split = self.split
        num_contexts = self.num_contexts
        last_entropy = self.last_entropy
        histograms = self.histograms
        alphabet_size = self.alphabet_size
        if self.block_size < self.min_block_size:
            self.block_size = self.min_block_size
        if self.num_blocks == 0:
            # Create first block.
            split.lengths[0] = self.block_size
            split.types[0] = 0
            for i in range(num_contexts):
                last_entropy[i] = brotli_bits_entropy(histograms[i].data,
                                                      alphabet_size)
                last_entropy[num_contexts + i] = last_entropy[i]
            self.num_blocks += 1
            split.num_types += 1
            self.curr_histogram_ix += num_contexts
            if self.curr_histogram_ix < len(histograms):
                for i in range(num_contexts):
                    histograms[self.curr_histogram_ix + i].clear()
            self.block_size = 0
        elif self.block_size > 0:
            # Try merging the set of histograms for the current block type
            # with the respective set of histograms for the last and second
            # last block types. Decide over the split based on the total
            # reduction of entropy across all contexts.
            entropy = [0.0] * num_contexts
            combined_histo = [None] * (2 * num_contexts)
            combined_entropy = [0.0] * (2 * num_contexts)
            diff = [0.0, 0.0]
            for i in range(num_contexts):
                curr_histo_ix = self.curr_histogram_ix + i
                entropy[i] = brotli_bits_entropy(
                    histograms[curr_histo_ix].data, alphabet_size)
                for j in range(2):
                    jx = j * num_contexts + i
                    last_histogram_ix = self.last_histogram_ix[j] + i
                    c = histograms[curr_histo_ix].copy()
                    c.add_histogram(histograms[last_histogram_ix])
                    combined_histo[jx] = c
                    combined_entropy[jx] = brotli_bits_entropy(c.data,
                                                               alphabet_size)
                    diff[j] += combined_entropy[jx] - entropy[i] - last_entropy[jx]

            if (split.num_types < self.max_block_types and
                    diff[0] > self.split_threshold and
                    diff[1] > self.split_threshold):
                # Create new block.
                split.lengths[self.num_blocks] = self.block_size
                split.types[self.num_blocks] = split.num_types
                self.last_histogram_ix[1] = self.last_histogram_ix[0]
                self.last_histogram_ix[0] = split.num_types * num_contexts
                for i in range(num_contexts):
                    last_entropy[num_contexts + i] = last_entropy[i]
                    last_entropy[i] = entropy[i]
                self.num_blocks += 1
                split.num_types += 1
                self.curr_histogram_ix += num_contexts
                if self.curr_histogram_ix < len(histograms):
                    for i in range(num_contexts):
                        histograms[self.curr_histogram_ix + i].clear()
                self.block_size = 0
                self.merge_last_count = 0
                self.target_block_size = self.min_block_size
            elif diff[1] < diff[0] - 20.0:
                # Combine this block with second last block.
                split.lengths[self.num_blocks] = self.block_size
                split.types[self.num_blocks] = split.types[self.num_blocks - 2]
                lhi = self.last_histogram_ix
                lhi[0], lhi[1] = lhi[1], lhi[0]
                for i in range(num_contexts):
                    histograms[lhi[0] + i] = combined_histo[num_contexts + i]
                    last_entropy[num_contexts + i] = last_entropy[i]
                    last_entropy[i] = combined_entropy[num_contexts + i]
                    histograms[self.curr_histogram_ix + i].clear()
                self.num_blocks += 1
                self.block_size = 0
                self.merge_last_count = 0
                self.target_block_size = self.min_block_size
            else:
                # Combine this block with last block.
                split.lengths[self.num_blocks - 1] += self.block_size
                for i in range(num_contexts):
                    histograms[self.last_histogram_ix[0] + i] = combined_histo[i]
                    last_entropy[i] = combined_entropy[i]
                    if split.num_types == 1:
                        last_entropy[num_contexts + i] = last_entropy[i]
                    histograms[self.curr_histogram_ix + i].clear()
                self.block_size = 0
                self.merge_last_count += 1
                if self.merge_last_count > 1:
                    self.target_block_size += self.min_block_size
        if is_final:
            del histograms[split.num_types * num_contexts:]
            split.num_blocks = self.num_blocks

    def add_symbol(self, symbol, context):
        """ContextBlockSplitterAddSymbol."""
        h = self.histograms[self.curr_histogram_ix + context]
        h.data[symbol] += 1
        h.total_count += 1
        self.block_size += 1
        if self.block_size == self.target_block_size:
            self.finish_block(False)


def _map_static_contexts(num_contexts, static_context_map, mb):
    """MapStaticContexts."""
    n = mb.literal_split.num_types << BROTLI_LITERAL_CONTEXT_BITS
    cmap = [0] * n
    for i in range(mb.literal_split.num_types):
        offset = i * num_contexts
        base = i << BROTLI_LITERAL_CONTEXT_BITS
        for j in range(1 << BROTLI_LITERAL_CONTEXT_BITS):
            cmap[base + j] = offset + static_context_map[j]
    mb.literal_context_map = cmap
    mb.literal_context_map_size = n


def brotli_build_meta_block_greedy(ringbuffer, pos, mask, prev_byte,
                                   prev_byte2, literal_context_lut,
                                   num_contexts, static_context_map, commands,
                                   n_commands, mb):
    """BrotliBuildMetaBlockGreedy (+ BrotliBuildMetaBlockGreedyInternal):
    fills MetaBlockSplit ``mb`` in place."""
    num_literals = 0
    for i in range(n_commands):
        num_literals += commands[i].insert_len

    if num_contexts == 1:
        lit_blocks = BlockSplitter(256, BROTLI_NUM_LITERAL_SYMBOLS, 512, 400.0,
                                   num_literals, mb.literal_split,
                                   mb.literal_histograms)
    else:
        lit_blocks = ContextBlockSplitter(256, num_contexts, 512, 400.0,
                                          num_literals, mb.literal_split,
                                          mb.literal_histograms)
    cmd_blocks = BlockSplitter(BROTLI_NUM_COMMAND_SYMBOLS,
                               BROTLI_NUM_COMMAND_SYMBOLS, 1024, 500.0,
                               n_commands, mb.command_split,
                               mb.command_histograms)
    dist_blocks = BlockSplitter(64, BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS, 512,
                                100.0, n_commands, mb.distance_split,
                                mb.distance_histograms)

    lut = CONTEXT_LOOKUP
    lut_off = literal_context_lut
    lit_add = lit_blocks.add_symbol
    cmd_add = cmd_blocks.add_symbol
    dist_add = dist_blocks.add_symbol
    for i in range(n_commands):
        cmd = commands[i]
        cmd_add(cmd.cmd_prefix)
        insert_len = cmd.insert_len
        if num_contexts == 1:
            for _ in range(insert_len):
                literal = ringbuffer[pos & mask]
                lit_add(literal)
                pos += 1
            if insert_len >= 2:
                prev_byte2 = ringbuffer[(pos - 2) & mask]
                prev_byte = ringbuffer[(pos - 1) & mask]
            elif insert_len == 1:
                prev_byte2 = prev_byte
                prev_byte = ringbuffer[(pos - 1) & mask]
        else:
            for _ in range(insert_len):
                literal = ringbuffer[pos & mask]
                context = lut[lut_off + prev_byte] | lut[lut_off + 256 + prev_byte2]
                lit_add(literal, static_context_map[context])
                prev_byte2 = prev_byte
                prev_byte = literal
                pos += 1
        copy_len = cmd.copy_len & 0x1FFFFFF
        pos += copy_len
        if copy_len:
            prev_byte2 = ringbuffer[(pos - 2) & mask]
            prev_byte = ringbuffer[(pos - 1) & mask]
            if cmd.cmd_prefix >= 128:
                dist_add(cmd.dist_prefix & 0x3FF)

    lit_blocks.finish_block(True)
    cmd_blocks.finish_block(True)
    dist_blocks.finish_block(True)
    mb.literal_histograms_size = len(mb.literal_histograms)
    mb.command_histograms_size = len(mb.command_histograms)
    mb.distance_histograms_size = len(mb.distance_histograms)

    if num_contexts > 1:
        _map_static_contexts(num_contexts, static_context_map, mb)


def brotli_optimize_histograms(num_distance_codes, mb):
    """BrotliOptimizeHistograms."""
    good_for_rle = None
    for i in range(mb.literal_histograms_size):
        brotli_optimize_huffman_counts_for_rle(
            256, mb.literal_histograms[i].data, good_for_rle)
    for i in range(mb.command_histograms_size):
        brotli_optimize_huffman_counts_for_rle(
            BROTLI_NUM_COMMAND_SYMBOLS, mb.command_histograms[i].data,
            good_for_rle)
    for i in range(mb.distance_histograms_size):
        brotli_optimize_huffman_counts_for_rle(
            num_distance_codes, mb.distance_histograms[i].data, good_for_rle)
