"""Decoder state (port of c/dec/state.h / state.c).

Only the fields that are meaningful for a one-shot, whole-input decoder are
kept: the resumable sub-state enums of the C state machine are not needed
because ``decode.py`` runs each stage to completion.  The output is kept as
one linear ``bytearray`` instead of a ring buffer (see the note in
decode.py).
"""

from .bit_reader import BrotliBitReader

BLOCK_SIZE_CAP = 1 << 24  # BROTLI_BLOCK_SIZE_CAP


class HuffmanTreeGroup(object):
    """HuffmanTreeGroup (huffman.h): a collection of Huffman tables with the
    same alphabet size."""

    __slots__ = ("htrees", "alphabet_size_max", "alphabet_size_limit",
                 "num_htrees")

    def __init__(self, alphabet_size_max, alphabet_size_limit, ntrees):
        # BrotliDecoderHuffmanTreeGroupInit (state.c)
        self.alphabet_size_max = alphabet_size_max
        self.alphabet_size_limit = alphabet_size_limit
        self.num_htrees = ntrees
        self.htrees = [None] * ntrees


class BrotliDecoderState(object):
    """BrotliDecoderStateStruct (state.h)."""

    def __init__(self, data):
        # BrotliDecoderStateInit (state.c)
        self.br = BrotliBitReader(data)
        self.large_window = False
        self.output = bytearray()   # replaces ringbuffer / pos / rb_roundtrips
        self.window_bits = 0
        self.max_backward_distance = 0
        self.max_distance = 0
        self.dist_rb = [16, 15, 11, 4]
        self.dist_rb_idx = 0
        self.mtf_upper_bound = 63
        self.mtf = bytearray(range(256))
        self.is_last_metablock = 0
        self.is_uncompressed = 0
        self.is_metadata = 0
        self.size_nibbles = 0
        self.meta_block_remaining_len = 0
        self.metablock_begin()

    def metablock_begin(self):
        """BrotliDecoderStateMetablockBegin (state.c)."""
        self.meta_block_remaining_len = 0
        self.block_length = [BLOCK_SIZE_CAP, BLOCK_SIZE_CAP, BLOCK_SIZE_CAP]
        self.num_block_types = [1, 1, 1]
        self.block_type_rb = [1, 0, 1, 0, 1, 0]
        self.block_type_trees = [None, None, None]
        self.block_len_trees = [None, None, None]
        self.context_map = None
        self.context_modes = None
        self.dist_context_map = None
        self.context_map_slice = 0       # offset into context_map
        self.literal_htree = None
        self.dist_context_map_slice = 0  # offset into dist_context_map
        self.dist_htree_index = 0
        self.context_lookup = 0
        self.trivial_literal_context = 0
        self.trivial_literal_contexts = None
        self.literal_hgroup = None
        self.insert_copy_hgroup = None
        self.distance_hgroup = None
        self.htree_command = None
        self.num_literal_htrees = 0
        self.num_dist_htrees = 0
        self.distance_postfix_bits = 0
        self.num_direct_distance_codes = 0
        self.dist_extra_bits = None
        self.dist_offset = None
        self.distance_code = 0
        self.distance_context = 0

    def cleanup_after_metablock(self):
        """BrotliDecoderStateCleanupAfterMetablock (state.c)."""
        self.context_modes = None
        self.context_map = None
        self.dist_context_map = None
        self.literal_hgroup = None
        self.insert_copy_hgroup = None
        self.distance_hgroup = None
