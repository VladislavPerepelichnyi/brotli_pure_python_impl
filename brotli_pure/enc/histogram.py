"""Models the histograms of literals, commands and distance codes
(port of c/enc/histogram.h, histogram_inc.h, histogram.c).

The three C template instantiations (HistogramLiteral, HistogramCommand,
HistogramDistance) are represented by one class ``Histogram`` parametrised by
the alphabet size; use ``histogram_literal()``, ``histogram_command()`` and
``histogram_distance()`` to create the C-sized variants.
"""

from ..common.context import CONTEXT_LOOKUP
from .command import command_copy_len, command_distance_context

BROTLI_NUM_LITERAL_SYMBOLS = 256
BROTLI_NUM_COMMAND_SYMBOLS = 704
BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS = 544
BROTLI_LITERAL_CONTEXT_BITS = 6
BROTLI_DISTANCE_CONTEXT_BITS = 2

_HUGE_VAL = float('inf')


class Histogram(object):
    """FN(Histogram): ``data`` (list of counts), ``total_count``, ``bit_cost``."""

    __slots__ = ('data', 'total_count', 'bit_cost')

    def __init__(self, alphabet_size):
        # FN(HistogramClear)
        self.data = [0] * alphabet_size
        self.total_count = 0
        self.bit_cost = _HUGE_VAL

    @property
    def data_size(self):
        """FN(HistogramDataSize)."""
        return len(self.data)

    def clear(self):
        """FN(HistogramClear): bit_cost is reset to HUGE_VAL (+inf)."""
        self.data = [0] * len(self.data)
        self.total_count = 0
        self.bit_cost = _HUGE_VAL

    def add(self, val):
        """FN(HistogramAdd)."""
        self.data[val] += 1
        self.total_count += 1

    def add_vector(self, p, n=None):
        """FN(HistogramAddVector): adds the first n symbols of sequence p."""
        if n is None:
            n = len(p)
        self.total_count += n
        data = self.data
        for i in range(n):
            data[p[i]] += 1

    def add_histogram(self, v):
        """FN(HistogramAddHistogram)."""
        self.total_count += v.total_count
        self.data = [a + b for a, b in zip(self.data, v.data)]

    def copy(self):
        """Struct assignment (``*a = *b`` in C)."""
        h = Histogram.__new__(Histogram)
        h.data = self.data[:]
        h.total_count = self.total_count
        h.bit_cost = self.bit_cost
        return h

    def assign(self, other):
        """In-place struct assignment ``*self = *other``."""
        self.data = other.data[:]
        self.total_count = other.total_count
        self.bit_cost = other.bit_cost

    def __repr__(self):
        return 'Histogram(size=%d, total=%d, bit_cost=%r)' % (
            len(self.data), self.total_count, self.bit_cost)


def histogram_literal():
    return Histogram(BROTLI_NUM_LITERAL_SYMBOLS)


def histogram_command():
    return Histogram(BROTLI_NUM_COMMAND_SYMBOLS)


def histogram_distance():
    return Histogram(BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS)


def clear_histograms(array, length=None):
    """FN(ClearHistograms)."""
    if length is None:
        length = len(array)
    for i in range(length):
        array[i].clear()


class BlockSplitIterator(object):
    """BlockSplitIterator (histogram.c)."""

    __slots__ = ('split', 'idx', 'type', 'length')

    def __init__(self, split):
        # InitBlockSplitIterator
        self.split = split
        self.idx = 0
        self.type = 0
        self.length = split.lengths[0] if split.lengths else 0

    def next(self):
        # BlockSplitIteratorNext
        if self.length == 0:
            self.idx += 1
            self.type = self.split.types[self.idx]
            self.length = self.split.lengths[self.idx]
        self.length -= 1


def brotli_build_histograms_with_context(
        cmds, num_commands, literal_split, insert_and_copy_split, dist_split,
        ringbuffer, start_pos, mask, prev_byte, prev_byte2, context_modes,
        literal_histograms, insert_and_copy_histograms, copy_dist_histograms):
    """BrotliBuildHistogramsWithContext (histogram.c).

    ``context_modes`` is a sequence of ContextType values (or None);
    histogram arguments are lists of Histogram objects (mutated in place).
    """
    pos = start_pos
    literal_it = BlockSplitIterator(literal_split)
    insert_and_copy_it = BlockSplitIterator(insert_and_copy_split)
    dist_it = BlockSplitIterator(dist_split)
    lut = CONTEXT_LOOKUP
    for i in range(num_commands):
        cmd = cmds[i]
        insert_and_copy_it.next()
        insert_and_copy_histograms[insert_and_copy_it.type].add(cmd.cmd_prefix)
        for _ in range(cmd.insert_len):
            literal_it.next()
            context = literal_it.type
            if context_modes is not None:
                off = context_modes[context] << 9
                context = ((context << BROTLI_LITERAL_CONTEXT_BITS) +
                           (lut[off + prev_byte] | lut[off + 256 + prev_byte2]))
            literal_histograms[context].add(ringbuffer[pos & mask])
            prev_byte2 = prev_byte
            prev_byte = ringbuffer[pos & mask]
            pos += 1
        copy_len = command_copy_len(cmd)
        pos += copy_len
        if copy_len:
            prev_byte2 = ringbuffer[(pos - 2) & mask]
            prev_byte = ringbuffer[(pos - 1) & mask]
            if cmd.cmd_prefix >= 128:
                dist_it.next()
                context = ((dist_it.type << BROTLI_DISTANCE_CONTEXT_BITS) +
                           command_distance_context(cmd))
                copy_dist_histograms[context].add(cmd.dist_prefix & 0x3FF)
