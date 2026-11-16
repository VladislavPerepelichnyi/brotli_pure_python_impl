"""Sliding window over the input data (port of c/enc/ringbuffer.h).

A RingBuffer(window_bits, tail_bits) contains `1 << window_bits' bytes of
data in a circular manner: writing a byte writes it to
`position() % (1 << window_bits)'.  For convenience, the RingBuffer array
contains another copy of the first `1 << tail_bits' bytes:
buffer[i] == buffer[i + (1 << window_bits)], if i < (1 << tail_bits).

``buffer`` corresponds to the C ``buffer_`` pointer (``data_ + 2``); the two
bytes before it (copies of the last two bytes) are never read by the
encoder and are not modelled.
"""

from .quality import compute_rb_bits

kSlackForEightByteHashingEverywhere = 7


class RingBuffer(object):
    __slots__ = ('size', 'mask', 'tail_size', 'total_size', 'cur_size', 'pos',
                 'buffer')

    def __init__(self):
        # RingBufferInit
        self.size = 0
        self.mask = 0
        self.tail_size = 0
        self.total_size = 0
        self.cur_size = 0
        self.pos = 0
        self.buffer = None

    def setup(self, params):
        """RingBufferSetup."""
        window_bits = compute_rb_bits(params)
        tail_bits = params.lgblock
        self.size = 1 << window_bits
        self.mask = (1 << window_bits) - 1
        self.tail_size = 1 << tail_bits
        self.total_size = self.size + self.tail_size

    def init_buffer(self, buflen):
        """RingBufferInitBuffer: (re)allocates the buffer to buflen bytes plus
        slack; the slack after buflen is zeroed."""
        new_data = bytearray(buflen + kSlackForEightByteHashingEverywhere)
        if self.buffer is not None:
            n = self.cur_size + kSlackForEightByteHashingEverywhere
            new_data[0:n] = self.buffer[0:n]
        self.buffer = new_data
        self.cur_size = buflen
        for i in range(kSlackForEightByteHashingEverywhere):
            new_data[buflen + i] = 0

    def write_tail(self, data, start, n):
        """RingBufferWriteTail."""
        masked_pos = self.pos & self.mask
        if masked_pos < self.tail_size:
            # Just fill the tail buffer with the beginning data.
            p = self.size + masked_pos
            k = min(n, self.tail_size - masked_pos)
            self.buffer[p:p + k] = data[start:start + k]

    def write(self, data, start, n):
        """RingBufferWrite: pushes data[start:start+n] into the ring buffer."""
        if self.pos == 0 and n < self.tail_size:
            # Special case for the first write: to process the first block,
            # we don't need to allocate the whole ring-buffer and we don't
            # need the tail either. However, we do this memory usage
            # optimization only if the first write is less than the tail size,
            # which is also the input block size, otherwise it is likely that
            # other blocks will follow and we will need to reallocate to the
            # full size anyway.
            self.pos = n
            self.init_buffer(self.pos)
            self.buffer[0:n] = data[start:start + n]
            return
        if self.cur_size < self.total_size:
            # Lazily allocate the full buffer.
            self.init_buffer(self.total_size)
            # Initialize the last two bytes to zero, so that we don't have to
            # worry later when we copy the last two bytes to the first two
            # positions.
            self.buffer[self.size - 2] = 0
            self.buffer[self.size - 1] = 0
            # Initialize tail; might be touched by "best_len++" optimization
            # when ring buffer is "full".
            self.buffer[self.size] = 241
        masked_pos = self.pos & self.mask
        # The length of the writes is limited so that we do not need to worry
        # about a write
        self.write_tail(data, start, n)
        buf = self.buffer
        if masked_pos + n <= self.size:
            # A single write fits.
            buf[masked_pos:masked_pos + n] = data[start:start + n]
        else:
            # Split into two writes.
            # Copy into the end of the buffer, including the tail buffer.
            k = min(n, self.total_size - masked_pos)
            buf[masked_pos:masked_pos + k] = data[start:start + k]
            # Copy into the beginning of the buffer
            k2 = n - (self.size - masked_pos)
            s2 = start + (self.size - masked_pos)
            buf[0:k2] = data[s2:s2 + k2]
        not_first_lap = (self.pos & (1 << 31)) != 0
        rb_pos_mask = (1 << 31) - 1
        self.pos = (self.pos & rb_pos_mask) + (n & rb_pos_mask)
        if not_first_lap:
            # Wrap, but preserve not-a-first-lap feature.
            self.pos |= 1 << 31
