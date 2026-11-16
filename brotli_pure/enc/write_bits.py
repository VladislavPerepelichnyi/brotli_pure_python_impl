"""Write bits into a byte array (port of c/enc/write_bits.h).

The C functions operate on ``(size_t* pos, uint8_t* array)``; here a
``BitWriter`` object holds both.  Bits are written into bytes in increasing
addresses, and within a byte least-significant-bit first.  ``storage_ix`` is
the C bit position.
"""


class BitWriter(object):
    """Accumulating bit writer equivalent to BrotliWriteBits on a zeroed
    storage array.  ``initial_bits`` / ``initial_nbits`` model the C idiom
    ``storage[0..1] = last_bytes; storage_ix = last_bytes_bits``."""

    __slots__ = ('buf', 'acc', 'nbits')

    def __init__(self, initial_bits=0, initial_nbits=0):
        self.buf = bytearray()
        self.acc = initial_bits
        self.nbits = initial_nbits

    @property
    def storage_ix(self):
        return (len(self.buf) << 3) + self.nbits

    def write_bits(self, n_bits, bits):
        """BrotliWriteBits(n_bits, bits, &storage_ix, storage)."""
        self.acc |= bits << self.nbits
        self.nbits += n_bits
        if self.nbits >= 64:
            k = self.nbits >> 3
            self.buf += (self.acc & ((1 << (k << 3)) - 1)).to_bytes(k, 'little')
            self.acc >>= k << 3
            self.nbits &= 7

    def flush_bytes(self):
        """Moves all complete bytes of the accumulator into ``buf``."""
        k = self.nbits >> 3
        if k:
            self.buf += (self.acc & ((1 << (k << 3)) - 1)).to_bytes(k, 'little')
            self.acc >>= k << 3
            self.nbits &= 7

    def jump_to_byte_boundary(self):
        """JumpToByteBoundary (brotli_bit_stream.c): pad with zero bits."""
        self.nbits = (self.nbits + 7) & ~7
        self.flush_bytes()

    def append_bytes(self, data):
        """memcpy(&storage[storage_ix >> 3], data, len); storage_ix += len*8.
        REQUIRES: byte-aligned position."""
        self.flush_bytes()
        assert self.nbits == 0
        self.buf += data

    def get_output(self):
        """Returns (complete_bytes, last_byte, last_bits): C
        ``storage[0 : storage_ix >> 3]``, ``storage[storage_ix >> 3]`` and
        ``storage_ix & 7``."""
        self.flush_bytes()
        return bytes(self.buf), self.acc & 0xFF, self.nbits
