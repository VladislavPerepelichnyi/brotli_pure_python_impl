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

    def update_bits(self, n_bits, bits, pos):
        """UpdateBits (compress_fragment.c): overwrites n_bits bits at bit
        position pos (which may already have been written)."""
        self.flush_bytes()
        buf = self.buf
        nbuf = len(buf)
        while n_bits > 0:
            byte_pos = pos >> 3
            n_unchanged_bits = pos & 7
            n_changed_bits = min(n_bits, 8 - n_unchanged_bits)
            total_bits = n_unchanged_bits + n_changed_bits
            mask = (~((1 << total_bits) - 1)) | ((1 << n_unchanged_bits) - 1)
            changed_bits = bits & ((1 << n_changed_bits) - 1)
            if byte_pos < nbuf:
                buf[byte_pos] = ((changed_bits << n_unchanged_bits) |
                                 (buf[byte_pos] & mask)) & 0xFF
            else:
                assert byte_pos == nbuf
                self.acc = (((changed_bits << n_unchanged_bits) |
                             (self.acc & mask)) & 0xFF)
            n_bits -= n_changed_bits
            bits >>= n_changed_bits
            pos += n_changed_bits

    def rewind(self, new_storage_ix):
        """RewindBitPosition (compress_fragment*.c)."""
        self.flush_bytes()
        bp = new_storage_ix >> 3
        if bp < len(self.buf):
            byte = self.buf[bp]
            del self.buf[bp:]
        else:
            byte = self.acc & 0xFF
        self.acc = byte & ((1 << (new_storage_ix & 7)) - 1)
        self.nbits = new_storage_ix & 7

    def get_output(self):
        """Returns (complete_bytes, last_byte, last_bits): C
        ``storage[0 : storage_ix >> 3]``, ``storage[storage_ix >> 3]`` and
        ``storage_ix & 7``."""
        self.flush_bytes()
        return bytes(self.buf), self.acc & 0xFF, self.nbits
