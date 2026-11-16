"""Bit reading helpers (port of c/dec/bit_reader.h / bit_reader.c).

The C bit reader keeps a 64-bit accumulator ``val_`` holding ``bit_pos_``
not-yet-consumed bits, and pulls more bytes from ``next_in`` when needed.
This port does the same with a Python int accumulator.  Because the whole
input is available up front, the "safe" (resumable) and "fast" variants of
the C reader collapse into one:

* Refills load 4 bytes at once.  Reading past the end of the input yields
  zero bits (a short slice), i.e. the reader may *peek* beyond the end.
* Whether bits beyond the end were actually *consumed* is checked on every
  refill and, exactly, at byte-alignment points and at the end of the
  stream.  Consuming bits past the end is exactly the situation where the C
  decoder returns BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT, which the Python
  binding (and we) report as an error.

Hot loops in ``decode.py`` copy ``val`` / ``bit_pos`` / ``next_in`` into
local variables and inline the operations below.
"""

from ..common.errors import BrotliError

# kBrotliBitMask
BIT_MASK = tuple((1 << n) - 1 for n in range(33))


def _overrun():
    raise BrotliError("brotli: decoder failed (truncated input: "
                      "BROTLI_DECODER_NEEDS_MORE_INPUT)")


class BrotliBitReader(object):
    """BrotliBitReader (bit_reader.h)."""

    __slots__ = ("buf", "val", "bit_pos", "next_in", "last_in")

    def __init__(self, data):
        # BrotliInitBitReader + BrotliBitReaderSetInput
        self.buf = data          # bytes
        self.val = 0             # pre-fetched bits (val_)
        self.bit_pos = 0         # number of valid bits in val (bit_pos_)
        self.next_in = 0         # index of the next byte to load
        self.last_in = len(data)

    # -- bookkeeping -------------------------------------------------------

    def consumed_bits(self):
        return (self.next_in << 3) - self.bit_pos

    def check_overrun(self):
        """Raises if bits past the end of input were consumed."""
        if (self.next_in << 3) - self.bit_pos > (self.last_in << 3):
            _overrun()

    def warmup(self):
        """BrotliWarmupBitReader: at least one byte of input is required."""
        if self.bit_pos == 0 and self.next_in >= self.last_in:
            _overrun()

    def fill(self, n_bits):
        """BrotliFillBitWindow: guarantees at least ``n_bits`` (<= 32) bits."""
        bit_pos = self.bit_pos
        if bit_pos < n_bits:
            next_in = self.next_in
            if (next_in << 3) - bit_pos > (self.last_in << 3):
                _overrun()
            self.val |= int.from_bytes(
                self.buf[next_in:next_in + 4], "little") << bit_pos
            self.next_in = next_in + 4
            self.bit_pos = bit_pos + 32

    # -- reading -----------------------------------------------------------

    def get_bits(self, n_bits):
        """BrotliSafeGetBits / BrotliGetBits: peek ``n_bits`` bits."""
        if self.bit_pos < n_bits:
            self.fill(n_bits)
        return self.val & BIT_MASK[n_bits]

    def drop_bits(self, n_bits):
        """BrotliDropBits."""
        self.val >>= n_bits
        self.bit_pos -= n_bits

    def read_bits(self, n_bits):
        """BrotliSafeReadBits / BrotliReadBits24 / BrotliReadBits32."""
        if n_bits == 0:
            return 0
        if self.bit_pos < n_bits:
            self.fill(n_bits)
        v = self.val & BIT_MASK[n_bits]
        self.val >>= n_bits
        self.bit_pos -= n_bits
        return v

    def read_symbol(self, table):
        """ReadSymbol / SafeReadSymbol (decode.c): decodes one Huffman symbol
        using a table built by huffman.py (8-bit root table)."""
        if self.bit_pos < 15:
            self.fill(15)
        val = self.val
        e = table[val & 0xFF]
        nb = e & 15
        if nb > 8:
            val >>= 8
            e = table[(self.val & 0xFF) + (e >> 4) + (val & BIT_MASK[nb - 8])]
            nb = (e & 15) + 8
            self.val = val >> (nb - 8)
        else:
            self.val = val >> nb
        self.bit_pos -= nb
        return e >> 4

    # -- byte-aligned access -------------------------------------------------

    def jump_to_byte_boundary(self):
        """BrotliJumpToByteBoundary: skips to the byte boundary; returns
        False if any skipped (padding) bit is non-zero."""
        self.check_overrun()
        pad_bits_count = self.bit_pos & 7
        pad_bits = 0
        if pad_bits_count:
            pad_bits = self.read_bits(pad_bits_count)
        return pad_bits == 0

    def byte_position(self):
        """Index of the next unread byte; the reader must be byte-aligned."""
        return ((self.next_in << 3) - self.bit_pos) >> 3

    def set_byte_position(self, pos):
        """Discards the accumulator and continues reading at byte ``pos``
        (BrotliCopyBytes / BrotliDropBytes after draining the accumulator)."""
        self.val = 0
        self.bit_pos = 0
        self.next_in = pos

    def remaining_bytes(self):
        """BrotliGetRemainingBytes (uncapped); reader must be byte-aligned."""
        return self.last_in - self.byte_position()
