"""Heuristics for deciding about the UTF8-ness of strings
(port of c/enc/utf8_util.c / utf8_util.h)."""

import re

# kMinUTF8Ratio (utf8_util.h)
kMinUTF8Ratio = 0.75

# Runs of bytes 0x01..0x7F: each parses as a 1-byte UTF8 symbol.
_ASCII_RUN = re.compile(b"[\x01-\x7f]+")


def brotli_parse_as_utf8(input_, off, size):
    """BrotliParseAsUTF8: parses input_[off:off+size]; returns
    (symbol, bytes_read)."""
    b0 = input_[off]
    # ASCII
    if (b0 & 0x80) == 0:
        if b0 > 0:
            return b0, 1
    # 2-byte UTF8
    if size > 1:
        b1 = input_[off + 1]
        if (b0 & 0xE0) == 0xC0 and (b1 & 0xC0) == 0x80:
            symbol = ((b0 & 0x1F) << 6) | (b1 & 0x3F)
            if symbol > 0x7F:
                return symbol, 2
        # 3-byte UFT8
        if size > 2:
            b2 = input_[off + 2]
            if ((b0 & 0xF0) == 0xE0 and (b1 & 0xC0) == 0x80 and
                    (b2 & 0xC0) == 0x80):
                symbol = (((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) |
                          (b2 & 0x3F))
                if symbol > 0x7FF:
                    return symbol, 3
            # 4-byte UFT8
            if size > 3:
                b3 = input_[off + 3]
                if ((b0 & 0xF8) == 0xF0 and (b1 & 0xC0) == 0x80 and
                        (b2 & 0xC0) == 0x80 and (b3 & 0xC0) == 0x80):
                    symbol = (((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) |
                              ((b2 & 0x3F) << 6) | (b3 & 0x3F))
                    if 0xFFFF < symbol <= 0x10FFFF:
                        return symbol, 4
    # Not UTF8, emit a special symbol above the UTF8-code space
    return 0x110000 | b0, 1


def brotli_is_mostly_utf8(data, pos, mask, length, min_fraction):
    """BrotliIsMostlyUTF8: returns True if at least min_fraction of the data
    is UTF8-encoded.

    The C code parses &data[(pos + i) & mask] with up to 3 bytes of
    look-ahead read linearly past the masked position (i.e. from the ring
    buffer tail copy, which mirrors the wrapped-around data); the bytes are
    gathered here in logical order, which is equivalent.  Runs of ASCII
    bytes 0x01..0x7F (1-byte symbols) are counted in bulk."""
    start = pos & mask
    if start + length <= mask + 1:
        buf = bytes(data[start:start + length])
    else:
        head = mask + 1 - start
        buf = bytes(data[start:mask + 1]) + bytes(data[0:length - head])
    size_utf8 = 0
    i = 0
    parse = brotli_parse_as_utf8
    match = _ASCII_RUN.match
    while i < length:
        m = match(buf, i)
        if m is not None:
            e = m.end()
            size_utf8 += e - i
            i = e
            continue
        symbol, bytes_read = parse(buf, i, length - i)
        i += bytes_read
        if symbol < 0x110000:
            size_utf8 += bytes_read
    return float(size_utf8) > min_fraction * float(length)
