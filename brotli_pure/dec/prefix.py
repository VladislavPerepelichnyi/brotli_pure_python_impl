"""Lookup tables mapping prefix codes to value ranges (port of c/dec/prefix.c,
c/dec/prefix.h, c/dec/prefix_inc.h and _kBrotliPrefixCodeRanges from
c/common/constants.c).

Used for decoding block lengths, literal insertion lengths and copy lengths.
"""

NUM_COMMAND_SYMBOLS = 704
NUM_BLOCK_LEN_SYMBOLS = 26

# _kBrotliPrefixCodeRanges (constants.c): (offset, nbits) for block lengths.
PREFIX_CODE_RANGES = (
    (1, 2), (5, 2), (9, 2), (13, 2), (17, 3), (25, 3),
    (33, 3), (41, 3), (49, 4), (65, 4), (81, 4), (97, 4),
    (113, 5), (145, 5), (177, 5), (209, 5), (241, 6), (305, 6),
    (369, 7), (497, 8), (753, 9), (1265, 10), (2289, 11), (4337, 12),
    (8433, 13), (16625, 24),
)
BLOCK_LENGTH_OFFSET = tuple(r[0] for r in PREFIX_CODE_RANGES)
BLOCK_LENGTH_NBITS = tuple(r[1] for r in PREFIX_CODE_RANGES)

_INSERT_LENGTH_EXTRA_BITS = (
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03,
    0x04, 0x04, 0x05, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0C, 0x0E, 0x18)
_COPY_LENGTH_EXTRA_BITS = (
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x02, 0x02,
    0x03, 0x03, 0x04, 0x04, 0x05, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x18)
_CELL_POS = (0, 1, 0, 1, 8, 9, 2, 16, 10, 17, 18)


def brotli_decoder_init_cmd_lut():
    """BrotliDecoderInitCmdLut (prefix.c).

    Returns a tuple of 704 CmdLutElement tuples:
    ``(insert_len_extra_bits, copy_len_extra_bits, distance_code, context,
    insert_len_offset, copy_len_offset)``.
    """
    insert_length_offsets = [0] * 24
    copy_length_offsets = [0] * 24
    copy_length_offsets[0] = 2
    for i in range(23):
        insert_length_offsets[i + 1] = (
            insert_length_offsets[i] + (1 << _INSERT_LENGTH_EXTRA_BITS[i]))
        copy_length_offsets[i + 1] = (
            copy_length_offsets[i] + (1 << _COPY_LENGTH_EXTRA_BITS[i]))

    items = []
    for symbol in range(NUM_COMMAND_SYMBOLS):
        cell_idx = symbol >> 6
        cell_pos = _CELL_POS[cell_idx]
        copy_code = ((cell_pos << 3) & 0x18) + (symbol & 0x7)
        copy_len_offset = copy_length_offsets[copy_code]
        insert_code = (cell_pos & 0x18) + ((symbol >> 3) & 0x7)
        items.append((
            _INSERT_LENGTH_EXTRA_BITS[insert_code],
            _COPY_LENGTH_EXTRA_BITS[copy_code],
            -1 if cell_idx >= 2 else 0,
            3 if copy_len_offset > 4 else copy_len_offset - 2,
            insert_length_offsets[insert_code],
            copy_len_offset,
        ))
    return tuple(items)


# kCmdLut
CMD_LUT = brotli_decoder_init_cmd_lut()
