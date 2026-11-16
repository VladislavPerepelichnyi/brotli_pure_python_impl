"""This class models a sequence of literals and a backward reference copy
(port of c/enc/command.h, command.c).

``Command`` mirrors the C struct; field names drop the trailing underscore:
``insert_len``, ``copy_len`` (copy length in the low 25 bits and
``copy_code - copy_len`` in the high 7 bits), ``dist_extra``, ``cmd_prefix``,
``dist_prefix`` (distance symbol in the low 10 bits, number of extra bits in
the high 6 bits).
"""

from .prefix import prefix_encode_copy_distance, BROTLI_NUM_DISTANCE_SHORT_CODES

BROTLI_NUM_INS_COPY_CODES = 24

kBrotliInsBase = (
    0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26,
    34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114, 6210, 22594)
kBrotliInsExtra = (
    0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24)
kBrotliCopyBase = (
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18,
    22, 30, 38, 54, 70, 102, 134, 198, 326, 582, 1094, 2118)
kBrotliCopyExtra = (
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24)


def get_insert_length_code(insertlen):
    """GetInsertLengthCode."""
    if insertlen < 6:
        return insertlen
    elif insertlen < 130:
        nbits = (insertlen - 2).bit_length() - 2
        return (nbits << 1) + ((insertlen - 2) >> nbits) + 2
    elif insertlen < 2114:
        return (insertlen - 66).bit_length() - 1 + 10
    elif insertlen < 6210:
        return 21
    elif insertlen < 22594:
        return 22
    else:
        return 23


def get_copy_length_code(copylen):
    """GetCopyLengthCode."""
    if copylen < 10:
        return (copylen - 2) & 0xFFFF
    elif copylen < 134:
        nbits = (copylen - 6).bit_length() - 2
        return (nbits << 1) + ((copylen - 6) >> nbits) + 4
    elif copylen < 2118:
        return (copylen - 70).bit_length() - 1 + 12
    else:
        return 23


def combine_length_codes(inscode, copycode, use_last_distance):
    """CombineLengthCodes."""
    bits64 = (copycode & 0x7) | ((inscode & 0x7) << 3)
    if use_last_distance and inscode < 8 and copycode < 16:
        return bits64 if copycode < 8 else (bits64 | 64)
    else:
        # Specification: 5 Encoding of ... (last table)
        # offset = 2 * index, where index is in range [0..8]
        offset = 2 * ((copycode >> 3) + 3 * (inscode >> 3))
        # All values in specification are K * 64,
        # where   K = [2, 3, 6, 4, 5, 8, 7, 9, 10],
        #     i + 1 = [1, 2, 3, 4, 5, 6, 7, 8,  9],
        # K - i - 1 = [1, 1, 3, 0, 0, 2, 0, 1,  2] = D.
        # All values in D require only 2 bits to encode.
        # Magic constant is shifted 6 bits left, to avoid final multiplication.
        offset = (offset << 5) + 0x40 + ((0x520D40 >> offset) & 0xC0)
        return (offset | bits64) & 0xFFFF


def get_length_code(insertlen, copylen, use_last_distance):
    """GetLengthCode; returns the code (C out-parameter)."""
    inscode = get_insert_length_code(insertlen)
    copycode = get_copy_length_code(copylen)
    return combine_length_codes(inscode, copycode, use_last_distance)


def get_insert_base(inscode):
    return kBrotliInsBase[inscode]


def get_insert_extra(inscode):
    return kBrotliInsExtra[inscode]


def get_copy_base(copycode):
    return kBrotliCopyBase[copycode]


def get_copy_extra(copycode):
    return kBrotliCopyExtra[copycode]


class Command(object):
    """Command struct (command.h)."""

    __slots__ = ('insert_len', 'copy_len', 'dist_extra', 'cmd_prefix',
                 'dist_prefix')

    def __init__(self, insert_len=0, copy_len=0, dist_extra=0, cmd_prefix=0,
                 dist_prefix=0):
        self.insert_len = insert_len
        self.copy_len = copy_len
        self.dist_extra = dist_extra
        self.cmd_prefix = cmd_prefix
        self.dist_prefix = dist_prefix

    def copy(self):
        return Command(self.insert_len, self.copy_len, self.dist_extra,
                       self.cmd_prefix, self.dist_prefix)

    def __eq__(self, other):
        return (isinstance(other, Command) and
                self.insert_len == other.insert_len and
                self.copy_len == other.copy_len and
                self.dist_extra == other.dist_extra and
                self.cmd_prefix == other.cmd_prefix and
                self.dist_prefix == other.dist_prefix)

    __hash__ = None

    def __repr__(self):
        return ('Command(insert_len=%d, copy_len=0x%x, dist_extra=%d, '
                'cmd_prefix=%d, dist_prefix=0x%x)' % (
                    self.insert_len, self.copy_len, self.dist_extra,
                    self.cmd_prefix, self.dist_prefix))


def init_command(dist, insertlen, copylen, copylen_code_delta, distance_code):
    """InitCommand: returns a new, initialised Command.

    ``dist`` is a BrotliDistanceParams.  distance_code is e.g. 0 for
    same-as-last short code, or 16 for offset 1."""
    self = Command.__new__(Command)
    # Don't rely on signed int representation, use honest casts.
    delta = copylen_code_delta & 0xFF
    self.insert_len = insertlen & 0xFFFFFFFF
    self.copy_len = (copylen | (delta << 25)) & 0xFFFFFFFF
    # The distance prefix and extra bits are stored in this Command as if
    # npostfix and ndirect were 0, they are only recomputed later after the
    # clustering if needed.
    self.dist_prefix, self.dist_extra = prefix_encode_copy_distance(
        distance_code, dist.num_direct_distance_codes,
        dist.distance_postfix_bits)
    self.cmd_prefix = get_length_code(
        insertlen, copylen + copylen_code_delta,
        (self.dist_prefix & 0x3FF) == 0)
    return self


def init_insert_command(insertlen):
    """InitInsertCommand: returns a new insert-only Command."""
    self = Command.__new__(Command)
    self.insert_len = insertlen & 0xFFFFFFFF
    self.copy_len = 4 << 25
    self.dist_extra = 0
    self.dist_prefix = BROTLI_NUM_DISTANCE_SHORT_CODES
    self.cmd_prefix = get_length_code(insertlen, 4, False)
    return self


def command_restore_distance_code(self, dist):
    """CommandRestoreDistanceCode."""
    dist_prefix = self.dist_prefix
    if (dist_prefix & 0x3FF) < (BROTLI_NUM_DISTANCE_SHORT_CODES +
                                dist.num_direct_distance_codes):
        return dist_prefix & 0x3FF
    else:
        npostfix = dist.distance_postfix_bits
        ndirect = dist.num_direct_distance_codes
        dcode = dist_prefix & 0x3FF
        nbits = dist_prefix >> 10
        extra = self.dist_extra
        postfix_mask = (1 << npostfix) - 1
        hcode = (dcode - ndirect - BROTLI_NUM_DISTANCE_SHORT_CODES) >> npostfix
        lcode = (dcode - ndirect - BROTLI_NUM_DISTANCE_SHORT_CODES) & postfix_mask
        offset = ((2 + (hcode & 1)) << nbits) - 4
        return (((offset + extra) << npostfix) + lcode + ndirect +
                BROTLI_NUM_DISTANCE_SHORT_CODES) & 0xFFFFFFFF


def command_distance_context(self):
    """CommandDistanceContext."""
    r = self.cmd_prefix >> 6
    c = self.cmd_prefix & 7
    if (r == 0 or r == 2 or r == 4 or r == 7) and c <= 2:
        return c
    return 3


def command_copy_len(self):
    """CommandCopyLen."""
    return self.copy_len & 0x1FFFFFF


def command_copy_len_code(self):
    """CommandCopyLenCode."""
    modifier = self.copy_len >> 25
    delta = (modifier | ((modifier & 0x40) << 1)) & 0xFF
    if delta >= 128:
        delta -= 256
    return ((self.copy_len & 0x1FFFFFF) + delta) & 0xFFFFFFFF
