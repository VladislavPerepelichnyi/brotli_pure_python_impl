"""Utilities for fast computation of logarithms (port of c/enc/fast_log.h/.c)."""

import math
import struct
from array import array

BROTLI_LOG2_TABLE_SIZE = 256
LOG_2_INV = 1.4426950408889634


def _to_float32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


# kBrotliLog2Table: the C table is written with "f" literals, i.e. every entry
# is log2(i) rounded to IEEE single precision and then widened to double.
# (Verified entry-by-entry against the compiled C table, see
# tools/check_log2_table.c.)
kBrotliLog2Table = array('d', [0.0] + [_to_float32(math.log2(i))
                                       for i in range(1, BROTLI_LOG2_TABLE_SIZE)])
_LOG2_TABLE = tuple(kBrotliLog2Table)

_log2 = math.log2


def log2_floor_non_zero(n):
    """Log2FloorNonZero (fast_log.h)."""
    return n.bit_length() - 1


def fast_log2(v):
    """FastLog2 (fast_log.h): log2 with log2(0) == 0, table for v < 256."""
    if v < 256:
        return _LOG2_TABLE[v]
    return _log2(v)


def fast_pow2(v):
    """FastPow2: not present in brotli v1.2.1 (kept for API completeness)."""
    return 2.0 ** v
