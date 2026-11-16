"""Basic common hash functions / constants (port of c/enc/hash_base.h)."""

import struct

# kHashMul32 multiplier has these properties:
# * The multiplier must be odd. Otherwise we may lose the highest bit.
# * No long streaks of ones or zeros.
# * There is no effort to ensure that it is a prime, the oddity is enough
#   for this use.
# * The number has been tuned heuristically against compression benchmarks.
kHashMul32 = 0x1E35A7BD
kHashMul64 = (0x1FE35A7B << 32) | 0xD3579BD3

#: BROTLI_UNALIGNED_LOAD32LE(&buf[off]) == load32le(buf, off)[0]
load32le = struct.Struct('<I').unpack_from
#: BROTLI_UNALIGNED_LOAD64LE(&buf[off]) == load64le(buf, off)[0]
load64le = struct.Struct('<Q').unpack_from


def hash14(data, off):
    """Hash14(&data[off])."""
    return ((load32le(data, off)[0] * kHashMul32) & 0xFFFFFFFF) >> (32 - 14)


def hash15(data, off):
    """Hash15(&data[off])."""
    return ((load32le(data, off)[0] * kHashMul32) & 0xFFFFFFFF) >> (32 - 15)
