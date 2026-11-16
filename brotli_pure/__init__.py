"""Pure-Python port of Google's Brotli (reference implementation: google/brotli v1.2.1).

Public API (mirrors the C-backed ``brotli`` PyPI module):

    decompress(data: bytes) -> bytes
    compress(data: bytes, mode=MODE_GENERIC, quality=11, lgwin=22, lgblock=0) -> bytes
"""

from .common.constants import MODE_GENERIC, MODE_TEXT, MODE_FONT


class error(Exception):
    """Raised on invalid compressed data or invalid parameters."""


def decompress(data):
    from .dec.decode import decompress as _decompress
    return _decompress(data)


def compress(data, mode=MODE_GENERIC, quality=11, lgwin=22, lgblock=0):
    from .enc.encode import compress as _compress
    return _compress(data, mode=mode, quality=quality, lgwin=lgwin, lgblock=lgblock)


__all__ = ["compress", "decompress", "error", "MODE_GENERIC", "MODE_TEXT", "MODE_FONT"]
