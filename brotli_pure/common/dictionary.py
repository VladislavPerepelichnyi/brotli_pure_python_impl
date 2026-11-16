"""Brotli static dictionary (port of c/common/dictionary.c).

The 122784-byte dictionary blob is copied verbatim from
google/brotli c/common/dictionary.bin (v1.2.1).
"""

import os

_PATH = os.path.join(os.path.dirname(__file__), "dictionary.bin")

with open(_PATH, "rb") as _f:
    DATA = _f.read()

assert len(DATA) == 122784

# Number of bits encoding the number of words of each length (index = length).
SIZE_BITS_BY_LENGTH = (
    0, 0, 0, 0, 10, 10, 11, 11,
    10, 10, 10, 10, 10, 9, 9, 8,
    7, 7, 8, 7, 7, 6, 6, 5,
    5, 0, 0, 0, 0, 0, 0, 0,
)

# Byte offset of the first word of each length (index = length).
OFFSETS_BY_LENGTH = (
    0, 0, 0, 0, 0, 4096, 9216, 21504,
    35840, 44032, 53248, 63488, 74752, 87040, 93696, 100864,
    104704, 106752, 108928, 113536, 115968, 118528, 119872, 121280,
    122016, 122784, 122784, 122784, 122784, 122784, 122784, 122784,
)

MIN_DICTIONARY_WORD_LENGTH = 4
MAX_DICTIONARY_WORD_LENGTH = 24
