"""Dictionary word transforms (port of c/common/transform.c / transform.h).

RFC 7932 Appendix B defines 121 transforms; each one is a triplet
``[prefix_id, type, suffix_id]``.  ``prefix_id`` / ``suffix_id`` index
``PREFIX_SUFFIX_MAP`` which points into the ``PREFIX_SUFFIX`` blob; every
entry there is a length byte followed by that many bytes.

Public API (usable by both decoder and encoder):

    RFC_TRANSFORMS                       the BrotliTransforms instance (C: BrotliGetTransforms())
    get_transforms()                     same, as a function
    transform_dictionary_word(dst, word, length, transforms, transform_idx)
                                         C-style: appends to bytearray ``dst``, returns length
    transform_word(word, transform_idx, transforms=RFC_TRANSFORMS) -> bytes
                                         convenience wrapper
"""

# enum BrotliWordTransformType (transform.h)
IDENTITY = 0
OMIT_LAST_1 = 1
OMIT_LAST_2 = 2
OMIT_LAST_3 = 3
OMIT_LAST_4 = 4
OMIT_LAST_5 = 5
OMIT_LAST_6 = 6
OMIT_LAST_7 = 7
OMIT_LAST_8 = 8
OMIT_LAST_9 = 9
UPPERCASE_FIRST = 10
UPPERCASE_ALL = 11
OMIT_FIRST_1 = 12
OMIT_FIRST_2 = 13
OMIT_FIRST_3 = 14
OMIT_FIRST_4 = 15
OMIT_FIRST_5 = 16
OMIT_FIRST_6 = 17
OMIT_FIRST_7 = 18
OMIT_FIRST_8 = 19
OMIT_FIRST_9 = 20
SHIFT_FIRST = 21
SHIFT_ALL = 22
NUM_TRANSFORM_TYPES = 23

TRANSFORMS_MAX_CUT_OFF = OMIT_LAST_9

# RFC 7932 transforms string data (kPrefixSuffix); "\0" - implicit trailing zero.
PREFIX_SUFFIX = (
    b"\1 \2, \10 of the \4 of \2s \1.\5 and \4 "
    b"in \1\"\4 to \2\">\1\n\2. \1]\5 for \3 a \6 "
    b"that \1\'\6 with \6 from \4 by \1(\6. T"
    b"he \4 on \4 as \4 is \4ing \2\n\t\1:\3ed "
    b"\2=\"\4 at \3ly \1,\2=\'\5.com/\7. This \5"
    b" not \3er \3al \4ful \4ive \5less \4es"
    b"t \4ize \2\xc2\xa0\4ous \5 the \2e "
    b"\0"
)
assert len(PREFIX_SUFFIX) == 217

# kPrefixSuffixMap
PREFIX_SUFFIX_MAP = (
    0x00, 0x02, 0x05, 0x0E, 0x13, 0x16, 0x18, 0x1E, 0x23, 0x25,
    0x2A, 0x2D, 0x2F, 0x32, 0x34, 0x3A, 0x3E, 0x45, 0x47, 0x4E,
    0x55, 0x5A, 0x5C, 0x63, 0x68, 0x6D, 0x72, 0x77, 0x7A, 0x7C,
    0x80, 0x83, 0x88, 0x8C, 0x8E, 0x91, 0x97, 0x9F, 0xA5, 0xA9,
    0xAD, 0xB2, 0xB7, 0xBD, 0xC2, 0xC7, 0xCA, 0xCF, 0xD5, 0xD8,
)

# kTransformsData: [prefix_id, type, suffix_id] triplets.
TRANSFORMS_DATA = (
    49, IDENTITY, 49,
    49, IDENTITY,  0,
     0, IDENTITY,  0,
    49, OMIT_FIRST_1, 49,
    49, UPPERCASE_FIRST,  0,
    49, IDENTITY, 47,
     0, IDENTITY, 49,
     4, IDENTITY,  0,
    49, IDENTITY,  3,
    49, UPPERCASE_FIRST, 49,
    49, IDENTITY,  6,
    49, OMIT_FIRST_2, 49,
    49, OMIT_LAST_1, 49,
     1, IDENTITY,  0,
    49, IDENTITY,  1,
     0, UPPERCASE_FIRST,  0,
    49, IDENTITY,  7,
    49, IDENTITY,  9,
    48, IDENTITY,  0,
    49, IDENTITY,  8,
    49, IDENTITY,  5,
    49, IDENTITY, 10,
    49, IDENTITY, 11,
    49, OMIT_LAST_3, 49,
    49, IDENTITY, 13,
    49, IDENTITY, 14,
    49, OMIT_FIRST_3, 49,
    49, OMIT_LAST_2, 49,
    49, IDENTITY, 15,
    49, IDENTITY, 16,
     0, UPPERCASE_FIRST, 49,
    49, IDENTITY, 12,
     5, IDENTITY, 49,
     0, IDENTITY,  1,
    49, OMIT_FIRST_4, 49,
    49, IDENTITY, 18,
    49, IDENTITY, 17,
    49, IDENTITY, 19,
    49, IDENTITY, 20,
    49, OMIT_FIRST_5, 49,
    49, OMIT_FIRST_6, 49,
    47, IDENTITY, 49,
    49, OMIT_LAST_4, 49,
    49, IDENTITY, 22,
    49, UPPERCASE_ALL, 49,
    49, IDENTITY, 23,
    49, IDENTITY, 24,
    49, IDENTITY, 25,
    49, OMIT_LAST_7, 49,
    49, OMIT_LAST_1, 26,
    49, IDENTITY, 27,
    49, IDENTITY, 28,
     0, IDENTITY, 12,
    49, IDENTITY, 29,
    49, OMIT_FIRST_9, 49,
    49, OMIT_FIRST_7, 49,
    49, OMIT_LAST_6, 49,
    49, IDENTITY, 21,
    49, UPPERCASE_FIRST,  1,
    49, OMIT_LAST_8, 49,
    49, IDENTITY, 31,
    49, IDENTITY, 32,
    47, IDENTITY,  3,
    49, OMIT_LAST_5, 49,
    49, OMIT_LAST_9, 49,
     0, UPPERCASE_FIRST,  1,
    49, UPPERCASE_FIRST,  8,
     5, IDENTITY, 21,
    49, UPPERCASE_ALL,  0,
    49, UPPERCASE_FIRST, 10,
    49, IDENTITY, 30,
     0, IDENTITY,  5,
    35, IDENTITY, 49,
    47, IDENTITY,  2,
    49, UPPERCASE_FIRST, 17,
    49, IDENTITY, 36,
    49, IDENTITY, 33,
     5, IDENTITY,  0,
    49, UPPERCASE_FIRST, 21,
    49, UPPERCASE_FIRST,  5,
    49, IDENTITY, 37,
     0, IDENTITY, 30,
    49, IDENTITY, 38,
     0, UPPERCASE_ALL,  0,
    49, IDENTITY, 39,
     0, UPPERCASE_ALL, 49,
    49, IDENTITY, 34,
    49, UPPERCASE_ALL,  8,
    49, UPPERCASE_FIRST, 12,
     0, IDENTITY, 21,
    49, IDENTITY, 40,
     0, UPPERCASE_FIRST, 12,
    49, IDENTITY, 41,
    49, IDENTITY, 42,
    49, UPPERCASE_ALL, 17,
    49, IDENTITY, 43,
     0, UPPERCASE_FIRST,  5,
    49, UPPERCASE_ALL, 10,
     0, IDENTITY, 34,
    49, UPPERCASE_FIRST, 33,
    49, IDENTITY, 44,
    49, UPPERCASE_ALL,  5,
    45, IDENTITY, 49,
     0, IDENTITY, 33,
    49, UPPERCASE_FIRST, 30,
    49, UPPERCASE_ALL, 30,
    49, IDENTITY, 46,
    49, UPPERCASE_ALL,  1,
    49, UPPERCASE_FIRST, 34,
     0, UPPERCASE_FIRST, 33,
     0, UPPERCASE_ALL, 30,
     0, UPPERCASE_ALL,  1,
    49, UPPERCASE_ALL, 33,
    49, UPPERCASE_ALL, 21,
    49, UPPERCASE_ALL, 12,
     0, UPPERCASE_ALL,  5,
    49, UPPERCASE_ALL, 34,
     0, UPPERCASE_ALL, 12,
     0, UPPERCASE_FIRST, 30,
     0, UPPERCASE_ALL, 34,
     0, UPPERCASE_FIRST, 34,
)
assert len(TRANSFORMS_DATA) == 3 * 121


class BrotliTransforms(object):
    """struct BrotliTransforms (transform.h).

    In addition to the C fields, ``prefixes`` / ``suffixes`` / ``types`` hold
    the per-transform prefix bytes, suffix bytes and operator, pre-extracted
    for fast access.
    """

    __slots__ = ("prefix_suffix_size", "prefix_suffix", "prefix_suffix_map",
                 "num_transforms", "transforms", "params", "cut_off_transforms",
                 "prefixes", "suffixes", "types")

    def __init__(self, prefix_suffix, prefix_suffix_map, transforms, params,
                 cut_off_transforms):
        self.prefix_suffix_size = len(prefix_suffix)
        self.prefix_suffix = prefix_suffix
        self.prefix_suffix_map = prefix_suffix_map
        self.num_transforms = len(transforms) // 3
        self.transforms = transforms
        # Shift parameters (2 bytes per transform) for SHIFT_FIRST / SHIFT_ALL;
        # None if and only if no such transforms are present.
        self.params = params
        # Indices of transforms like ["", OMIT_LAST_#, ""]; -1 if absent.
        self.cut_off_transforms = cut_off_transforms
        self.prefixes = tuple(self._affix(transforms[3 * i])
                              for i in range(self.num_transforms))
        self.suffixes = tuple(self._affix(transforms[3 * i + 2])
                              for i in range(self.num_transforms))
        self.types = tuple(transforms[3 * i + 1]
                           for i in range(self.num_transforms))

    def _affix(self, affix_id):
        # BROTLI_TRANSFORM_PREFIX / BROTLI_TRANSFORM_SUFFIX: length-prefixed.
        off = self.prefix_suffix_map[affix_id]
        n = self.prefix_suffix[off]
        return bytes(self.prefix_suffix[off + 1:off + 1 + n])

    def prefix_id(self, i):
        return self.transforms[i * 3]

    def type(self, i):
        return self.transforms[i * 3 + 1]

    def suffix_id(self, i):
        return self.transforms[i * 3 + 2]


# kBrotliTransforms
RFC_TRANSFORMS = BrotliTransforms(
    PREFIX_SUFFIX, PREFIX_SUFFIX_MAP, TRANSFORMS_DATA, None,
    (0, 12, 27, 23, 42, 63, 56, 48, 59, 64))
NUM_RFC_TRANSFORMS = RFC_TRANSFORMS.num_transforms


def get_transforms():
    """BrotliGetTransforms()."""
    return RFC_TRANSFORMS


def to_upper_case(p, i):
    """ToUpperCase (transform.c): uppercases the "scalar" at p[i] in place.

    Returns the number of bytes consumed.  ``p`` is a bytearray.  Like the C
    code, may touch p[i + 1] / p[i + 2] for multi-byte sequences; callers
    guarantee those bytes exist (they are part of the word or the suffix).
    """
    c = p[i]
    if c < 0xC0:
        if 97 <= c <= 122:  # 'a'..'z'
            p[i] = c ^ 32
        return 1
    # An overly simplified uppercasing model for UTF-8.
    if c < 0xE0:
        p[i + 1] ^= 32
        return 2
    # An arbitrary transform for three byte characters.
    p[i + 2] ^= 5
    return 3


def shift(word, i, word_len, parameter):
    """Shift (transform.c): shifts the UTF-8-like scalar at word[i] in place.

    Returns the number of bytes consumed.
    """
    # Limited sign extension: scalar < (1 << 24).
    scalar = ((parameter & 0x7FFF) + (0x1000000 - (parameter & 0x8000))) & 0xFFFFFFFF
    c0 = word[i]
    if c0 < 0x80:
        # 1-byte rune / 0sssssss / 7 bit scalar (ASCII).
        scalar += c0
        word[i] = scalar & 0x7F
        return 1
    elif c0 < 0xC0:
        # Continuation / 10AAAAAA.
        return 1
    elif c0 < 0xE0:
        # 2-byte rune / 110sssss AAssssss / 11 bit scalar.
        if word_len < 2:
            return 1
        scalar += (word[i + 1] & 0x3F) | ((c0 & 0x1F) << 6)
        word[i] = 0xC0 | ((scalar >> 6) & 0x1F)
        word[i + 1] = (word[i + 1] & 0xC0) | (scalar & 0x3F)
        return 2
    elif c0 < 0xF0:
        # 3-byte rune / 1110ssss AAssssss BBssssss / 16 bit scalar.
        if word_len < 3:
            return word_len
        scalar += ((word[i + 2] & 0x3F) | ((word[i + 1] & 0x3F) << 6) |
                   ((c0 & 0x0F) << 12))
        word[i] = 0xE0 | ((scalar >> 12) & 0x0F)
        word[i + 1] = (word[i + 1] & 0xC0) | ((scalar >> 6) & 0x3F)
        word[i + 2] = (word[i + 2] & 0xC0) | (scalar & 0x3F)
        return 3
    elif c0 < 0xF8:
        # 4-byte rune / 11110sss AAssssss BBssssss CCssssss / 21 bit scalar.
        if word_len < 4:
            return word_len
        scalar += ((word[i + 3] & 0x3F) | ((word[i + 2] & 0x3F) << 6) |
                   ((word[i + 1] & 0x3F) << 12) | ((c0 & 0x07) << 18))
        word[i] = 0xF0 | ((scalar >> 18) & 0x07)
        word[i + 1] = (word[i + 1] & 0xC0) | ((scalar >> 12) & 0x3F)
        word[i + 2] = (word[i + 2] & 0xC0) | ((scalar >> 6) & 0x3F)
        word[i + 3] = (word[i + 3] & 0xC0) | (scalar & 0x3F)
        return 4
    return 1


def transform_dictionary_word(dst, word, length, transforms, transform_idx):
    """BrotliTransformDictionaryWord (transform.c).

    Appends the transformed word to bytearray ``dst``.  ``word`` is a
    bytes-like object holding (at least) ``length`` bytes of the dictionary
    word.  Returns the number of bytes appended.
    """
    start = len(dst)
    prefix = transforms.prefixes[transform_idx]
    t = transforms.types[transform_idx]
    suffix = transforms.suffixes[transform_idx]
    dst += prefix
    skip = 0
    if t <= OMIT_LAST_9:
        length -= t
    elif OMIT_FIRST_1 <= t <= OMIT_FIRST_9:
        skip = t - (OMIT_FIRST_1 - 1)
        length -= skip
    if length > 0:
        dst += word[skip:skip + length]
        i = len(dst) - length
        if t == UPPERCASE_FIRST:
            # The C code operates in place on dst; ToUpperCase may look one
            # or two bytes past the word (into the suffix / slack area).
            _upper_with_slack(dst, i, suffix, first_only=True, length=length)
        elif t == UPPERCASE_ALL:
            _upper_with_slack(dst, i, suffix, first_only=False, length=length)
        elif t == SHIFT_FIRST:
            param = (transforms.params[transform_idx * 2] +
                     (transforms.params[transform_idx * 2 + 1] << 8)) & 0xFFFF
            shift(dst, i, length, param)
        elif t == SHIFT_ALL:
            param = (transforms.params[transform_idx * 2] +
                     (transforms.params[transform_idx * 2 + 1] << 8)) & 0xFFFF
            while length > 0:
                step = shift(dst, i, length, param)
                i += step
                length -= step
    dst += suffix
    return len(dst) - start


def _upper_with_slack(dst, i, suffix, first_only, length):
    """Runs ToUpperCase over dst[i:i+length] exactly as the C code does.

    In C the bytes following the word in ``dst`` are the suffix (written
    afterwards, overwriting any change) or uninitialised slack.  A multi-byte
    "scalar" near the end of the word can therefore modify up to two bytes
    beyond the word; those modifications are discarded because the suffix is
    written over them later.  We emulate that with temporary padding.
    """
    end = i + length
    dst += b"\0\0"
    if first_only:
        to_upper_case(dst, i)
    else:
        while length > 0:
            step = to_upper_case(dst, i)
            i += step
            length -= step
    del dst[end:]


def transform_word(word, transform_idx, transforms=RFC_TRANSFORMS):
    """Convenience: returns the transformed ``word`` (bytes) as bytes."""
    dst = bytearray()
    transform_dictionary_word(dst, word, len(word), transforms, transform_idx)
    return bytes(dst)
