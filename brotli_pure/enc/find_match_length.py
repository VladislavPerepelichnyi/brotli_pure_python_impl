"""Function to find maximal matching prefixes of strings
(port of c/enc/find_match_length.h).

The C function takes two pointers; here each pointer is a (buffer, offset)
pair.  The result is the same for the 64-bit TZCNT variant and the portable
variant: the length of the common prefix of s1[i1:] and s2[i2:], capped at
``limit``.
"""


def find_match_length_with_limit(s1, i1, s2, i2, limit):
    """FindMatchLengthWithLimit(&s1[i1], &s2[i2], limit)."""
    n = 0
    # Short matches are the common case: compare a few bytes directly.
    while n < limit and n < 8:
        if s1[i1 + n] != s2[i2 + n]:
            return n
        n += 1
    if n == limit:
        return n
    # Exponential + binary search on slice equality for long matches.
    step = 8
    while n + step <= limit and s1[i1 + n:i1 + n + step] == s2[i2 + n:i2 + n + step]:
        n += step
        step <<= 1
    while step > 8:
        step >>= 1
        if n + step <= limit and s1[i1 + n:i1 + n + step] == s2[i2 + n:i2 + n + step]:
            n += step
    while n < limit and s1[i1 + n] == s2[i2 + n]:
        n += 1
    return n
