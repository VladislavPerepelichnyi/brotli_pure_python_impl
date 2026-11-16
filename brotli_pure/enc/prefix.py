"""Functions for encoding of integers into prefix codes the amount of extra
bits, and the actual values of the extra bits (port of c/enc/prefix.h)."""

BROTLI_NUM_DISTANCE_SHORT_CODES = 16


def prefix_encode_copy_distance(distance_code, num_direct_codes, postfix_bits):
    """PrefixEncodeCopyDistance (prefix.h).

    Returns ``(code, extra_bits)`` (the C out-parameters).  ``code`` holds the
    distance symbol in the low 10 bits and the number of extra bits above."""
    if distance_code < BROTLI_NUM_DISTANCE_SHORT_CODES + num_direct_codes:
        return distance_code, 0
    dist = ((1 << (postfix_bits + 2)) +
            (distance_code - BROTLI_NUM_DISTANCE_SHORT_CODES - num_direct_codes))
    bucket = dist.bit_length() - 2  # Log2FloorNonZero(dist) - 1
    postfix_mask = (1 << postfix_bits) - 1
    postfix = dist & postfix_mask
    prefix = (dist >> bucket) & 1
    offset = (2 + prefix) << bucket
    nbits = bucket - postfix_bits
    code = ((nbits << 10) |
            (BROTLI_NUM_DISTANCE_SHORT_CODES + num_direct_codes +
             ((2 * (nbits - 1) + prefix) << postfix_bits) + postfix)) & 0xFFFF
    extra_bits = ((dist - offset) >> postfix_bits) & 0xFFFFFFFF
    return code, extra_bits
