"""Entropy encoding (Huffman) utilities (port of c/enc/entropy_encode.h/.c).

Huffman tree pools (``HuffmanTree*`` scratch arrays in C) are internal here:
the ``tree`` arguments are accepted for signature parity and ignored.
Depth / bits arrays are mutable sequences (list or bytearray) written in
place.
"""

BROTLI_REPEAT_PREVIOUS_CODE_LENGTH = 16
BROTLI_REPEAT_ZERO_CODE_LENGTH = 17
BROTLI_CODE_LENGTH_CODES = 18
BROTLI_INITIAL_REPEATED_CODE_LENGTH = 8
BROTLI_UINT32_MAX = 0xFFFFFFFF

kBrotliShellGaps = (132, 57, 23, 10, 4, 1)


def brotli_set_depth(p0, pool_left, pool_right, depth, max_depth):
    """BrotliSetDepth.  The pool is given as two parallel arrays
    (index_left_, index_right_or_value_).  Returns True on success."""
    stack = [-1] * 16
    level = 0
    p = p0
    while True:
        if pool_left[p] >= 0:
            level += 1
            if level > max_depth:
                return False
            stack[level] = pool_right[p]
            p = pool_left[p]
            continue
        else:
            depth[pool_right[p]] = level
        while level >= 0 and stack[level] == -1:
            level -= 1
        if level < 0:
            return True
        p = stack[level]
        stack[level] = -1


def sort_huffman_tree_items(items, n, comparator):
    """SortHuffmanTreeItems: input size optimized Shell sort, ported
    literally because callers with a non-strict comparator (e.g.
    BrotliBuildAndStoreHuffmanTreeFast) depend on the exact order of equal
    elements.  ``items`` is a list, ``comparator(a, b)`` returns a bool."""
    if n < 13:
        # Insertion sort.
        for i in range(1, n):
            tmp = items[i]
            k = i
            j = i - 1
            while comparator(tmp, items[j]):
                items[k] = items[j]
                k = j
                if j == 0:
                    break
                j -= 1
            items[k] = tmp
    else:
        # Shell sort.
        g = 2 if n < 57 else 0
        while g < 6:
            gap = kBrotliShellGaps[g]
            for i in range(gap, n):
                j = i
                tmp = items[i]
                while j >= gap and comparator(tmp, items[j - gap]):
                    items[j] = items[j - gap]
                    j -= gap
                items[j] = tmp
            g += 1


def _sort_key(item):
    # SortHuffmanTree: least popular first; ties: larger value first.
    return (item[0], -item[1])


def brotli_create_huffman_tree(data, length, tree_limit, tree, depth):
    """BrotliCreateHuffmanTree: builds a length-limited Huffman tree for
    data[0:length] and writes the code lengths into depth[]."""
    count_limit = 1
    while True:
        leaves = []
        for i in range(length - 1, -1, -1):
            d = data[i]
            if d:
                leaves.append((d if d > count_limit else count_limit, i))
        n = len(leaves)
        if n == 1:
            depth[leaves[0][1]] = 1  # Only one element.
            break
        # SortHuffmanTreeItems: the comparator is a strict total order here
        # (values are unique), so any correct sort yields the same order.
        leaves.sort(key=_sort_key)
        size = 2 * n + 1
        total = [BROTLI_UINT32_MAX] * size
        left = [-1] * size
        right = [-1] * size
        for idx in range(n):
            total[idx] = leaves[idx][0]
            right[idx] = leaves[idx][1]
        # [n] and [n+1] are sentinels already.
        i = 0       # Points to the next leaf node.
        j = n + 1   # Points to the next non-leaf node.
        for k in range(n - 1, 0, -1):
            if total[i] <= total[j]:
                l_ = i
                i += 1
            else:
                l_ = j
                j += 1
            if total[i] <= total[j]:
                r_ = i
                i += 1
            else:
                r_ = j
                j += 1
            # The sentinel node becomes the parent node.
            j_end = 2 * n - k
            total[j_end] = total[l_] + total[r_]
            left[j_end] = l_
            right[j_end] = r_
            # Add back the last sentinel node.
            if j_end + 1 < size:
                total[j_end + 1] = BROTLI_UINT32_MAX
                left[j_end + 1] = -1
                right[j_end + 1] = -1
        if brotli_set_depth(2 * n - 1, left, right, depth, tree_limit):
            # We need to pack the Huffman tree in tree_limit bits. If this was
            # not successful, add fake entities to the lowest values and retry.
            break
        count_limit *= 2


def _reverse(v, start, end):
    """Reverse(v, start, end)."""
    v[start:end] = v[start:end][::-1]


def _write_huffman_tree_repetitions(previous_value, value, repetitions,
                                    tree, extra_bits_data):
    """BrotliWriteHuffmanTreeRepetitions."""
    if previous_value != value:
        tree.append(value)
        extra_bits_data.append(0)
        repetitions -= 1
    if repetitions == 7:
        tree.append(value)
        extra_bits_data.append(0)
        repetitions -= 1
    if repetitions < 3:
        for _ in range(repetitions):
            tree.append(value)
            extra_bits_data.append(0)
    else:
        start = len(tree)
        repetitions -= 3
        while True:
            tree.append(BROTLI_REPEAT_PREVIOUS_CODE_LENGTH)
            extra_bits_data.append(repetitions & 0x3)
            repetitions >>= 2
            if repetitions == 0:
                break
            repetitions -= 1
        _reverse(tree, start, len(tree))
        _reverse(extra_bits_data, start, len(extra_bits_data))


def _write_huffman_tree_repetitions_zeros(repetitions, tree, extra_bits_data):
    """BrotliWriteHuffmanTreeRepetitionsZeros."""
    if repetitions == 11:
        tree.append(0)
        extra_bits_data.append(0)
        repetitions -= 1
    if repetitions < 3:
        for _ in range(repetitions):
            tree.append(0)
            extra_bits_data.append(0)
    else:
        start = len(tree)
        repetitions -= 3
        while True:
            tree.append(BROTLI_REPEAT_ZERO_CODE_LENGTH)
            extra_bits_data.append(repetitions & 0x7)
            repetitions >>= 3
            if repetitions == 0:
                break
            repetitions -= 1
        _reverse(tree, start, len(tree))
        _reverse(extra_bits_data, start, len(extra_bits_data))


def brotli_optimize_huffman_counts_for_rle(length, counts, good_for_rle=None):
    """BrotliOptimizeHuffmanCountsForRle: modifies counts[] in place.

    Arithmetic is done with the C integer widths (uint32_t / size_t)."""
    M32 = 0xFFFFFFFF
    M64 = 0xFFFFFFFFFFFFFFFF
    streak_limit = 1240
    nonzero_count = 0
    for i in range(length):
        if counts[i]:
            nonzero_count += 1
    if nonzero_count < 16:
        return
    while length != 0 and counts[length - 1] == 0:
        length -= 1
    if length == 0:
        return  # All zeros.
    # Now counts[0..length - 1] does not have trailing zeros.
    nonzeros = 0
    smallest_nonzero = 1 << 30
    for i in range(length):
        c = counts[i]
        if c != 0:
            nonzeros += 1
            if smallest_nonzero > c:
                smallest_nonzero = c
    if nonzeros < 5:
        # Small histogram will model it well.
        return
    if smallest_nonzero < 4:
        zeros = length - nonzeros
        if zeros < 6:
            for i in range(1, length - 1):
                if counts[i - 1] != 0 and counts[i] == 0 and counts[i + 1] != 0:
                    counts[i] = 1
    if nonzeros < 28:
        return
    # 2) Let's mark all population counts that already can be encoded
    # with an RLE code.
    good_for_rle = bytearray(length)
    # Let's not spoil any of the existing good RLE codes.
    # Mark any seq of 0's that is longer as 5 as a good_for_rle.
    # Mark any seq of non-0's that is longer as 7 as a good_for_rle.
    symbol = counts[0]
    step = 0
    for i in range(length + 1):
        if i == length or counts[i] != symbol:
            if (symbol == 0 and step >= 5) or (symbol != 0 and step >= 7):
                for k in range(step):
                    good_for_rle[i - k - 1] = 1
            step = 1
            if i != length:
                symbol = counts[i]
        else:
            step += 1
    # 3) Let's replace those population counts that lead to more RLE codes.
    # Math here is in 24.8 fixed point representation.
    stride = 0
    limit = ((256 * (counts[0] + counts[1] + counts[2])) & M32) // 3 + 420
    s = 0
    for i in range(length + 1):
        if (i == length or good_for_rle[i] or
                (i != 0 and good_for_rle[i - 1]) or
                ((((256 * counts[i]) & M32) - limit + streak_limit) & M64) >=
                2 * streak_limit):
            if stride >= 4 or (stride >= 3 and s == 0):
                # The stride must end, collapse what we have, if we have
                # enough (4).
                count = (s + stride // 2) // stride
                if count == 0:
                    count = 1
                if s == 0:
                    # Don't make an all zeros stride to be upgraded to ones.
                    count = 0
                for k in range(stride):
                    # We don't want to change value at counts[i],
                    # that is already belonging to the next stride. Thus - 1.
                    counts[i - k - 1] = count & M32
            stride = 0
            s = 0
            if i < length - 2:
                # All interesting strides have a count of at least 4,
                # at least when non-zeros.
                limit = (((256 * (counts[i] + counts[i + 1] + counts[i + 2]))
                          & M32) // 3 + 420)
            elif i < length:
                limit = (256 * counts[i]) & M32
            else:
                limit = 0
        stride += 1
        if i != length:
            s += counts[i]
            if stride >= 4:
                limit = (256 * s + stride // 2) // stride
            if stride == 4:
                limit += 120


def _decide_over_rle_use(depth, length):
    """DecideOverRleUse; returns (use_rle_for_non_zero, use_rle_for_zero)."""
    total_reps_zero = 0
    total_reps_non_zero = 0
    count_reps_zero = 1
    count_reps_non_zero = 1
    i = 0
    while i < length:
        value = depth[i]
        reps = 1
        k = i + 1
        while k < length and depth[k] == value:
            reps += 1
            k += 1
        if reps >= 3 and value == 0:
            total_reps_zero += reps
            count_reps_zero += 1
        if reps >= 4 and value != 0:
            total_reps_non_zero += reps
            count_reps_non_zero += 1
        i += reps
    return (total_reps_non_zero > count_reps_non_zero * 2,
            total_reps_zero > count_reps_zero * 2)


def brotli_write_huffman_tree(depth, length, tree, extra_bits_data):
    """BrotliWriteHuffmanTree: appends the RLE-coded code lengths to the
    lists ``tree`` / ``extra_bits_data``; returns the new tree size."""
    previous_value = BROTLI_INITIAL_REPEATED_CODE_LENGTH
    use_rle_for_non_zero = False
    use_rle_for_zero = False
    # Throw away trailing zeros.
    new_length = length
    for i in range(length):
        if depth[length - i - 1] == 0:
            new_length -= 1
        else:
            break
    # First gather statistics on if it is a good idea to do RLE.
    if length > 50:
        # Find RLE coding for longer codes.
        # Shorter codes seem not to benefit from RLE.
        use_rle_for_non_zero, use_rle_for_zero = _decide_over_rle_use(
            depth, new_length)
    # Actual RLE coding.
    i = 0
    while i < new_length:
        value = depth[i]
        reps = 1
        if (value != 0 and use_rle_for_non_zero) or (value == 0 and use_rle_for_zero):
            k = i + 1
            while k < new_length and depth[k] == value:
                reps += 1
                k += 1
        if value == 0:
            _write_huffman_tree_repetitions_zeros(reps, tree, extra_bits_data)
        else:
            _write_huffman_tree_repetitions(previous_value, value, reps,
                                            tree, extra_bits_data)
            previous_value = value
        i += reps
    return len(tree)


_REV_LUT = (0x00, 0x08, 0x04, 0x0C, 0x02, 0x0A, 0x06, 0x0E,
            0x01, 0x09, 0x05, 0x0D, 0x03, 0x0B, 0x07, 0x0F)


def _brotli_reverse_bits(num_bits, bits):
    """BrotliReverseBits."""
    retval = _REV_LUT[bits & 0x0F]
    i = 4
    while i < num_bits:
        retval <<= 4
        bits >>= 4
        retval |= _REV_LUT[bits & 0x0F]
        i += 4
    retval >>= (-num_bits) & 0x03
    return retval & 0xFFFF


MAX_HUFFMAN_BITS = 16


def brotli_convert_bit_depths_to_symbols(depth, length, bits):
    """BrotliConvertBitDepthsToSymbols: writes bits[i] for depth[i] != 0."""
    bl_count = [0] * MAX_HUFFMAN_BITS
    next_code = [0] * MAX_HUFFMAN_BITS
    for i in range(length):
        bl_count[depth[i]] += 1
    bl_count[0] = 0
    code = 0
    for i in range(1, MAX_HUFFMAN_BITS):
        code = (code + bl_count[i - 1]) << 1
        next_code[i] = code & 0xFFFF
    for i in range(length):
        d = depth[i]
        if d:
            bits[i] = _brotli_reverse_bits(d, next_code[d])
            next_code[d] += 1
