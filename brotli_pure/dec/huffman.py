"""Utilities for building Huffman decoding tables (port of c/dec/huffman.c/.h).

Tables are plain Python lists of ints.  A C ``HuffmanCode {bits, value}`` is
packed as ``(value << 4) | bits`` (``bits`` <= 15 always fits into 4 bits);
see ``construct_huffman_code``.  As in C, a root table has
``1 << HUFFMAN_TABLE_BITS`` (= 256) entries; root entries whose ``bits`` exceed
the root size point to a 2nd-level table located ``value`` entries after the
root entry itself.
"""

HUFFMAN_MAX_CODE_LENGTH = 15

# BROTLI_NUM_BLOCK_LEN_SYMBOLS == 26
HUFFMAN_MAX_SIZE_26 = 396
# BROTLI_MAX_BLOCK_TYPE_SYMBOLS == 258
HUFFMAN_MAX_SIZE_258 = 632
# BROTLI_MAX_CONTEXT_MAP_SYMBOLS == 272
HUFFMAN_MAX_SIZE_272 = 646

HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH = 5

CODE_LENGTH_CODES = 18  # BROTLI_CODE_LENGTH_CODES

REVERSE_BITS_MAX = 8
REVERSE_BITS_LOWEST = 1 << (REVERSE_BITS_MAX - 1)

# kReverseBits: bit-wise reversal of an 8-bit value.
REVERSE_BITS = tuple(int("{:08b}".format(i)[::-1], 2) for i in range(256))


def construct_huffman_code(bits, value):
    """ConstructHuffmanCode (huffman.h)."""
    return (value << 4) | bits


def _replicate_value(table, base, step, end, code):
    """ReplicateValue: stores code in table[base + 0], [base + step], ...,
    up to (excluding) table[base + end]; ``end`` is a multiple of ``step``."""
    for k in range(base, base + end, step):
        table[k] = code


def _next_table_bit_size(count, length, root_bits):
    """NextTableBitSize: width of the next 2nd level table."""
    left = 1 << (length - root_bits)
    while length < HUFFMAN_MAX_CODE_LENGTH:
        left -= count[length]
        if left <= 0:
            break
        length += 1
        left <<= 1
    return length - root_bits


def build_code_lengths_huffman_table(table, code_lengths, count):
    """BrotliBuildCodeLengthsHuffmanTable (huffman.c).

    ``table`` must have 32 entries; ``code_lengths`` has 18 entries,
    ``count`` is the histogram of ``code_lengths`` (index 0..5).
    """
    offset = [0] * (HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH + 1)
    sorted_ = [0] * CODE_LENGTH_CODES

    # Generate offsets into sorted symbol table by code length.
    symbol = -1
    for bits in range(1, HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH + 1):
        symbol += count[bits]
        offset[bits] = symbol
    # Symbols with code length 0 are placed after all other symbols.
    offset[0] = CODE_LENGTH_CODES - 1

    # Sort symbols by length, by symbol order within each length.
    for symbol in range(CODE_LENGTH_CODES - 1, -1, -1):
        cl = code_lengths[symbol]
        sorted_[offset[cl]] = symbol
        offset[cl] -= 1

    table_size = 1 << HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH

    # Special case: all symbols but one have 0 code length.
    if offset[0] == 0:
        code = construct_huffman_code(0, sorted_[0])
        for key in range(table_size):
            table[key] = code
        return

    # Fill in table.
    key = 0
    key_step = REVERSE_BITS_LOWEST
    symbol = 0
    step = 2
    for bits in range(1, HUFFMAN_MAX_CODE_LENGTH_CODE_LENGTH + 1):
        for _ in range(count[bits]):
            code = construct_huffman_code(bits, sorted_[symbol])
            symbol += 1
            _replicate_value(table, REVERSE_BITS[key], step, table_size, code)
            key += key_step
        step <<= 1
        key_step >>= 1


def build_huffman_table(root_table, root_bits, symbol_lists, sl_off, count):
    """BrotliBuildHuffmanTable (huffman.c).

    ``symbol_lists`` is the linked-list array produced by the code length
    reader; C indexes it with negative indices relative to ``symbol_lists``,
    here ``sl_off`` is the position of C's ``symbol_lists[0]``.
    ``count`` (histogram of code lengths) is modified, as in C.
    Returns the total table size (root + 2nd level tables).
    """
    max_length = -1
    while symbol_lists[sl_off + max_length] == 0xFFFF:
        max_length -= 1
    max_length += HUFFMAN_MAX_CODE_LENGTH + 1

    table = 0  # offset of the current table inside root_table
    table_bits = root_bits
    table_size = 1 << table_bits
    total_size = table_size

    # Fill in the root table. Reduce the table size to if possible,
    # and create the repetitions by memcpy.
    if table_bits > max_length:
        table_bits = max_length
        table_size = 1 << table_bits
    key = 0
    key_step = REVERSE_BITS_LOWEST
    step = 2
    for bits in range(1, table_bits + 1):
        symbol = bits - (HUFFMAN_MAX_CODE_LENGTH + 1)
        for _ in range(count[bits]):
            symbol = symbol_lists[sl_off + symbol]
            code = (symbol << 4) | bits
            _replicate_value(root_table, table + REVERSE_BITS[key], step,
                             table_size, code)
            key += key_step
        step <<= 1
        key_step >>= 1

    # If root_bits != table_bits then replicate to fill the remaining slots.
    while total_size != table_size:
        root_table[table_size:2 * table_size] = root_table[0:table_size]
        table_size <<= 1

    # Fill in 2nd level tables and add pointers to root table.
    key_step = REVERSE_BITS_LOWEST >> (root_bits - 1)
    sub_key = REVERSE_BITS_LOWEST << 1
    sub_key_step = REVERSE_BITS_LOWEST
    step = 2
    for length in range(root_bits + 1, max_length + 1):
        symbol = length - (HUFFMAN_MAX_CODE_LENGTH + 1)
        while count[length] != 0:
            if sub_key == (REVERSE_BITS_LOWEST << 1):
                table += table_size
                table_bits = _next_table_bit_size(count, length, root_bits)
                table_size = 1 << table_bits
                total_size += table_size
                sub_key = REVERSE_BITS[key]
                key += key_step
                root_table[sub_key] = construct_huffman_code(
                    table_bits + root_bits, table - sub_key)
                sub_key = 0
            symbol = symbol_lists[sl_off + symbol]
            code = (symbol << 4) | (length - root_bits)
            _replicate_value(root_table, table + REVERSE_BITS[sub_key], step,
                             table_size, code)
            sub_key += sub_key_step
            count[length] -= 1
        step <<= 1
        sub_key_step >>= 1
    return total_size


def build_simple_huffman_table(table, root_bits, val, num_symbols):
    """BrotliBuildSimpleHuffmanTable (huffman.c).

    ``num_symbols``: 0 means 1 symbol, 1 means 2 symbols, 2 means 3 symbols,
    3 means 4 symbols with lengths [2, 2, 2, 2], 4 means 4 symbols with
    lengths [1, 2, 3, 3].  ``val`` may be reordered (as in C).
    """
    table_size = 1
    goal_size = 1 << root_bits
    c = construct_huffman_code
    if num_symbols == 0:
        table[0] = c(0, val[0])
    elif num_symbols == 1:
        if val[1] > val[0]:
            table[0] = c(1, val[0])
            table[1] = c(1, val[1])
        else:
            table[0] = c(1, val[1])
            table[1] = c(1, val[0])
        table_size = 2
    elif num_symbols == 2:
        table[0] = c(1, val[0])
        table[2] = c(1, val[0])
        if val[2] > val[1]:
            table[1] = c(2, val[1])
            table[3] = c(2, val[2])
        else:
            table[1] = c(2, val[2])
            table[3] = c(2, val[1])
        table_size = 4
    elif num_symbols == 3:
        for i in range(3):
            for k in range(i + 1, 4):
                if val[k] < val[i]:
                    val[k], val[i] = val[i], val[k]
        table[0] = c(2, val[0])
        table[2] = c(2, val[1])
        table[1] = c(2, val[2])
        table[3] = c(2, val[3])
        table_size = 4
    elif num_symbols == 4:
        if val[3] < val[2]:
            val[3], val[2] = val[2], val[3]
        table[0] = c(1, val[0])
        table[1] = c(2, val[1])
        table[2] = c(1, val[0])
        table[3] = c(3, val[2])
        table[4] = c(1, val[0])
        table[5] = c(2, val[1])
        table[6] = c(1, val[0])
        table[7] = c(3, val[3])
        table_size = 8
    while table_size != goal_size:
        table[table_size:2 * table_size] = table[0:table_size]
        table_size <<= 1
    return goal_size
