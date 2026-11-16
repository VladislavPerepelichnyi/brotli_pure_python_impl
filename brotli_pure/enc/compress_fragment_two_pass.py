"""Function for fast encoding of an input fragment, independently from the
input history (port of c/enc/compress_fragment_two_pass.h / .c; quality 1).

This function uses two-pass processing: in the first pass we save the found
backward matches and literal bytes into a buffer, and in the second pass we
emit them into the bit stream using prefix codes built based on the actual
command and literal byte histograms.

Pointers into the input are represented as integer offsets into ``data``;
the command and literal buffers are Python lists.
"""

from .bit_cost import brotli_bits_entropy
from .brotli_bit_stream import (brotli_build_and_store_huffman_tree_fast,
                                brotli_store_huffman_tree)
from .compress_fragment import store_meta_block_header
from .entropy_encode import (brotli_create_huffman_tree,
                             brotli_convert_bit_depths_to_symbols)
from .find_match_length import find_match_length_with_limit
from .hash_base import kHashMul32, load64le

BROTLI_NUM_COMMAND_SYMBOLS = 704
kCompressFragmentTwoPassBlockSize = 1 << 17
MAX_DISTANCE = (1 << 18) - 16  # BROTLI_MAX_BACKWARD_LIMIT(18)
_M64 = 0xFFFFFFFFFFFFFFFF


class BrotliTwoPassArena(object):
    """BrotliTwoPassArena (compress_fragment_two_pass.h)."""

    def __init__(self):
        self.lit_histo = [0] * 256
        self.lit_depth = [0] * 256
        self.lit_bits = [0] * 256
        self.cmd_histo = [0] * 128
        self.cmd_depth = [0] * 128
        self.cmd_bits = [0] * 128
        self.tmp_depth = [0] * BROTLI_NUM_COMMAND_SYMBOLS
        self.tmp_bits = [0] * 64


def _hash(data, p, shift, length):
    """Hash (compress_fragment_two_pass.c)."""
    return ((((load64le(data, p)[0] << ((8 - length) * 8)) & _M64) *
             kHashMul32) & _M64) >> shift


def _hash_bytes_at_offset(v, offset, shift, length):
    """HashBytesAtOffset."""
    return (((((v >> (8 * offset)) << ((8 - length) * 8)) & _M64) *
             kHashMul32) & _M64) >> shift


def _is_match(data, p1, p2, length):
    """IsMatch: 4 (or 6 when length != 4) bytes equal."""
    if data[p1:p1 + 4] == data[p2:p2 + 4]:
        if length == 4:
            return True
        return data[p1 + 4] == data[p2 + 4] and data[p1 + 5] == data[p2 + 5]
    return False


def _build_and_store_command_prefix_code(s, w):
    """BuildAndStoreCommandPrefixCode (two-pass variant)."""
    tmp_depth = s.tmp_depth
    tmp_bits = s.tmp_bits
    depth = s.cmd_depth
    bits = s.cmd_bits
    # Tree size for building a tree over 64 symbols is 2 * 64 + 1.
    for i in range(BROTLI_NUM_COMMAND_SYMBOLS):
        tmp_depth[i] = 0
    d0 = depth[0:64]
    brotli_create_huffman_tree(s.cmd_histo, 64, 15, None, d0)
    depth[0:64] = d0
    d1 = depth[64:128]
    brotli_create_huffman_tree(s.cmd_histo[64:128], 64, 14, None, d1)
    depth[64:128] = d1
    # We have to jump through a few hoops here in order to compute
    # the command bits because the symbols are in a different order than in
    # the full alphabet. This looks complicated, but having the symbols
    # in this order in the command bits saves a few branches in the Emit*
    # functions.
    tmp_depth[0:24] = depth[24:48]
    tmp_depth[24:32] = depth[0:8]
    tmp_depth[32:40] = depth[48:56]
    tmp_depth[40:48] = depth[8:16]
    tmp_depth[48:56] = depth[56:64]
    tmp_depth[56:64] = depth[16:24]
    brotli_convert_bit_depths_to_symbols(tmp_depth, 64, tmp_bits)
    bits[0:8] = tmp_bits[24:32]
    bits[8:16] = tmp_bits[40:48]
    bits[16:24] = tmp_bits[56:64]
    bits[24:48] = tmp_bits[0:24]
    bits[48:56] = tmp_bits[32:40]
    bits[56:64] = tmp_bits[48:56]
    b1 = bits[64:128]
    brotli_convert_bit_depths_to_symbols(depth[64:128], 64, b1)
    bits[64:128] = b1
    # Create the bit length array for the full command alphabet.
    for i in range(64):
        tmp_depth[i] = 0  # only 64 first values were used
    tmp_depth[0:8] = depth[24:32]
    tmp_depth[64:72] = depth[32:40]
    tmp_depth[128:136] = depth[40:48]
    tmp_depth[192:200] = depth[48:56]
    tmp_depth[384:392] = depth[56:64]
    for i in range(8):
        tmp_depth[128 + 8 * i] = depth[i]
        tmp_depth[256 + 8 * i] = depth[8 + i]
        tmp_depth[448 + 8 * i] = depth[16 + i]
    brotli_store_huffman_tree(tmp_depth, BROTLI_NUM_COMMAND_SYMBOLS, None, w)
    brotli_store_huffman_tree(depth[64:128], 64, None, w)


def _emit_insert_len(insertlen, commands):
    """EmitInsertLen (two-pass): appends to the command list."""
    if insertlen < 6:
        commands.append(insertlen)
    elif insertlen < 130:
        tail = insertlen - 2
        nbits = tail.bit_length() - 2
        prefix = tail >> nbits
        inscode = (nbits << 1) + prefix + 2
        extra = tail - (prefix << nbits)
        commands.append(inscode | (extra << 8))
    elif insertlen < 2114:
        tail = insertlen - 66
        nbits = tail.bit_length() - 1
        code = nbits + 10
        extra = tail - (1 << nbits)
        commands.append(code | (extra << 8))
    elif insertlen < 6210:
        extra = insertlen - 2114
        commands.append(21 | (extra << 8))
    elif insertlen < 22594:
        extra = insertlen - 6210
        commands.append(22 | (extra << 8))
    else:
        extra = insertlen - 22594
        commands.append(23 | (extra << 8))


def _emit_copy_len(copylen, commands):
    """EmitCopyLen (two-pass)."""
    if copylen < 10:
        commands.append(copylen + 38)
    elif copylen < 134:
        tail = copylen - 6
        nbits = tail.bit_length() - 2
        prefix = tail >> nbits
        code = (nbits << 1) + prefix + 44
        extra = tail - (prefix << nbits)
        commands.append(code | (extra << 8))
    elif copylen < 2118:
        tail = copylen - 70
        nbits = tail.bit_length() - 1
        code = nbits + 52
        extra = tail - (1 << nbits)
        commands.append(code | (extra << 8))
    else:
        extra = copylen - 2118
        commands.append(63 | (extra << 8))


def _emit_copy_len_last_distance(copylen, commands):
    """EmitCopyLenLastDistance (two-pass)."""
    if copylen < 12:
        commands.append(copylen + 20)
    elif copylen < 72:
        tail = copylen - 8
        nbits = tail.bit_length() - 2
        prefix = tail >> nbits
        code = (nbits << 1) + prefix + 28
        extra = tail - (prefix << nbits)
        commands.append(code | (extra << 8))
    elif copylen < 136:
        tail = copylen - 8
        code = (tail >> 5) + 54
        extra = tail & 31
        commands.append(code | (extra << 8))
        commands.append(64)
    elif copylen < 2120:
        tail = copylen - 72
        nbits = tail.bit_length() - 1
        code = nbits + 52
        extra = tail - (1 << nbits)
        commands.append(code | (extra << 8))
        commands.append(64)
    else:
        extra = copylen - 2120
        commands.append(63 | (extra << 8))
        commands.append(64)


def _emit_distance(distance, commands):
    """EmitDistance (two-pass)."""
    d = distance + 3
    nbits = d.bit_length() - 2
    prefix = (d >> nbits) & 1
    offset = (2 + prefix) << nbits
    distcode = 2 * (nbits - 1) + prefix + 80
    extra = d - offset
    commands.append(distcode | (extra << 8))


def _update_table(data, ip, base_ip, table, shift, min_match, third_offset):
    """Updates "table" with the hashes of some positions within the last copy
    and returns the next candidate.

    ``third_offset`` reproduces an inconsistency of the C code: after a match
    with literals (compress_fragment_two_pass.c:359) the hash stored for
    position ip - 1 is computed at byte offset 0 (i.e. the hash of ip - 3),
    while after a literal-free match (line 418) it is correctly computed at
    offset 2.  See docs/notes/enc-q5.md."""
    if min_match == 4:
        input_bytes = load64le(data, ip - 3)[0]
        cur_hash = _hash_bytes_at_offset(input_bytes, 3, shift, min_match)
        prev_hash = _hash_bytes_at_offset(input_bytes, 0, shift, min_match)
        table[prev_hash] = ip - base_ip - 3
        prev_hash = _hash_bytes_at_offset(input_bytes, 1, shift, min_match)
        table[prev_hash] = ip - base_ip - 2
        prev_hash = _hash_bytes_at_offset(input_bytes, third_offset, shift,
                                          min_match)
        table[prev_hash] = ip - base_ip - 1
    else:
        input_bytes = load64le(data, ip - 5)[0]
        prev_hash = _hash_bytes_at_offset(input_bytes, 0, shift, min_match)
        table[prev_hash] = ip - base_ip - 5
        prev_hash = _hash_bytes_at_offset(input_bytes, 1, shift, min_match)
        table[prev_hash] = ip - base_ip - 4
        prev_hash = _hash_bytes_at_offset(input_bytes, 2, shift, min_match)
        table[prev_hash] = ip - base_ip - 3
        input_bytes = load64le(data, ip - 2)[0]
        cur_hash = _hash_bytes_at_offset(input_bytes, 2, shift, min_match)
        prev_hash = _hash_bytes_at_offset(input_bytes, 0, shift, min_match)
        table[prev_hash] = ip - base_ip - 2
        prev_hash = _hash_bytes_at_offset(input_bytes, 1, shift, min_match)
        table[prev_hash] = ip - base_ip - 1
    candidate = base_ip + table[cur_hash]
    table[cur_hash] = ip - base_ip
    return candidate


def _create_commands(data, input_, block_size, input_size, base_ip, table,
                     table_bits, min_match, literals, commands):
    """CreateCommands: first pass; appends literal bytes to ``literals``
    (bytearray) and command words to ``commands`` (list)."""
    # "ip" is the input pointer.
    ip = input_
    shift = 64 - table_bits
    ip_end = input_ + block_size
    # "next_emit" is a pointer to the first byte that is not covered by a
    # previous copy. Bytes between "next_emit" and the start of the next copy
    # or the end of the input will be emitted as literal bytes.
    next_emit = input_
    last_distance = -1
    kInputMarginBytes = 16  # BROTLI_WINDOW_GAP

    if block_size >= kInputMarginBytes:
        # For the last block, we need to keep a 16 bytes margin so that we can
        # be sure that all distances are at most window size - 16.
        # For all other blocks, we only need to keep a margin of 5 bytes so
        # that we don't go over the block size with a copy.
        len_limit = min(block_size - min_match, input_size - kInputMarginBytes)
        ip_limit = input_ + len_limit

        ip += 1
        next_hash = _hash(data, ip, shift, min_match)
        while True:
            # Step 1: Scan forward in the input looking for a 6-byte-long
            # match. If we get close to exhausting the input then goto
            # emit_remainder.  (Heuristic match skipping, see C.)
            skip = 32
            next_ip = ip
            emit_remainder = False
            while True:  # trawl:
                while True:
                    hash_ = next_hash
                    bytes_between_hash_lookups = skip >> 5
                    skip += 1
                    ip = next_ip
                    next_ip = ip + bytes_between_hash_lookups
                    if next_ip > ip_limit:
                        emit_remainder = True
                        break
                    next_hash = _hash(data, next_ip, shift, min_match)
                    candidate = ip - last_distance
                    if _is_match(data, ip, candidate, min_match):
                        if candidate < ip:
                            table[hash_] = ip - base_ip
                            break
                    candidate = base_ip + table[hash_]
                    table[hash_] = ip - base_ip
                    if _is_match(data, ip, candidate, min_match):
                        break
                if emit_remainder:
                    break
                # Check copy distance. If candidate is not feasible, continue
                # search.
                if ip - candidate > MAX_DISTANCE:
                    continue  # goto trawl
                break
            if emit_remainder:
                break

            # Step 2: Emit the found match together with the literal bytes from
            # "next_emit", and then see if we can find a next match immediately
            # afterwards. Repeat until we find no match for the input without
            # emitting some literal bytes.
            base = ip
            matched = min_match + find_match_length_with_limit(
                data, candidate + min_match, data, ip + min_match,
                ip_end - ip - min_match)
            distance = base - candidate  # > 0
            insert = base - next_emit
            ip += matched
            _emit_insert_len(insert, commands)
            literals += data[next_emit:next_emit + insert]
            if distance == last_distance:
                commands.append(64)
            else:
                _emit_distance(distance, commands)
                last_distance = distance
            _emit_copy_len_last_distance(matched, commands)

            next_emit = ip
            if ip >= ip_limit:
                break  # goto emit_remainder
            # We could immediately start working at ip now, but to improve
            # compression we first update "table" with the hashes of some
            # positions within the last copy.
            candidate = _update_table(data, ip, base_ip, table, shift,
                                      min_match, 0)

            reached_limit = False
            while (ip - candidate <= MAX_DISTANCE and
                   _is_match(data, ip, candidate, min_match)):
                # We have a 6-byte match at ip, and no need to emit any
                # literal bytes prior to ip.
                base = ip
                matched = min_match + find_match_length_with_limit(
                    data, candidate + min_match, data, ip + min_match,
                    ip_end - ip - min_match)
                ip += matched
                last_distance = base - candidate  # > 0
                _emit_copy_len(matched, commands)
                _emit_distance(last_distance, commands)
                next_emit = ip
                if ip >= ip_limit:
                    reached_limit = True
                    break
                candidate = _update_table(data, ip, base_ip, table, shift,
                                          min_match, 2)
            if reached_limit:
                break  # goto emit_remainder

            ip += 1
            next_hash = _hash(data, ip, shift, min_match)

    # emit_remainder:
    # Emit the remaining bytes as literals.
    if next_emit < ip_end:
        insert = ip_end - next_emit
        _emit_insert_len(insert, commands)
        literals += data[next_emit:ip_end]


kNumExtraBits = (
    0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
    6, 7, 8, 9, 10, 12, 14, 24, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16,
    17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24,
)
kInsertOffset = (
    0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26,
    34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114, 6210, 22594,
)


def _store_commands(s, literals, num_literals, commands, num_commands, w):
    """StoreCommands: second pass."""
    lit_histo = [0] * 256
    s.lit_histo = lit_histo
    # TODO(eustas): is that necessary?
    s.cmd_depth = [0] * 128
    # TODO(eustas): is that necessary?
    s.cmd_bits = [0] * 128
    cmd_histo = [0] * 128
    s.cmd_histo = cmd_histo
    for i in range(num_literals):
        lit_histo[literals[i]] += 1
    brotli_build_and_store_huffman_tree_fast(None, lit_histo, num_literals, 8,
                                             s.lit_depth, s.lit_bits, w)

    for i in range(num_commands):
        cmd_histo[commands[i] & 0xFF] += 1
    cmd_histo[1] += 1
    cmd_histo[2] += 1
    cmd_histo[64] += 1
    cmd_histo[84] += 1
    _build_and_store_command_prefix_code(s, w)

    cmd_depth = s.cmd_depth
    cmd_bits = s.cmd_bits
    lit_depth = s.lit_depth
    lit_bits = s.lit_bits
    write_bits = w.write_bits
    lit_pos = 0
    for i in range(num_commands):
        cmd = commands[i]
        code = cmd & 0xFF
        extra = cmd >> 8
        write_bits(cmd_depth[code], cmd_bits[code])
        write_bits(kNumExtraBits[code], extra)
        if code < 24:
            insert = kInsertOffset[code] + extra
            for j in range(lit_pos, lit_pos + insert):
                lit = literals[j]
                write_bits(lit_depth[lit], lit_bits[lit])
            lit_pos += insert


# Acceptable loss for uncompressible speedup is 2%
MIN_RATIO = 0.98
SAMPLE_RATE = 43


def _should_compress(s, data, input_, input_size, num_literals):
    """ShouldCompress (compress_fragment_two_pass.c)."""
    corpus_size = float(input_size)
    if float(num_literals) < MIN_RATIO * corpus_size:
        return True
    max_total_bit_cost = corpus_size * 8 * MIN_RATIO / SAMPLE_RATE
    lit_histo = [0] * 256
    s.lit_histo = lit_histo
    for i in range(input_, input_ + input_size, SAMPLE_RATE):
        lit_histo[data[i]] += 1
    return brotli_bits_entropy(lit_histo, 256) < max_total_bit_cost


def _emit_uncompressed_meta_block(data, input_, input_size, w):
    """EmitUncompressedMetaBlock (two-pass variant, no rewind)."""
    store_meta_block_header(input_size, True, w)
    w.jump_to_byte_boundary()
    w.append_bytes(data[input_:input_ + input_size])


def _compress_fragment_two_pass_impl(s, data, input_, input_size, is_last,
                                     table, table_bits, min_match, w):
    """BrotliCompressFragmentTwoPassImpl."""
    # Save the start of the first block for position and distance
    # computations.
    base_ip = input_
    while input_size > 0:
        block_size = min(input_size, kCompressFragmentTwoPassBlockSize)
        commands = []
        literals = bytearray()
        _create_commands(data, input_, block_size, input_size, base_ip, table,
                         table_bits, min_match, literals, commands)
        num_literals = len(literals)
        if _should_compress(s, data, input_, block_size, num_literals):
            num_commands = len(commands)
            store_meta_block_header(block_size, False, w)
            # No block splits, no contexts.
            w.write_bits(13, 0)
            _store_commands(s, literals, num_literals, commands, num_commands,
                            w)
        else:
            # Since we did not find many backward references and the entropy
            # of the data is close to 8 bits, we can simply emit an
            # uncompressed block. This makes compression speed of
            # uncompressible data about 3x faster.
            _emit_uncompressed_meta_block(data, input_, block_size, w)
        input_ += block_size
        input_size -= block_size


def brotli_compress_fragment_two_pass(s, data, input_, input_size, is_last,
                                      table, table_size, w):
    """BrotliCompressFragmentTwoPass: compresses data[input_:input_+size] as
    one or more complete meta-blocks; if is_last, emits an additional empty
    last meta-block.  ``table`` is a list of table_size zeros (a power of
    two between 2^8 and 2^17)."""
    initial_storage_ix = w.storage_ix
    table_bits = table_size.bit_length() - 1
    if not 8 <= table_bits <= 17:
        raise AssertionError('invalid table size')
    min_match = 4 if table_bits <= 15 else 6
    _compress_fragment_two_pass_impl(s, data, input_, input_size, is_last,
                                     table, table_bits, min_match, w)

    # If output is larger than single uncompressed block, rewrite it.
    if w.storage_ix - initial_storage_ix > 31 + (input_size << 3):
        w.rewind(initial_storage_ix)
        _emit_uncompressed_meta_block(data, input_, input_size, w)

    if is_last:
        w.write_bits(1, 1)  # islast
        w.write_bits(1, 1)  # isempty
        w.jump_to_byte_boundary()
