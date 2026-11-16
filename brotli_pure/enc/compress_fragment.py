"""Function for fast encoding of an input fragment, independently from the
input history (port of c/enc/compress_fragment.h / .c; quality 0).

This function uses one-pass processing: when we find a backward match, we
immediately emit the corresponding command and literal codes to the bit
stream.  Adapted (in C) from the CompressFragment() function in snappy.

Pointers into the input are represented as integer offsets into ``data``.
"""

from .brotli_bit_stream import (brotli_build_and_store_huffman_tree_fast,
                                brotli_store_huffman_tree)
from .entropy_encode import (brotli_create_huffman_tree,
                             brotli_convert_bit_depths_to_symbols)
from .fast_log import fast_log2
from .find_match_length import find_match_length_with_limit
from .hash_base import kHashMul32, load64le
from .write_bits import BitWriter

BROTLI_NUM_COMMAND_SYMBOLS = 704
MAX_DISTANCE = (1 << 18) - 16  # BROTLI_MAX_BACKWARD_LIMIT(18)
_M64 = 0xFFFFFFFFFFFFFFFF

kDefaultCommandDepths = (
    0, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8,
    0, 0, 0, 4, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7,
    7, 7, 10, 10, 10, 10, 10, 10, 0, 4, 4, 5, 5, 5, 6, 6,
    7, 8, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4,
    4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 10,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 0, 0, 0, 0,
)
kDefaultCommandBits = (
    0, 0, 8, 9, 3, 35, 7, 71,
    39, 103, 23, 47, 175, 111, 239, 31,
    0, 0, 0, 4, 12, 2, 10, 6,
    13, 29, 11, 43, 27, 59, 87, 55,
    15, 79, 319, 831, 191, 703, 447, 959,
    0, 14, 1, 25, 5, 21, 19, 51,
    119, 159, 95, 223, 479, 991, 63, 575,
    127, 639, 383, 895, 255, 767, 511, 1023,
    14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    27, 59, 7, 39, 23, 55, 30, 1, 17, 9, 25, 5, 0, 8, 4, 12,
    2, 10, 6, 21, 13, 29, 3, 19, 11, 15, 47, 31, 95, 63, 127, 255,
    767, 2815, 1791, 3839, 511, 2559, 1535, 3583, 1023, 3071, 2047, 4095,
    0, 0, 0, 0,
)
kDefaultCommandCode = bytes((
    0xff, 0x77, 0xd5, 0xbf, 0xe7, 0xde, 0xea, 0x9e, 0x51, 0x5d, 0xde, 0xc6,
    0x70, 0x57, 0xbc, 0x58, 0x58, 0x58, 0xd8, 0xd8, 0x58, 0xd5, 0xcb, 0x8c,
    0xea, 0xe0, 0xc3, 0x87, 0x1f, 0x83, 0xc1, 0x60, 0x1c, 0x67, 0xb2, 0xaa,
    0x06, 0x83, 0xc1, 0x60, 0x30, 0x18, 0xcc, 0xa1, 0xce, 0x88, 0x54, 0x94,
    0x46, 0xe1, 0xb0, 0xd0, 0x4e, 0xb2, 0xf7, 0x04, 0x00,
))
kDefaultCommandCodeNumBits = 448

kCmdHistoSeed = (
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 0, 0, 0, 0,
)


class BrotliOnePassArena(object):
    """BrotliOnePassArena (compress_fragment.h)."""

    def __init__(self):
        self.lit_depth = [0] * 256
        self.lit_bits = [0] * 256
        # Command and distance prefix codes (each 64 symbols, stored
        # back-to-back) used for the next block.
        self.cmd_depth = [0] * 128
        self.cmd_bits = [0] * 128
        self.cmd_histo = [0] * 128
        # The compressed form of the command and distance prefix codes for
        # the next block.
        self.cmd_code = b''
        self.cmd_code_numbits = 0
        self.histogram = [0] * 256
        self.tmp_depth = [0] * BROTLI_NUM_COMMAND_SYMBOLS
        self.tmp_bits = [0] * 64


def init_command_prefix_codes(s):
    """InitCommandPrefixCodes (encode.c): initializes the command and
    distance prefix codes for the first block."""
    s.cmd_depth = list(kDefaultCommandDepths)
    s.cmd_bits = list(kDefaultCommandBits)
    # Initialize the pre-compressed form of the command and distance prefix
    # codes.
    s.cmd_code = kDefaultCommandCode
    s.cmd_code_numbits = kDefaultCommandCodeNumBits


def _hash(data, p, shift):
    """Hash (compress_fragment.c)."""
    return ((((load64le(data, p)[0] << 24) & _M64) * kHashMul32) & _M64) >> shift


def _hash_bytes_at_offset(v, offset, shift):
    """HashBytesAtOffset."""
    return (((((v >> (8 * offset)) << 24) & _M64) * kHashMul32) & _M64) >> shift


def _is_match(data, p1, p2):
    """IsMatch: 5 bytes equal."""
    return data[p1:p1 + 5] == data[p2:p2 + 5]


def _build_and_store_literal_prefix_code(s, data, input_, input_size, depths,
                                         bits, w):
    """BuildAndStoreLiteralPrefixCode: returns the estimated compression
    ratio in millibytes per char."""
    histogram = [0] * 256
    s.histogram = histogram
    if input_size < (1 << 15):
        for i in range(input_, input_ + input_size):
            histogram[data[i]] += 1
        histogram_total = input_size
        for i in range(256):
            # We weigh the first 11 samples with weight 3 to account for the
            # balancing effect of the LZ77 phase on the histogram.
            adjust = 2 * min(histogram[i], 11)
            histogram[i] += adjust
            histogram_total += adjust
    else:
        kSampleRate = 29
        for i in range(input_, input_ + input_size, kSampleRate):
            histogram[data[i]] += 1
        histogram_total = (input_size + kSampleRate - 1) // kSampleRate
        for i in range(256):
            # We add 1 to each population count to avoid 0 bit depths (since
            # this is only a sample and we don't know if the symbol appears
            # or not), and we weigh the first 11 samples with weight 3 to
            # account for the balancing effect of the LZ77 phase on the
            # histogram (more frequent symbols are more likely to be in
            # backward references instead as literals).
            adjust = 1 + 2 * min(histogram[i], 11)
            histogram[i] += adjust
            histogram_total += adjust
    brotli_build_and_store_huffman_tree_fast(None, histogram, histogram_total,
                                             8, depths, bits, w)
    literal_ratio = 0
    for i in range(256):
        if histogram[i]:
            literal_ratio += histogram[i] * depths[i]
    # Estimated encoding ratio, millibytes per symbol.
    return (literal_ratio * 125) // histogram_total


def _build_and_store_command_prefix_code(s, w):
    """BuildAndStoreCommandPrefixCode (compress_fragment.c): builds a command
    and distance prefix code (each 64 symbols) into s.cmd_depth / s.cmd_bits
    based on s.cmd_histo and stores it into the bit stream."""
    histogram = s.cmd_histo
    depth = s.cmd_depth
    bits = s.cmd_bits
    tmp_depth = s.tmp_depth
    tmp_bits = s.tmp_bits
    for i in range(BROTLI_NUM_COMMAND_SYMBOLS):
        tmp_depth[i] = 0
    # BrotliCreateHuffmanTree only writes the depth of present symbols; the
    # other entries keep their previous values, as in C.
    d0 = depth[0:64]
    brotli_create_huffman_tree(histogram, 64, 15, None, d0)
    depth[0:64] = d0
    d1 = depth[64:128]
    brotli_create_huffman_tree(histogram[64:128], 64, 14, None, d1)
    depth[64:128] = d1
    # We have to jump through a few hoops here in order to compute
    # the command bits because the symbols are in a different order than in
    # the full alphabet. This looks complicated, but having the symbols
    # in this order in the command bits saves a few branches in the Emit*
    # functions.
    tmp_depth[0:24] = depth[0:24]
    tmp_depth[24:32] = depth[40:48]
    tmp_depth[32:40] = depth[24:32]
    tmp_depth[40:48] = depth[48:56]
    tmp_depth[48:56] = depth[32:40]
    tmp_depth[56:64] = depth[56:64]
    brotli_convert_bit_depths_to_symbols(tmp_depth, 64, tmp_bits)
    bits[0:24] = tmp_bits[0:24]
    bits[24:32] = tmp_bits[32:40]
    bits[32:40] = tmp_bits[48:56]
    bits[40:48] = tmp_bits[24:32]
    bits[48:56] = tmp_bits[40:48]
    bits[56:64] = tmp_bits[56:64]
    b1 = bits[64:128]
    brotli_convert_bit_depths_to_symbols(depth[64:128], 64, b1)
    bits[64:128] = b1
    # Create the bit length array for the full command alphabet.
    for i in range(64):
        tmp_depth[i] = 0  # only 64 first values were used
    tmp_depth[0:8] = depth[0:8]
    tmp_depth[64:72] = depth[8:16]
    tmp_depth[128:136] = depth[16:24]
    tmp_depth[192:200] = depth[24:32]
    tmp_depth[384:392] = depth[32:40]
    for i in range(8):
        tmp_depth[128 + 8 * i] = depth[40 + i]
        tmp_depth[256 + 8 * i] = depth[48 + i]
        tmp_depth[448 + 8 * i] = depth[56 + i]
    # TODO(eustas): could/should full-length machinery be avoided?
    brotli_store_huffman_tree(tmp_depth, BROTLI_NUM_COMMAND_SYMBOLS, None, w)
    brotli_store_huffman_tree(depth[64:128], 64, None, w)


def _emit_insert_len(insertlen, depth, bits, histo, w):
    """EmitInsertLen.  REQUIRES: insertlen < 6210."""
    if insertlen < 6:
        code = insertlen + 40
        w.write_bits(depth[code], bits[code])
        histo[code] += 1
    elif insertlen < 130:
        tail = insertlen - 2
        nbits = tail.bit_length() - 2
        prefix = tail >> nbits
        inscode = (nbits << 1) + prefix + 42
        w.write_bits(depth[inscode], bits[inscode])
        w.write_bits(nbits, tail - (prefix << nbits))
        histo[inscode] += 1
    elif insertlen < 2114:
        tail = insertlen - 66
        nbits = tail.bit_length() - 1
        code = nbits + 50
        w.write_bits(depth[code], bits[code])
        w.write_bits(nbits, tail - (1 << nbits))
        histo[code] += 1
    else:
        w.write_bits(depth[61], bits[61])
        w.write_bits(12, insertlen - 2114)
        histo[61] += 1


def _emit_long_insert_len(insertlen, depth, bits, histo, w):
    """EmitLongInsertLen."""
    if insertlen < 22594:
        w.write_bits(depth[62], bits[62])
        w.write_bits(14, insertlen - 6210)
        histo[62] += 1
    else:
        w.write_bits(depth[63], bits[63])
        w.write_bits(24, insertlen - 22594)
        histo[63] += 1


def _emit_copy_len(copylen, depth, bits, histo, w):
    """EmitCopyLen."""
    if copylen < 10:
        w.write_bits(depth[copylen + 14], bits[copylen + 14])
        histo[copylen + 14] += 1
    elif copylen < 134:
        tail = copylen - 6
        nbits = tail.bit_length() - 2
        prefix = tail >> nbits
        code = (nbits << 1) + prefix + 20
        w.write_bits(depth[code], bits[code])
        w.write_bits(nbits, tail - (prefix << nbits))
        histo[code] += 1
    elif copylen < 2118:
        tail = copylen - 70
        nbits = tail.bit_length() - 1
        code = nbits + 28
        w.write_bits(depth[code], bits[code])
        w.write_bits(nbits, tail - (1 << nbits))
        histo[code] += 1
    else:
        w.write_bits(depth[39], bits[39])
        w.write_bits(24, copylen - 2118)
        histo[39] += 1


def _emit_copy_len_last_distance(copylen, depth, bits, histo, w):
    """EmitCopyLenLastDistance."""
    if copylen < 12:
        w.write_bits(depth[copylen - 4], bits[copylen - 4])
        histo[copylen - 4] += 1
    elif copylen < 72:
        tail = copylen - 8
        nbits = tail.bit_length() - 2
        prefix = tail >> nbits
        code = (nbits << 1) + prefix + 4
        w.write_bits(depth[code], bits[code])
        w.write_bits(nbits, tail - (prefix << nbits))
        histo[code] += 1
    elif copylen < 136:
        tail = copylen - 8
        code = (tail >> 5) + 30
        w.write_bits(depth[code], bits[code])
        w.write_bits(5, tail & 31)
        w.write_bits(depth[64], bits[64])
        histo[code] += 1
        histo[64] += 1
    elif copylen < 2120:
        tail = copylen - 72
        nbits = tail.bit_length() - 1
        code = nbits + 28
        w.write_bits(depth[code], bits[code])
        w.write_bits(nbits, tail - (1 << nbits))
        w.write_bits(depth[64], bits[64])
        histo[code] += 1
        histo[64] += 1
    else:
        w.write_bits(depth[39], bits[39])
        w.write_bits(24, copylen - 2120)
        w.write_bits(depth[64], bits[64])
        histo[39] += 1
        histo[64] += 1


def _emit_distance(distance, depth, bits, histo, w):
    """EmitDistance."""
    d = distance + 3
    nbits = d.bit_length() - 2
    prefix = (d >> nbits) & 1
    offset = (2 + prefix) << nbits
    distcode = 2 * (nbits - 1) + prefix + 80
    w.write_bits(depth[distcode], bits[distcode])
    w.write_bits(nbits, d - offset)
    histo[distcode] += 1


def _emit_literals(data, start, len_, depth, bits, w):
    """EmitLiterals."""
    write_bits = w.write_bits
    for j in range(start, start + len_):
        lit = data[j]
        write_bits(depth[lit], bits[lit])


def store_meta_block_header(len_, is_uncompressed, w):
    """BrotliStoreMetaBlockHeader (compress_fragment*.c).
    REQUIRES: len <= 1 << 24."""
    nibbles = 6
    # ISLAST
    w.write_bits(1, 0)
    if len_ <= (1 << 16):
        nibbles = 4
    elif len_ <= (1 << 20):
        nibbles = 5
    w.write_bits(2, nibbles - 4)
    w.write_bits(nibbles * 4, len_ - 1)
    # ISUNCOMPRESSED
    w.write_bits(1, 1 if is_uncompressed else 0)


def _should_merge_block(s, data, start, len_, depths):
    """ShouldMergeBlock."""
    histo = [0] * 256
    s.histogram = histo
    kSampleRate = 43
    for i in range(start, start + len_, kSampleRate):
        histo[data[i]] += 1
    total = (len_ + kSampleRate - 1) // kSampleRate
    r = (fast_log2(total) + 0.5) * float(total) + 200
    for i in range(256):
        r -= float(histo[i]) * (depths[i] + fast_log2(histo[i]))
    return r >= 0.0


# Acceptable loss for uncompressible speedup is 2%
MIN_RATIO = 980


def _should_use_uncompressed_mode(metablock_start, next_emit, insertlen,
                                  literal_ratio):
    """ShouldUseUncompressedMode."""
    compressed = next_emit - metablock_start
    if compressed * 50 > insertlen:
        return False
    return literal_ratio > MIN_RATIO


def _emit_uncompressed_meta_block(data, begin, end, storage_ix_start, w):
    """EmitUncompressedMetaBlock (compress_fragment.c)."""
    len_ = end - begin
    w.rewind(storage_ix_start)
    store_meta_block_header(len_, True, w)
    w.jump_to_byte_boundary()
    w.append_bytes(data[begin:end])


def _update_table(data, ip, base_ip, table, shift):
    """Updates "table" with the hashes of some positions within the last copy
    and returns the next candidate (shared tail of the two match loops)."""
    input_bytes = load64le(data, ip - 3)[0]
    prev_hash = _hash_bytes_at_offset(input_bytes, 0, shift)
    cur_hash = _hash_bytes_at_offset(input_bytes, 3, shift)
    table[prev_hash] = ip - base_ip - 3
    prev_hash = _hash_bytes_at_offset(input_bytes, 1, shift)
    table[prev_hash] = ip - base_ip - 2
    prev_hash = _hash_bytes_at_offset(input_bytes, 2, shift)
    table[prev_hash] = ip - base_ip - 1
    candidate = base_ip + table[cur_hash]
    table[cur_hash] = ip - base_ip
    return candidate


def _compress_fragment_fast_impl(s, data, input_, input_size, is_last, table,
                                 table_bits, w):
    """BrotliCompressFragmentFastImpl."""
    cmd_depth = s.cmd_depth
    cmd_bits = s.cmd_bits
    lit_depth = s.lit_depth
    lit_bits = s.lit_bits

    # "next_emit" is a pointer to the first byte that is not covered by a
    # previous copy. Bytes between "next_emit" and the start of the next copy
    # or the end of the input will be emitted as literal bytes.
    next_emit = input_
    # Save the start of the first block for position and distance
    # computations.
    base_ip = input_

    kFirstBlockSize = 3 << 15
    kMergeBlockSize = 1 << 16
    kInputMarginBytes = 16  # BROTLI_WINDOW_GAP
    kMinMatchLen = 5

    metablock_start = input_
    block_size = min(input_size, kFirstBlockSize)
    total_block_size = block_size
    # Save the bit position of the MLEN field of the meta-block header, so
    # that we can update it later if we decide to extend this meta-block.
    mlen_storage_ix = w.storage_ix + 3

    shift = 64 - table_bits

    store_meta_block_header(block_size, False, w)
    # No block splits, no contexts.
    w.write_bits(13, 0)

    literal_ratio = _build_and_store_literal_prefix_code(
        s, data, input_, block_size, lit_depth, lit_bits, w)

    # Store the pre-compressed command and distance prefix codes.
    i = 0
    while i + 7 < s.cmd_code_numbits:
        w.write_bits(8, s.cmd_code[i >> 3])
        i += 8
    w.write_bits(s.cmd_code_numbits & 7,
                 s.cmd_code[s.cmd_code_numbits >> 3]
                 if (s.cmd_code_numbits >> 3) < len(s.cmd_code) else 0)

    while True:  # emit_commands:
        # Initialize the command and distance histograms. We will gather
        # statistics of command and distance codes during the processing
        # of this block and use it to update the command and distance
        # prefix codes for the next block.
        cmd_histo = list(kCmdHistoSeed)
        s.cmd_histo = cmd_histo

        # "ip" is the input pointer.
        ip = input_
        last_distance = -1
        ip_end = input_ + block_size
        goto_next_block = False

        if block_size >= kInputMarginBytes:
            # For the last block, we need to keep a 16 bytes margin so that we
            # can be sure that all distances are at most window size - 16.
            # For all other blocks, we only need to keep a margin of 5 bytes
            # so that we don't go over the block size with a copy.
            len_limit = min(block_size - kMinMatchLen,
                            input_size - kInputMarginBytes)
            ip_limit = input_ + len_limit

            ip += 1
            next_hash = _hash(data, ip, shift)
            while True:
                # Step 1: Scan forward in the input looking for a 5-byte-long
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
                        next_hash = _hash(data, next_ip, shift)
                        candidate = ip - last_distance
                        if _is_match(data, ip, candidate):
                            if candidate < ip:
                                table[hash_] = ip - base_ip
                                break
                        candidate = base_ip + table[hash_]
                        table[hash_] = ip - base_ip
                        if _is_match(data, ip, candidate):
                            break
                    if emit_remainder:
                        break
                    # Check copy distance. If candidate is not feasible,
                    # continue search.
                    if ip - candidate > MAX_DISTANCE:
                        continue  # goto trawl
                    break
                if emit_remainder:
                    break

                # Step 2: Emit the found match together with the literal bytes
                # from "next_emit" to the bit stream, and then see if we can
                # find a next match immediately afterwards. Repeat until we
                # find no match for the input without emitting some literal
                # bytes.
                base = ip
                matched = 5 + find_match_length_with_limit(
                    data, candidate + 5, data, ip + 5, ip_end - ip - 5)
                distance = base - candidate  # > 0
                insert = base - next_emit
                ip += matched
                if insert < 6210:
                    _emit_insert_len(insert, cmd_depth, cmd_bits, cmd_histo, w)
                elif _should_use_uncompressed_mode(metablock_start, next_emit,
                                                   insert, literal_ratio):
                    _emit_uncompressed_meta_block(data, metablock_start, base,
                                                  mlen_storage_ix - 3, w)
                    input_size -= base - input_
                    input_ = base
                    next_emit = input_
                    goto_next_block = True
                    break
                else:
                    _emit_long_insert_len(insert, cmd_depth, cmd_bits,
                                          cmd_histo, w)
                _emit_literals(data, next_emit, insert, lit_depth, lit_bits, w)
                if distance == last_distance:
                    w.write_bits(cmd_depth[64], cmd_bits[64])
                    cmd_histo[64] += 1
                else:
                    _emit_distance(distance, cmd_depth, cmd_bits, cmd_histo, w)
                    last_distance = distance
                _emit_copy_len_last_distance(matched, cmd_depth, cmd_bits,
                                             cmd_histo, w)

                next_emit = ip
                if ip >= ip_limit:
                    break  # goto emit_remainder
                # We could immediately start working at ip now, but to improve
                # compression we first update "table" with the hashes of some
                # positions within the last copy.
                candidate = _update_table(data, ip, base_ip, table, shift)

                reached_limit = False
                while _is_match(data, ip, candidate):
                    # We have a 5-byte match at ip, and no need to emit any
                    # literal bytes prior to ip.
                    base = ip
                    matched = 5 + find_match_length_with_limit(
                        data, candidate + 5, data, ip + 5, ip_end - ip - 5)
                    if ip - candidate > MAX_DISTANCE:
                        break
                    ip += matched
                    last_distance = base - candidate  # > 0
                    _emit_copy_len(matched, cmd_depth, cmd_bits, cmd_histo, w)
                    _emit_distance(last_distance, cmd_depth, cmd_bits,
                                   cmd_histo, w)
                    next_emit = ip
                    if ip >= ip_limit:
                        reached_limit = True
                        break
                    candidate = _update_table(data, ip, base_ip, table, shift)
                if reached_limit:
                    break  # goto emit_remainder

                ip += 1
                next_hash = _hash(data, ip, shift)

        if not goto_next_block:
            # emit_remainder:
            input_ += block_size
            input_size -= block_size
            block_size = min(input_size, kMergeBlockSize)

            # Decide if we want to continue this meta-block instead of
            # emitting the last insert-only command.
            if (input_size > 0 and
                    total_block_size + block_size <= (1 << 20) and
                    _should_merge_block(s, data, input_, block_size,
                                        lit_depth)):
                # Update the size of the current meta-block and continue
                # emitting commands. We can do this because the current size
                # and the new size both have 5 nibbles.
                total_block_size += block_size
                w.update_bits(20, total_block_size - 1, mlen_storage_ix)
                continue  # goto emit_commands

            # Emit the remaining bytes as literals.
            if next_emit < ip_end:
                insert = ip_end - next_emit
                if insert < 6210:
                    _emit_insert_len(insert, cmd_depth, cmd_bits, cmd_histo, w)
                    _emit_literals(data, next_emit, insert, lit_depth,
                                   lit_bits, w)
                elif _should_use_uncompressed_mode(metablock_start, next_emit,
                                                   insert, literal_ratio):
                    _emit_uncompressed_meta_block(data, metablock_start, ip_end,
                                                  mlen_storage_ix - 3, w)
                else:
                    _emit_long_insert_len(insert, cmd_depth, cmd_bits,
                                          cmd_histo, w)
                    _emit_literals(data, next_emit, insert, lit_depth,
                                   lit_bits, w)
            next_emit = ip_end

        # next_block:
        # If we have more data, write a new meta-block header and prefix codes
        # and then continue emitting commands.
        if input_size > 0:
            metablock_start = input_
            block_size = min(input_size, kFirstBlockSize)
            total_block_size = block_size
            # Save the bit position of the MLEN field of the meta-block
            # header, so that we can update it later if we decide to extend
            # this meta-block.
            mlen_storage_ix = w.storage_ix + 3
            store_meta_block_header(block_size, False, w)
            # No block splits, no contexts.
            w.write_bits(13, 0)
            literal_ratio = _build_and_store_literal_prefix_code(
                s, data, input_, block_size, lit_depth, lit_bits, w)
            _build_and_store_command_prefix_code(s, w)
            continue  # goto emit_commands
        break

    if not is_last:
        # If this is not the last block, update the command and distance
        # prefix codes for the next block and store the compressed forms.
        cw = BitWriter()
        _build_and_store_command_prefix_code(s, cw)
        s.cmd_code_numbits = cw.storage_ix
        out, last_byte, last_bits = cw.get_output()
        s.cmd_code = out + bytes([last_byte])


def brotli_compress_fragment_fast(s, data, input_, input_size, is_last, table,
                                  table_size, w):
    """BrotliCompressFragmentFast: compresses data[input_:input_+input_size]
    as one or more complete meta-blocks.  If is_last, emits an additional
    empty last meta-block.  ``table`` (a list of table_size zeros) is used as
    the hash table; table_size is an odd (9, 11, 13, 15) power of two."""
    initial_storage_ix = w.storage_ix
    table_bits = table_size.bit_length() - 1

    if input_size == 0:
        w.write_bits(1, 1)  # islast
        w.write_bits(1, 1)  # isempty
        w.jump_to_byte_boundary()
        return

    if table_bits not in (9, 11, 13, 15):
        raise AssertionError('invalid table size')
    _compress_fragment_fast_impl(s, data, input_, input_size, is_last, table,
                                 table_bits, w)

    # If output is larger than single uncompressed block, rewrite it.
    if w.storage_ix - initial_storage_ix > 31 + (input_size << 3):
        _emit_uncompressed_meta_block(data, input_, input_ + input_size,
                                      initial_storage_ix, w)

    if is_last:
        w.write_bits(1, 1)  # islast
        w.write_bits(1, 1)  # isempty
        w.jump_to_byte_boundary()
