"""H68: "longest match" hasher (64-bit hash of 5 bytes) with 8-bit tags
matched with SIMD (port of c/enc/hash_longest_match64_simd_inc.h, instantiated
as H68 in hash.h).

This is a hash map of fixed size (bucket_size) to a ring buffer of fixed size
(block_size).  The ring buffer contains the last block_size index positions of
the given hash key in the compressed data.  Each entry also stores an 8-bit
tag (low bits of the hash); GetMatchingTagMask (matching_tag_mask.h, SSE2
variant on x86-64) selects the candidate slots whose tag matches.  The slots
are visited from the newest to the oldest (TZCNT order of the rotated mask),
which is what the loop over ``j`` below reproduces.
"""

from .hash_base import kHashMul64, load32le, load64le
from .find_match_length import find_match_length_with_limit
from .hash import (search_in_static_dictionary, prepare_distance_cache,
                   BROTLI_SCORE_BASE, BROTLI_LITERAL_BYTE_SCORE,
                   BROTLI_DISTANCE_BIT_PENALTY)

TAG_HASH_BITS = 8
TAG_HASH_MASK = (1 << TAG_HASH_BITS) - 1

_LAST_DISTANCE_BONUS = BROTLI_SCORE_BASE + 15
_M64 = 0xFFFFFFFFFFFFFFFF
_SHIFT = 64 - 15 - TAG_HASH_BITS


class H68(object):
    """HashLongestMatch (64-bit SIMD variant); bucket_bits is always 15."""

    def __init__(self, common, params):
        # FN(Initialize)
        self.common = common
        # Hash multiplier tuned to match length.
        self.hash_mul = (kHashMul64 << (64 - 5 * 8)) & _M64
        self.bucket_size = 1 << common.params.bucket_bits
        self.block_size = 1 << common.params.block_bits
        self.block_mask = self.block_size - 1
        self.block_bits = common.params.block_bits
        self.num_last_distances_to_check = \
            common.params.num_last_distances_to_check
        # FN(HashMemAllocInBytes): num (uint16), tags (uint8), buckets (uint32)
        self.num = [65535] * self.bucket_size
        self.tags = bytearray(self.bucket_size * self.block_size)
        self.buckets = [0] * (self.bucket_size * self.block_size)

    @staticmethod
    def hash_type_length():
        return 8

    @staticmethod
    def store_lookahead():
        return 8

    def hash_bytes(self, data, off):
        """FN(HashBytes)."""
        return ((load64le(data, off)[0] * self.hash_mul) & _M64) >> _SHIFT

    def prepare(self, one_shot, input_size, data):
        """FN(Prepare)."""
        num = self.num
        # Partial preparation is 100 times slower (per socket).
        partial_prepare_threshold = self.bucket_size >> 6
        if one_shot and input_size <= partial_prepare_threshold:
            mul = self.hash_mul
            for i in range(input_size):
                h = ((load64le(data, i)[0] * mul) & _M64) >> _SHIFT
                num[h >> TAG_HASH_BITS] = 65535
        else:
            # Set all the bytes of num to 255, which makes each uint16_t 65535.
            self.num = [65535] * self.bucket_size

    def store(self, data, mask, ix):
        """FN(Store)."""
        h = ((load64le(data, ix & mask)[0] * self.hash_mul) & _M64) >> _SHIFT
        key = h >> TAG_HASH_BITS
        num = self.num
        nk = num[key]
        offset = (nk & self.block_mask) + (key << self.block_bits)
        num[key] = (nk - 1) & 0xFFFF
        self.buckets[offset] = ix & 0xFFFFFFFF
        self.tags[offset] = h & TAG_HASH_MASK

    def store_range(self, data, mask, ix_start, ix_end):
        """FN(StoreRange)."""
        mul = self.hash_mul
        num = self.num
        buckets = self.buckets
        tags = self.tags
        block_mask = self.block_mask
        block_bits = self.block_bits
        for ix in range(ix_start, ix_end):
            h = ((load64le(data, ix & mask)[0] * mul) & _M64) >> _SHIFT
            key = h >> TAG_HASH_BITS
            nk = num[key]
            offset = (nk & block_mask) + (key << block_bits)
            num[key] = (nk - 1) & 0xFFFF
            buckets[offset] = ix & 0xFFFFFFFF
            tags[offset] = h & TAG_HASH_MASK

    def stitch_to_previous_block(self, num_bytes, position, ringbuffer,
                                 ringbuffer_mask):
        """FN(StitchToPreviousBlock)."""
        if num_bytes >= self.hash_type_length() - 1 and position >= 3:
            # Prepare the hashes for three last bytes of the last write.
            # These could not be calculated before, since they require
            # knowledge of both the previous and the current block.
            self.store(ringbuffer, ringbuffer_mask, position - 3)
            self.store(ringbuffer, ringbuffer_mask, position - 2)
            self.store(ringbuffer, ringbuffer_mask, position - 1)

    def prepare_distance_cache(self, distance_cache):
        """FN(PrepareDistanceCache)."""
        prepare_distance_cache(distance_cache, self.num_last_distances_to_check)

    def find_longest_match(self, dictionary, data, ring_buffer_mask,
                           distance_cache, cur_ix, max_length, max_backward,
                           dictionary_distance, max_distance, out):
        """FN(FindLongestMatch): finds the longest backward match of
        data[cur_ix] (up to max_length, at most max_backward away), stores
        cur_ix in the table and updates ``out`` (HasherSearchResult) if a
        better match is found."""
        num = self.num
        buckets = self.buckets
        tags = self.tags
        block_mask = self.block_mask
        cur_ix_masked = cur_ix & ring_buffer_mask
        # Don't accept a short copy from far away.
        min_score = out.score
        best_score = out.score
        best_len = out.len
        h = ((load64le(data, cur_ix_masked)[0] * self.hash_mul) & _M64) >> _SHIFT
        key = h >> TAG_HASH_BITS
        base = key << self.block_bits
        out.len = 0
        out.len_code_delta = 0

        # Try last distance first.
        for i in range(self.num_last_distances_to_check):
            backward = distance_cache[i]
            prev_ix = cur_ix - backward
            if backward <= 0 or prev_ix < 0:
                # (size_t) wrap: prev_ix >= cur_ix, or backward > max_backward
                continue
            if backward > max_backward:
                continue
            prev_ix &= ring_buffer_mask
            if cur_ix_masked + best_len > ring_buffer_mask:
                break
            if (prev_ix + best_len > ring_buffer_mask or
                    data[cur_ix_masked + best_len] != data[prev_ix + best_len]):
                continue
            len_ = find_match_length_with_limit(data, prev_ix, data,
                                                cur_ix_masked, max_length)
            if len_ >= 3 or (len_ == 2 and i < 2):
                # Comparing for >= 2 does not change the semantics, but just
                # saves for a few unnecessary binary logarithms in backward
                # reference score, since we are not interested in such short
                # matches.
                score = BROTLI_LITERAL_BYTE_SCORE * len_ + _LAST_DISTANCE_BONUS
                if best_score < score:
                    if i != 0:
                        score -= 39 + ((0x1CA10 >> (i & 0xE)) & 0xE)
                    if best_score < score:
                        best_score = score
                        best_len = len_
                        out.len = best_len
                        out.distance = backward
                        out.score = best_score
        # we require matches of len >4, so increase best_len to 3, so we can
        # compare 4 bytes all the time.
        if best_len < 3:
            best_len = 3
        tag = h & TAG_HASH_MASK
        first4 = load32le(data, cur_ix_masked)[0]
        max_length_m4 = max_length - 4
        nk = num[key]
        head = (nk + 1) & block_mask
        # Mask off any matches from uninitialized tags.
        n = 65535 - nk
        if n > self.block_size:
            n = self.block_size
        for j in range(n):
            rb_index = base + ((head + j) & block_mask)
            if tags[rb_index] != tag:
                continue
            prev_ix = buckets[rb_index]
            backward = cur_ix - prev_ix
            if backward > max_backward or backward < 0:
                break
            prev_ix &= ring_buffer_mask
            if cur_ix_masked + best_len > ring_buffer_mask:
                break
            # compare 4 bytes ending at best_len + 1
            a = cur_ix_masked + best_len
            b = prev_ix + best_len
            if (b > ring_buffer_mask or data[a] != data[b] or
                    data[a - 1] != data[b - 1] or data[a - 2] != data[b - 2] or
                    data[a - 3] != data[b - 3]):
                continue
            if load32le(data, prev_ix)[0] != first4:
                continue
            len_ = find_match_length_with_limit(
                data, prev_ix + 4, data, cur_ix_masked + 4, max_length_m4) + 4
            score = (BROTLI_SCORE_BASE + BROTLI_LITERAL_BYTE_SCORE * len_ -
                     BROTLI_DISTANCE_BIT_PENALTY * (backward.bit_length() - 1))
            if best_score < score:
                best_score = score
                best_len = len_
                out.len = best_len
                out.distance = backward
                out.score = best_score
        slot = base + (nk & block_mask)
        buckets[slot] = cur_ix & 0xFFFFFFFF
        tags[slot] = tag
        num[key] = (nk - 1) & 0xFFFF
        if min_score == out.score:
            search_in_static_dictionary(
                dictionary, self.common, data, cur_ix_masked, max_length,
                dictionary_distance, max_distance, out, False)
