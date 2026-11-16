"""H2 / H3 / H4 / H54: "longest match quickly" hashers
(port of c/enc/hash_longest_match_quickly_inc.h, instantiated in hash.h;
used for qualities 2-4).

A (forgetful) hash table to the data seen by the compressor, to help create
backward references to previous data.  This is a hash map of fixed size
(BUCKET_SIZE); each position is stored in one of BUCKET_SWEEP neighbouring
slots.
"""

from .hash_base import kHashMul64, load64le
from .find_match_length import find_match_length_with_limit
from .hash import (search_in_static_dictionary, BROTLI_SCORE_BASE,
                   BROTLI_LITERAL_BYTE_SCORE, BROTLI_DISTANCE_BIT_PENALTY)

_M64 = 0xFFFFFFFFFFFFFFFF
_LAST_DISTANCE_BONUS = BROTLI_SCORE_BASE + 15


class HashLongestMatchQuickly(object):
    """Template; subclasses set BUCKET_BITS, BUCKET_SWEEP_BITS, HASH_LEN,
    USE_DICTIONARY."""

    BUCKET_BITS = 16
    BUCKET_SWEEP_BITS = 0
    HASH_LEN = 5
    USE_DICTIONARY = True

    def __init__(self, common, params, one_shot=False, input_size=0):
        # FN(Initialize) + FN(HashMemAllocInBytes)
        self.common = common
        self.bucket_size = 1 << self.BUCKET_BITS
        self.bucket_mask = self.bucket_size - 1
        self.bucket_sweep = 1 << self.BUCKET_SWEEP_BITS
        self.bucket_sweep_mask = (self.bucket_sweep - 1) << 3
        self.hash_shift_left = 64 - 8 * self.HASH_LEN
        self.buckets = [0] * self.bucket_size

    @staticmethod
    def hash_type_length():
        return 8

    @staticmethod
    def store_lookahead():
        return 8

    def hash_bytes(self, data, off):
        """FN(HashBytes)."""
        h = (((load64le(data, off)[0] << self.hash_shift_left) & _M64) *
             kHashMul64) & _M64
        # The higher bits contain more mixture from the multiplication,
        # so we take our results from there.
        return h >> (64 - self.BUCKET_BITS)

    def prepare(self, one_shot, input_size, data):
        """FN(Prepare)."""
        buckets = self.buckets
        # Partial preparation is 100 times slower (per socket).
        partial_prepare_threshold = self.bucket_size >> 5
        if one_shot and input_size <= partial_prepare_threshold:
            for i in range(input_size):
                key = self.hash_bytes(data, i)
                if self.bucket_sweep == 1:
                    buckets[key] = 0
                else:
                    for j in range(self.bucket_sweep):
                        buckets[(key + (j << 3)) & self.bucket_mask] = 0
        else:
            # It is not strictly necessary to fill this buffer here, but
            # not filling will make the results of the compression stochastic
            # (but correct). This is because random data would cause the
            # system to find accidentally good backward references here and
            # there.
            self.buckets = [0] * self.bucket_size

    def store(self, data, mask, ix):
        """FN(Store): computes a hash from 5 bytes at data[ix & mask] and
        stores the value somewhere within [ix .. ix+3]."""
        key = self.hash_bytes(data, ix & mask)
        if self.bucket_sweep == 1:
            self.buckets[key] = ix & 0xFFFFFFFF
        else:
            # Wiggle the value with the bucket sweep range.
            off = ix & self.bucket_sweep_mask
            self.buckets[(key + off) & self.bucket_mask] = ix & 0xFFFFFFFF

    def store_range(self, data, mask, ix_start, ix_end):
        """FN(StoreRange)."""
        store = self.store
        for i in range(ix_start, ix_end):
            store(data, mask, i)

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
        """FN(PrepareDistanceCache): no-op."""

    def find_longest_match(self, dictionary, data, ring_buffer_mask,
                           distance_cache, cur_ix, max_length, max_backward,
                           dictionary_distance, max_distance, out):
        """FN(FindLongestMatch)."""
        buckets = self.buckets
        best_len_in = out.len
        cur_ix_masked = cur_ix & ring_buffer_mask
        # TODO: compare 4 bytes at once (and set the minimum best len to 4)
        compare_char = data[cur_ix_masked + best_len_in]
        key = self.hash_bytes(data, cur_ix_masked)
        min_score = out.score
        best_score = out.score
        best_len = best_len_in
        cached_backward = distance_cache[0]
        prev_ix = cur_ix - cached_backward
        sweep1 = self.bucket_sweep == 1

        out.len_code_delta = 0
        if 0 <= prev_ix < cur_ix:
            prev_ix &= ring_buffer_mask
            if compare_char == data[prev_ix + best_len]:
                len_ = find_match_length_with_limit(data, prev_ix, data,
                                                    cur_ix_masked, max_length)
                if len_ >= 4:
                    score = BROTLI_LITERAL_BYTE_SCORE * len_ + _LAST_DISTANCE_BONUS
                    if best_score < score:
                        out.len = len_
                        out.distance = cached_backward
                        out.score = score
                        if sweep1:
                            buckets[key] = cur_ix & 0xFFFFFFFF
                            return
                        else:
                            best_len = len_
                            best_score = score
                            compare_char = data[cur_ix_masked + len_]
        if sweep1:
            # Only one to look for, don't bother to prepare for a loop.
            prev_ix = buckets[key]
            buckets[key] = cur_ix & 0xFFFFFFFF
            backward = cur_ix - prev_ix
            prev_ix &= ring_buffer_mask
            if compare_char != data[prev_ix + best_len_in]:
                return
            if backward <= 0 or backward > max_backward:
                return
            len_ = find_match_length_with_limit(data, prev_ix, data,
                                                cur_ix_masked, max_length)
            if len_ >= 4:
                score = (BROTLI_SCORE_BASE + BROTLI_LITERAL_BYTE_SCORE * len_ -
                         BROTLI_DISTANCE_BIT_PENALTY * (backward.bit_length() - 1))
                if best_score < score:
                    out.len = len_
                    out.distance = backward
                    out.score = score
                    return
        else:
            bucket_mask = self.bucket_mask
            keys = [(key + (i << 3)) & bucket_mask
                    for i in range(self.bucket_sweep)]
            key_out = keys[(cur_ix & self.bucket_sweep_mask) >> 3]
            for i in range(self.bucket_sweep):
                prev_ix = buckets[keys[i]]
                backward = cur_ix - prev_ix
                prev_ix &= ring_buffer_mask
                if compare_char != data[prev_ix + best_len]:
                    continue
                if backward <= 0 or backward > max_backward:
                    continue
                len_ = find_match_length_with_limit(data, prev_ix, data,
                                                    cur_ix_masked, max_length)
                if len_ >= 4:
                    score = (BROTLI_SCORE_BASE + BROTLI_LITERAL_BYTE_SCORE * len_ -
                             BROTLI_DISTANCE_BIT_PENALTY *
                             (backward.bit_length() - 1))
                    if best_score < score:
                        best_len = len_
                        out.len = len_
                        compare_char = data[cur_ix_masked + len_]
                        best_score = score
                        out.score = score
                        out.distance = backward
        if self.USE_DICTIONARY and min_score == out.score:
            search_in_static_dictionary(
                dictionary, self.common, data, cur_ix_masked, max_length,
                dictionary_distance, max_distance, out, True)
        if not sweep1:
            buckets[key_out] = cur_ix & 0xFFFFFFFF


class H2(HashLongestMatchQuickly):
    BUCKET_BITS = 16
    BUCKET_SWEEP_BITS = 0
    HASH_LEN = 5
    USE_DICTIONARY = True


class H3(HashLongestMatchQuickly):
    BUCKET_BITS = 16
    BUCKET_SWEEP_BITS = 1
    HASH_LEN = 5
    USE_DICTIONARY = False


class H4(HashLongestMatchQuickly):
    BUCKET_BITS = 17
    BUCKET_SWEEP_BITS = 2
    HASH_LEN = 5
    USE_DICTIONARY = True


class H54(HashLongestMatchQuickly):
    BUCKET_BITS = 20
    BUCKET_SWEEP_BITS = 2
    HASH_LEN = 7
    USE_DICTIONARY = False
