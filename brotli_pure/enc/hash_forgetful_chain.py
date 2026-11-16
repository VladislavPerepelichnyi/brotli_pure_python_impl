"""H40 / H41 / H42: "forgetful chain" hashers
(port of c/enc/hash_forgetful_chain_inc.h, instantiated in hash.h).

Hashes are stored in chains which are bucketed to groups. Group of chains
share a storage "bank". When more than "bank size" chain nodes are added,
oldest nodes are replaced; this way several chains may share a tail.
"""

from .hash_base import kHashMul32, load32le
from .find_match_length import find_match_length_with_limit
from .hash import (search_in_static_dictionary, prepare_distance_cache,
                   BROTLI_SCORE_BASE, BROTLI_LITERAL_BYTE_SCORE,
                   BROTLI_DISTANCE_BIT_PENALTY)

CAPPED_CHAINS = 0
_LAST_DISTANCE_BONUS = BROTLI_SCORE_BASE + 15


class HashForgetfulChain(object):
    """Template; subclasses set BUCKET_BITS, NUM_BANKS, BANK_BITS,
    NUM_LAST_DISTANCES_TO_CHECK."""

    BUCKET_BITS = 15
    NUM_BANKS = 1
    BANK_BITS = 16
    NUM_LAST_DISTANCES_TO_CHECK = 4

    def __init__(self, common, params, one_shot=False, input_size=0):
        # FN(Initialize)
        self.common = common
        self.max_hops = (7 if params.quality > 6 else 8) << (params.quality - 4)
        bucket_size = 1 << self.BUCKET_BITS
        bank_size = 1 << self.BANK_BITS
        self.bucket_size = bucket_size
        self.bank_size = bank_size
        # FN(HashMemAllocInBytes): addr (uint32), head (uint16),
        # tiny_hash (uint8[65536]), banks of (delta, next) uint16 slots.
        self.addr = [0xCCCCCCCC] * bucket_size
        self.head = [0] * bucket_size
        self.tiny_hash = bytearray(65536)
        self.slot_delta = [[0] * bank_size for _ in range(self.NUM_BANKS)]
        self.slot_next = [[0] * bank_size for _ in range(self.NUM_BANKS)]
        self.free_slot_idx = [0] * self.NUM_BANKS

    @staticmethod
    def hash_type_length():
        return 4

    @staticmethod
    def store_lookahead():
        return 4

    def hash_bytes(self, data, off):
        """FN(HashBytes)."""
        return (((load32le(data, off)[0] * kHashMul32) & 0xFFFFFFFF) >>
                (32 - self.BUCKET_BITS))

    def prepare(self, one_shot, input_size, data):
        """FN(Prepare)."""
        # Partial preparation is 100 times slower (per socket).
        partial_prepare_threshold = self.bucket_size >> 6
        if one_shot and input_size <= partial_prepare_threshold:
            addr = self.addr
            head = self.head
            shift = 32 - self.BUCKET_BITS
            for i in range(input_size):
                bucket = ((load32le(data, i)[0] * kHashMul32) & 0xFFFFFFFF) >> shift
                # See InitEmpty comment.
                addr[bucket] = 0xCCCCCCCC
                head[bucket] = 0xCCCC
        else:
            # Fill |addr| array with 0xCCCCCCCC value. Because of wrapping,
            # position processed by hasher never reaches 3GB + 64M; this makes
            # all new chains to be terminated after the first node.
            self.addr = [0xCCCCCCCC] * self.bucket_size
            self.head = [0] * self.bucket_size
        self.tiny_hash = bytearray(65536)
        self.free_slot_idx = [0] * self.NUM_BANKS

    def store(self, data, mask, ix):
        """FN(Store): prepends node to the chain of the hash of data[ix];
        also updates tiny_hash for the current position."""
        key = (((load32le(data, ix & mask)[0] * kHashMul32) & 0xFFFFFFFF) >>
               (32 - self.BUCKET_BITS))
        bank = key & (self.NUM_BANKS - 1)
        fsi = self.free_slot_idx
        idx = fsi[bank] & (self.bank_size - 1)
        fsi[bank] = (fsi[bank] + 1) & 0xFFFF
        addr = self.addr
        delta = ix - addr[key]
        self.tiny_hash[ix & 0xFFFF] = key & 0xFF
        if delta > 0xFFFF or delta < 0:
            delta = 0 if CAPPED_CHAINS else 0xFFFF
        self.slot_delta[bank][idx] = delta
        head = self.head
        self.slot_next[bank][idx] = head[key]
        addr[key] = ix & 0xFFFFFFFF
        head[key] = idx

    def store_range(self, data, mask, ix_start, ix_end):
        """FN(StoreRange)."""
        store = self.store
        for i in range(ix_start, ix_end):
            store(data, mask, i)

    def stitch_to_previous_block(self, num_bytes, position, ringbuffer,
                                 ring_buffer_mask):
        """FN(StitchToPreviousBlock)."""
        if num_bytes >= self.hash_type_length() - 1 and position >= 3:
            # Prepare the hashes for three last bytes of the last write.
            # These could not be calculated before, since they require
            # knowledge of both the previous and the current block.
            self.store(ringbuffer, ring_buffer_mask, position - 3)
            self.store(ringbuffer, ring_buffer_mask, position - 2)
            self.store(ringbuffer, ring_buffer_mask, position - 1)

    def prepare_distance_cache(self, distance_cache):
        """FN(PrepareDistanceCache)."""
        prepare_distance_cache(distance_cache, self.NUM_LAST_DISTANCES_TO_CHECK)

    def find_longest_match(self, dictionary, data, ring_buffer_mask,
                           distance_cache, cur_ix, max_length, max_backward,
                           dictionary_distance, max_distance, out):
        """FN(FindLongestMatch)."""
        addr = self.addr
        head = self.head
        tiny_hashes = self.tiny_hash
        cur_ix_masked = cur_ix & ring_buffer_mask
        # Don't accept a short copy from far away.
        min_score = out.score
        best_score = out.score
        best_len = out.len
        key = (((load32le(data, cur_ix_masked)[0] * kHashMul32) & 0xFFFFFFFF) >>
               (32 - self.BUCKET_BITS))
        tiny_hash = key & 0xFF
        out.len = 0
        out.len_code_delta = 0

        # Try last distance first.
        for i in range(self.NUM_LAST_DISTANCES_TO_CHECK):
            backward = distance_cache[i]
            prev_ix = cur_ix - backward
            # For distance code 0 we want to consider 2-byte matches.
            if i > 0 and tiny_hashes[prev_ix & 0xFFFF] != tiny_hash:
                continue
            if backward <= 0 or prev_ix < 0 or backward > max_backward:
                # prev_ix >= cur_ix (size_t wrap) || backward > max_backward
                continue
            prev_ix &= ring_buffer_mask
            len_ = find_match_length_with_limit(data, prev_ix, data,
                                                cur_ix_masked, max_length)
            if len_ >= 2:
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
        bank = key & (self.NUM_BANKS - 1)
        slot_delta = self.slot_delta[bank]
        slot_next = self.slot_next[bank]
        backward = 0
        hops = self.max_hops
        delta = cur_ix - addr[key]
        slot = head[key]
        if delta < 0:
            # size_t wrap of cur_ix - addr[key]: the first step already
            # exceeds max_backward.
            hops = 0
        while hops > 0:
            hops -= 1
            last = slot
            backward += delta
            if backward > max_backward or (CAPPED_CHAINS and not delta):
                break
            prev_ix = (cur_ix - backward) & ring_buffer_mask
            slot = slot_next[last]
            delta = slot_delta[last]
            a = cur_ix_masked + best_len
            b = prev_ix + best_len
            if (a > ring_buffer_mask or b > ring_buffer_mask or
                    # compare 4 bytes ending at best_len + 1
                    data[a] != data[b] or data[a - 1] != data[b - 1] or
                    data[a - 2] != data[b - 2] or data[a - 3] != data[b - 3]):
                continue
            len_ = find_match_length_with_limit(data, prev_ix, data,
                                                cur_ix_masked, max_length)
            if len_ >= 4:
                # Comparing for >= 3 does not change the semantics, but just
                # saves for a few unnecessary binary logarithms in backward
                # reference score, since we are not interested in such short
                # matches.
                score = (BROTLI_SCORE_BASE + BROTLI_LITERAL_BYTE_SCORE * len_ -
                         BROTLI_DISTANCE_BIT_PENALTY * (backward.bit_length() - 1))
                if best_score < score:
                    best_score = score
                    best_len = len_
                    out.len = best_len
                    out.distance = backward
                    out.score = best_score
        self.store(data, ring_buffer_mask, cur_ix)
        if out.score == min_score:
            search_in_static_dictionary(
                dictionary, self.common, data, cur_ix_masked, max_length,
                dictionary_distance, max_distance, out, False)


class H40(HashForgetfulChain):
    BUCKET_BITS = 15
    NUM_BANKS = 1
    BANK_BITS = 16
    NUM_LAST_DISTANCES_TO_CHECK = 4


class H41(HashForgetfulChain):
    BUCKET_BITS = 15
    NUM_BANKS = 1
    BANK_BITS = 16
    NUM_LAST_DISTANCES_TO_CHECK = 10


class H42(HashForgetfulChain):
    BUCKET_BITS = 15
    NUM_BANKS = 512
    BANK_BITS = 9
    NUM_LAST_DISTANCES_TO_CHECK = 16
