"""H10 binary-tree hasher used by qualities 10 and 11.

Port of c/enc/hash_to_binary_tree_inc.h instantiated in c/enc/hash.h with
    HASHER() = H10, BUCKET_BITS = 17, MAX_TREE_SEARCH_DEPTH = 64,
    MAX_TREE_COMP_LENGTH = 128
plus the BackwardMatch helpers of hash.h.

A (forgetful) hash table where each hash bucket contains a binary tree of
sequences whose first 4 bytes share the same hash code.  Each sequence is
MAX_TREE_COMP_LENGTH long and is identified by its starting position in the
input data.  The binary tree is sorted by the lexicographic order of the
sequences, and it is also a max-heap with respect to the starting positions.

BackwardMatch arrays are represented as two parallel Python lists:
``distance`` values and ``length_and_code`` values (len << 5 | len_code).
"""

from array import array

from .static_dict import (BROTLI_MAX_STATIC_DICTIONARY_MATCH_LEN,
                          K_INVALID_MATCH,
                          brotli_find_all_static_dictionary_matches)

BUCKET_BITS = 17
BUCKET_SIZE = 1 << BUCKET_BITS
MAX_TREE_SEARCH_DEPTH = 64
MAX_TREE_COMP_LENGTH = 128
# MAX_NUM_MATCHES == 64 + MAX_TREE_SEARCH_DEPTH
MAX_NUM_MATCHES_H10 = 128

HQ_ZOPFLIFICATION_QUALITY = 11
BROTLI_WINDOW_GAP = 16

K_HASH_MUL32 = 0x1E35A7BD
_HASH_SHIFT = 32 - BUCKET_BITS

_INVALID_MATCHES = [K_INVALID_MATCH] * (BROTLI_MAX_STATIC_DICTIONARY_MATCH_LEN + 1)


# --- BackwardMatch helpers (c/enc/hash.h) -----------------------------------

def init_backward_match(dist, ln):
    """InitBackwardMatch: returns (distance, length_and_code)."""
    return dist, ln << 5


def init_dictionary_backward_match(dist, ln, len_code):
    """InitDictionaryBackwardMatch: returns (distance, length_and_code)."""
    return dist, (ln << 5) | (0 if ln == len_code else len_code)


def backward_match_length(length_and_code):
    """BackwardMatchLength."""
    return length_and_code >> 5


def backward_match_length_code(length_and_code):
    """BackwardMatchLengthCode."""
    code = length_and_code & 31
    return code if code else length_and_code >> 5


# --- FindMatchLengthWithLimit (c/enc/find_match_length.h) -------------------

def find_match_length_with_limit(s, i, j, limit):
    """FindMatchLengthWithLimit(&s[i], &s[j], limit) for two positions of the
    same buffer; returns the length of the common prefix (<= limit).

    Compares growing slices (galloping + bisection) instead of single bytes,
    which is much faster in CPython for long matches."""
    if limit < 8:
        n = 0
        while n < limit and s[i + n] == s[j + n]:
            n += 1
        return n
    if s[i:i + 8] != s[j:j + 8]:
        n = 0
        while s[i + n] == s[j + n]:
            n += 1
        return n
    n = 8
    step = 32
    while True:
        m = limit - n
        if m <= 0:
            return n
        if m > step:
            m = step
        if s[i + n:i + n + m] == s[j + n:j + n + m]:
            n += m
            if step < 65536:
                step <<= 1
        else:
            hi = n + m
            while hi - n > 16:
                mid = (n + hi) >> 1
                if s[i + n:i + mid] == s[j + n:j + mid]:
                    n = mid
                else:
                    hi = mid
            while s[i + n] == s[j + n]:
                n += 1
            return n


class H10(object):
    """HashToBinaryTree (H10)."""

    __slots__ = ("common", "window_mask_", "buckets_", "invalid_pos_",
                 "forest_")

    def __init__(self, common, params):
        # Initialize.  The C HasherSetup allocates buckets_ / forest_
        # (HashMemAllocInBytes) right before Initialize, using the
        # one_shot / input_size of the first Prepare call; since that
        # information is only passed to prepare() here, the allocation is
        # done there (once).
        self.common = common
        self.window_mask_ = (1 << params.lgwin) - 1
        self.invalid_pos_ = (0 - self.window_mask_) & 0xFFFFFFFF
        self.buckets_ = None
        self.forest_ = None

    def _hash_mem_alloc(self, one_shot, input_size):
        """HashMemAllocInBytes + allocation."""
        num_nodes = self.window_mask_ + 1
        if one_shot and input_size < num_nodes:
            num_nodes = input_size
        # uint32_t[2 * num_nodes]; contents are never read before written.
        self.forest_ = array("I", bytes(8 * num_nodes))

    @staticmethod
    def hash_type_length():
        """HashTypeLength."""
        return 4

    @staticmethod
    def store_lookahead():
        """StoreLookahead."""
        return MAX_TREE_COMP_LENGTH

    @staticmethod
    def hash_bytes(data, p):
        """HashBytes."""
        v = (data[p] | (data[p + 1] << 8) | (data[p + 2] << 16) |
             (data[p + 3] << 24))
        return ((v * K_HASH_MUL32) & 0xFFFFFFFF) >> _HASH_SHIFT

    def prepare(self, one_shot, input_size, data):
        """Prepare."""
        if self.forest_ is None:
            self._hash_mem_alloc(one_shot, input_size)
        self.buckets_ = [self.invalid_pos_] * BUCKET_SIZE

    def prepare_distance_cache(self, distance_cache):
        """PrepareDistanceCache: not used by H10 (no-op)."""
        pass

    def store_and_find_matches(self, data, cur_ix, ring_buffer_mask,
                               max_length, max_backward, best_len,
                               m_dist, m_lac):
        """StoreAndFindMatches.

        Stores the hash of the next 4 bytes and in a single tree-traversal,
        the hash bucket's binary tree is searched for matches and is re-rooted
        at the current position.  Found matches are appended to m_dist /
        m_lac (if m_dist is not None).  Returns the updated *best_len."""
        cur_ix_masked = cur_ix & ring_buffer_mask
        max_comp_len = max_length if max_length < MAX_TREE_COMP_LENGTH \
            else MAX_TREE_COMP_LENGTH
        should_reroot_tree = max_length >= MAX_TREE_COMP_LENGTH
        key = ((((data[cur_ix_masked] | (data[cur_ix_masked + 1] << 8) |
                  (data[cur_ix_masked + 2] << 16) |
                  (data[cur_ix_masked + 3] << 24)) * K_HASH_MUL32)
                & 0xFFFFFFFF) >> _HASH_SHIFT)
        buckets = self.buckets_
        forest = self.forest_
        window_mask = self.window_mask_
        prev_ix = buckets[key]
        # The forest index of the rightmost node of the left subtree of the new
        # root, updated as we traverse and re-root the tree of the hash bucket.
        node_left = 2 * (cur_ix & window_mask)
        # The forest index of the leftmost node of the right subtree of the new
        # root, updated as we traverse and re-root the tree of the hash bucket.
        node_right = node_left + 1
        # The match length of the rightmost node of the left subtree of the new
        # root, updated as we traverse and re-root the tree of the hash bucket.
        best_len_left = 0
        # The match length of the leftmost node of the right subtree of the new
        # root, updated as we traverse and re-root the tree of the hash bucket.
        best_len_right = 0
        if should_reroot_tree:
            buckets[key] = cur_ix & 0xFFFFFFFF
        fml = find_match_length_with_limit
        depth_remaining = MAX_TREE_SEARCH_DEPTH
        while True:
            # size_t arithmetic: a "negative" backward wraps to a huge value.
            backward = cur_ix - prev_ix
            if backward <= 0 or backward > max_backward or depth_remaining == 0:
                if should_reroot_tree:
                    forest[node_left] = self.invalid_pos_
                    forest[node_right] = self.invalid_pos_
                break
            prev_ix_masked = prev_ix & ring_buffer_mask
            cur_len = best_len_left if best_len_left < best_len_right \
                else best_len_right
            ln = cur_len + fml(data, cur_ix_masked + cur_len,
                               prev_ix_masked + cur_len, max_length - cur_len)
            if m_dist is not None and ln > best_len:
                best_len = ln
                m_dist.append(backward)
                m_lac.append(ln << 5)
            if ln >= max_comp_len:
                if should_reroot_tree:
                    pi = 2 * (prev_ix & window_mask)
                    forest[node_left] = forest[pi]
                    forest[node_right] = forest[pi + 1]
                break
            if data[cur_ix_masked + ln] > data[prev_ix_masked + ln]:
                best_len_left = ln
                if should_reroot_tree:
                    forest[node_left] = prev_ix
                node_left = 2 * (prev_ix & window_mask) + 1
                prev_ix = forest[node_left]
            else:
                best_len_right = ln
                if should_reroot_tree:
                    forest[node_right] = prev_ix
                node_right = 2 * (prev_ix & window_mask)
                prev_ix = forest[node_right]
            depth_remaining -= 1
        return best_len

    def find_all_matches(self, dictionary, data, ring_buffer_mask, cur_ix,
                         max_length, max_backward, dictionary_distance,
                         params, m_dist, m_lac):
        """FindAllMatches.

        Finds all backward matches of &data[cur_ix & ring_buffer_mask] up to
        the length of max_length and stores the position cur_ix in the hash
        table.  Appends the matches to m_dist / m_lac, sorted by strictly
        increasing length and (non-strictly) increasing distance.  Returns
        the number of matches found."""
        orig_num = len(m_dist)
        cur_ix_masked = cur_ix & ring_buffer_mask
        best_len = 1
        short_match_max_backward = \
            16 if params.quality != HQ_ZOPFLIFICATION_QUALITY else 64
        stop = cur_ix - short_match_max_backward
        if cur_ix < short_match_max_backward:
            stop = 0
        c0 = data[cur_ix_masked]
        c1 = data[cur_ix_masked + 1]
        i = cur_ix - 1
        while i > stop and best_len <= 2:
            backward = cur_ix - i
            if backward > max_backward:
                break
            prev_ix = i & ring_buffer_mask
            if data[prev_ix] != c0 or data[prev_ix + 1] != c1:
                i -= 1
                continue
            ln = find_match_length_with_limit(data, prev_ix, cur_ix_masked,
                                              max_length)
            if ln > best_len:
                best_len = ln
                m_dist.append(backward)
                m_lac.append(ln << 5)
            i -= 1
        if best_len < max_length:
            best_len = self.store_and_find_matches(
                data, cur_ix, ring_buffer_mask, max_length, max_backward,
                best_len, m_dist, m_lac)
        dict_matches = _INVALID_MATCHES[:]
        minlen = best_len + 1 if best_len + 1 > 4 else 4
        if brotli_find_all_static_dictionary_matches(
                dictionary, data, cur_ix_masked, minlen, max_length,
                dict_matches):
            maxlen = BROTLI_MAX_STATIC_DICTIONARY_MATCH_LEN \
                if BROTLI_MAX_STATIC_DICTIONARY_MATCH_LEN < max_length \
                else max_length
            max_distance = params.dist.max_distance
            for l in range(minlen, maxlen + 1):
                dict_id = dict_matches[l]
                if dict_id < K_INVALID_MATCH:
                    distance = dictionary_distance + (dict_id >> 5) + 1
                    if distance <= max_distance:
                        len_code = dict_id & 31
                        m_dist.append(distance)
                        m_lac.append((l << 5) |
                                     (0 if l == len_code else len_code))
        return len(m_dist) - orig_num

    def store(self, data, mask, ix):
        """Store: stores the hash of the next 4 bytes and re-roots the binary
        tree at the current sequence, without returning any matches.
        REQUIRES: ix + MAX_TREE_COMP_LENGTH <= end-of-current-block."""
        # Maximum distance is window size - 16, see section 9.1. of the spec.
        max_backward = self.window_mask_ - BROTLI_WINDOW_GAP + 1
        self.store_and_find_matches(data, ix, mask, MAX_TREE_COMP_LENGTH,
                                    max_backward, 0, None, None)

    def store_range(self, data, mask, ix_start, ix_end):
        """StoreRange."""
        i = ix_start
        j = ix_start
        if ix_start + 63 <= ix_end:
            i = ix_end - 63
        if ix_start + 512 <= i:
            while j < i:
                self.store(data, mask, j)
                j += 8
        while i < ix_end:
            self.store(data, mask, i)
            i += 1

    def stitch_to_previous_block(self, num_bytes, position, ringbuffer,
                                 ringbuffer_mask):
        """StitchToPreviousBlock."""
        if num_bytes >= 4 - 1 and position >= MAX_TREE_COMP_LENGTH:
            # Store the last `MAX_TREE_COMP_LENGTH - 1` positions in the
            # hasher. These could not be calculated before, since they require
            # knowledge of both the previous and the current block.
            i_start = position - MAX_TREE_COMP_LENGTH + 1
            i_end = min(position, i_start + num_bytes)
            for i in range(i_start, i_end):
                # Maximum distance is window size - 16, see section 9.1. of
                # the spec. Furthermore, we have to make sure that we don't
                # look further back from the start of the next block than the
                # window size, otherwise we could access already overwritten
                # areas of the ring-buffer.
                max_backward = self.window_mask_ - max(
                    BROTLI_WINDOW_GAP - 1, position - i)
                # We know that i + MAX_TREE_COMP_LENGTH <= position +
                # num_bytes, i.e. the end of the current block and that we
                # have at least MAX_TREE_COMP_LENGTH tail in the ring-buffer.
                self.store_and_find_matches(ringbuffer, i, ringbuffer_mask,
                                            MAX_TREE_COMP_LENGTH, max_backward,
                                            0, None, None)
