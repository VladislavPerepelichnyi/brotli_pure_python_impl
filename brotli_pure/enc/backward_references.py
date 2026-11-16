"""Function to find backward reference copies
(port of c/enc/backward_references.h / .c and backward_references_inc.h).

``brotli_create_backward_references`` appends ``Command`` objects to the
``commands`` list and returns ``(last_insert_len, num_literals)`` (the C
in/out scalar parameters; pass their current values in).  ``dist_cache`` is a
list of 16 ints, mutated in place.
"""

from ..common.context import CONTEXT_LOOKUP
from .command import Command, init_command
from .hash import HasherSearchResult, BROTLI_SCORE_BASE
from .params import brotli_max_backward_limit
from .quality import (literal_spree_length_for_sparse_search,
                      MIN_QUALITY_FOR_EXTENSIVE_REFERENCE_SEARCH)

BROTLI_NUM_DISTANCE_SHORT_CODES = 16


def compute_distance_code(distance, max_distance, dist_cache):
    """ComputeDistanceCode."""
    if distance <= max_distance:
        distance_plus_3 = distance + 3
        offset0 = distance_plus_3 - dist_cache[0]
        offset1 = distance_plus_3 - dist_cache[1]
        if distance == dist_cache[0]:
            return 0
        elif distance == dist_cache[1]:
            return 1
        elif 0 <= offset0 < 7:
            return (0x9750468 >> (4 * offset0)) & 0xF
        elif 0 <= offset1 < 7:
            return (0xFDB1ACE >> (4 * offset1)) & 0xF
        elif distance == dist_cache[2]:
            return 2
        elif distance == dist_cache[3]:
            return 3
    return distance + BROTLI_NUM_DISTANCE_SHORT_CODES - 1


def create_backward_references(num_bytes, position, ringbuffer,
                               ringbuffer_mask, literal_context_lut, params,
                               hasher, dist_cache, last_insert_len, commands,
                               num_literals):
    """EXPORT_FN(CreateBackwardReferences) (backward_references_inc.h),
    generic over the hasher object (``hasher.privat``)."""
    privat = hasher.privat
    find_longest_match = privat.find_longest_match
    store_range = privat.store_range
    store = privat.store
    # Set maximum distance, see section 9.1. of the spec.
    max_backward_limit = brotli_max_backward_limit(params.lgwin)
    position_offset = params.stream_offset
    hash_type_length = privat.hash_type_length()
    store_lookahead = privat.store_lookahead()

    insert_length = last_insert_len
    pos_end = position + num_bytes
    store_end = (position + num_bytes - store_lookahead + 1
                 if num_bytes >= store_lookahead else position)

    # For speed up heuristics for random data.
    random_heuristics_window_size = literal_spree_length_for_sparse_search(params)
    apply_random_heuristics = position + random_heuristics_window_size
    gap = params.dictionary.compound.total_size
    if params.dictionary.compound.num_chunks != 0:
        raise NotImplementedError('compound dictionaries are not supported')
    contextual = params.dictionary.contextual
    context_based = contextual.context_based
    dicts = contextual.dict
    dist = params.dist
    max_distance_param = dist.max_distance
    quality_lt_extensive = params.quality < MIN_QUALITY_FOR_EXTENSIVE_REFERENCE_SEARCH

    # Minimum score to accept a backward reference.
    kMinScore = BROTLI_SCORE_BASE + 100
    cost_diff_lazy = 175

    privat.prepare_distance_cache(dist_cache)
    lut = CONTEXT_LOOKUP
    lut_off = literal_context_lut

    sr = HasherSearchResult()
    sr2 = HasherSearchResult()

    while position + hash_type_length < pos_end:
        max_length = pos_end - position
        max_distance = position if position < max_backward_limit else max_backward_limit
        dictionary_start = min(position + position_offset, max_backward_limit)
        dict_id = 0
        p1 = 0
        p2 = 0
        if context_based:
            p1 = ringbuffer[(position - 1) & ringbuffer_mask] if position >= 1 else 0
            p2 = ringbuffer[(position - 2) & ringbuffer_mask] if position >= 2 else 0
            dict_id = contextual.context_map[
                lut[lut_off + p1] | lut[lut_off + 256 + p2]]
        sr.len = 0
        sr.len_code_delta = 0
        sr.distance = 0
        sr.score = kMinScore
        find_longest_match(dicts[dict_id], ringbuffer, ringbuffer_mask,
                           dist_cache, position, max_length, max_distance,
                           dictionary_start + gap, max_distance_param, sr)
        if sr.score > kMinScore:
            # Found a match. Let's look for something even better ahead.
            delayed_backward_references_in_row = 0
            max_length -= 1
            while True:
                if quality_lt_extensive:
                    sr2.len = min(sr.len - 1, max_length)
                else:
                    sr2.len = 0
                sr2.len_code_delta = 0
                sr2.distance = 0
                sr2.score = kMinScore
                max_distance = min(position + 1, max_backward_limit)
                dictionary_start = min(position + 1 + position_offset,
                                       max_backward_limit)
                if context_based:
                    p2 = p1
                    p1 = ringbuffer[position & ringbuffer_mask]
                    dict_id = contextual.context_map[
                        lut[lut_off + p1] | lut[lut_off + 256 + p2]]
                find_longest_match(dicts[dict_id], ringbuffer, ringbuffer_mask,
                                   dist_cache, position + 1, max_length,
                                   max_distance, dictionary_start + gap,
                                   max_distance_param, sr2)
                if sr2.score >= sr.score + cost_diff_lazy:
                    # Ok, let's just write one byte for now and start a match
                    # from the next byte.
                    position += 1
                    insert_length += 1
                    sr, sr2 = sr2, sr
                    delayed_backward_references_in_row += 1
                    if (delayed_backward_references_in_row < 4 and
                            position + hash_type_length < pos_end):
                        max_length -= 1
                        continue
                break
            apply_random_heuristics = (position + 2 * sr.len +
                                       random_heuristics_window_size)
            dictionary_start = min(position + position_offset,
                                   max_backward_limit)
            # The first 16 codes are special short-codes,
            # and the minimum offset is 1.
            distance_code = compute_distance_code(
                sr.distance, dictionary_start + gap, dist_cache)
            if sr.distance <= dictionary_start + gap and distance_code > 0:
                dist_cache[3] = dist_cache[2]
                dist_cache[2] = dist_cache[1]
                dist_cache[1] = dist_cache[0]
                dist_cache[0] = sr.distance
                privat.prepare_distance_cache(dist_cache)
            commands.append(init_command(Command(), dist, insert_length,
                                         sr.len, sr.len_code_delta,
                                         distance_code))
            num_literals += insert_length
            insert_length = 0
            # Put the hash keys into the table, if there are enough bytes left.
            # Depending on the hasher implementation, it can push all positions
            # in the given range or only a subset of them.
            # Avoid hash poisoning with RLE data.
            range_start = position + 2
            range_end = min(position + sr.len, store_end)
            if sr.distance < (sr.len >> 2):
                range_start = min(range_end, max(
                    range_start, position + sr.len - (sr.distance << 2)))
            store_range(ringbuffer, ringbuffer_mask, range_start, range_end)
            position += sr.len
        else:
            insert_length += 1
            position += 1
            # If we have not seen matches for a long time, we can skip some
            # match lookups. Unsuccessful match lookups are very very expensive
            # and this kind of a heuristic speeds up compression quite a lot.
            if position > apply_random_heuristics:
                # Going through uncompressible data, jump.
                if position > apply_random_heuristics + 4 * random_heuristics_window_size:
                    # It is quite a long time since we saw a copy, so we assume
                    # that this data is not compressible, and store hashes less
                    # often. Hashes of non compressible data are less likely to
                    # turn out to be useful in the future, too, so we store less
                    # of them to not to flood out the hash table of good
                    # compressible data.
                    kMargin = max(store_lookahead - 1, 4)
                    pos_jump = min(position + 16, pos_end - kMargin)
                    while position < pos_jump:
                        store(ringbuffer, ringbuffer_mask, position)
                        insert_length += 4
                        position += 4
                else:
                    kMargin = max(store_lookahead - 1, 2)
                    pos_jump = min(position + 8, pos_end - kMargin)
                    while position < pos_jump:
                        store(ringbuffer, ringbuffer_mask, position)
                        insert_length += 2
                        position += 2
    insert_length += pos_end - position
    return insert_length, num_literals


def brotli_create_backward_references(num_bytes, position, ringbuffer,
                                      ringbuffer_mask, literal_context_lut,
                                      params, hasher, dist_cache,
                                      last_insert_len, commands, num_literals):
    """BrotliCreateBackwardReferences: dispatches on params.hasher.type
    (all ported generic hashers share the same template code)."""
    return create_backward_references(
        num_bytes, position, ringbuffer, ringbuffer_mask, literal_context_lut,
        params, hasher, dist_cache, last_insert_len, commands, num_literals)
