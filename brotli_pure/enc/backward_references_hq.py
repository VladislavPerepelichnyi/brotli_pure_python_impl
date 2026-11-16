"""Zopfli-style backward reference search used by qualities 10 and 11
(port of c/enc/backward_references_hq.c / backward_references_hq.h).

Floating point: the C code keeps the cost model, node costs and all the
arithmetic on them in ``float`` (FLT_EVAL_METHOD == 0, i.e. every float
operation is rounded to binary32).  Python floats are binary64; since the sum
/ difference of two binary32 values computed in binary64 and then rounded to
binary32 equals the directly rounded binary32 result, every C float
operation ``a op b`` is emulated as ``f32(a op b)``.  ``f32`` is implemented
by storing into / loading from a one-element ``array('f')``.

ZopfliNode arrays are represented as parallel lists ``length``,
``distance``, ``dcode_insert_length`` and ``u`` (the C union: holds the
float ``cost`` during the forward pass, then the int ``shortcut`` / ``next``).

BackwardMatch arrays are two parallel lists (distance, length_and_code).

Shared static / compound dictionaries (``params.dictionary``) are not
supported: only the built-in static dictionary is used, i.e.
``addon->num_chunks == 0`` and ``gap == 0``, and the contextual dictionary
is not context based.
"""

from array import array

from .command import init_command
from .fast_log import fast_log2
from .hash_to_binary_tree import MAX_NUM_MATCHES_H10  # noqa: F401 (doc)
from .literal_cost import brotli_estimate_bit_costs_for_literals

BROTLI_NUM_DISTANCE_SHORT_CODES = 16
BROTLI_NUM_LITERAL_SYMBOLS = 256
BROTLI_NUM_COMMAND_SYMBOLS = 704
BROTLI_WINDOW_GAP = 16
BROTLI_UINT32_MAX = 0xFFFFFFFF

# BrotliCalculateDistanceCodeLimit(BROTLI_MAX_ALLOWED_DISTANCE, 3, 120).
BROTLI_MAX_EFFECTIVE_DISTANCE_ALPHABET_SIZE = 544

# quality.h
MAX_ZOPFLI_LEN_QUALITY_10 = 150
MAX_ZOPFLI_LEN_QUALITY_11 = 325
BROTLI_LONG_COPY_QUICK_STEP = 16384

_F32 = array("f", [0.0])


def _f32(x):
    """Round a Python float to the nearest binary32 value."""
    _F32[0] = x
    return _F32[0]


# static const float kInfinity = 1.7e38f;  /* ~= 2 ^ 127 */
K_INFINITY = _f32(1.7e38)

K_DISTANCE_CACHE_INDEX = (0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
K_DISTANCE_CACHE_OFFSET = (0, 0, 0, 0, -1, 1, -2, 2, -3, 3, -1, 1, -2, 2, -3,
                           3)


def max_backward_limit(lgwin):
    """BROTLI_MAX_BACKWARD_LIMIT."""
    return (1 << lgwin) - BROTLI_WINDOW_GAP


def max_zopfli_len(params):
    """MaxZopfliLen (quality.h)."""
    return (MAX_ZOPFLI_LEN_QUALITY_10 if params.quality <= 10
            else MAX_ZOPFLI_LEN_QUALITY_11)


def max_zopfli_candidates(params):
    """MaxZopfliCandidates (quality.h)."""
    return 1 if params.quality <= 10 else 5


# --- command.h length-code helpers (local copies for the hot loops) --------

def _log2_floor_non_zero(n):
    return n.bit_length() - 1


def _get_insert_length_code(insertlen):
    """GetInsertLengthCode."""
    if insertlen < 6:
        return insertlen
    elif insertlen < 130:
        nbits = _log2_floor_non_zero(insertlen - 2) - 1
        return (nbits << 1) + ((insertlen - 2) >> nbits) + 2
    elif insertlen < 2114:
        return _log2_floor_non_zero(insertlen - 66) + 10
    elif insertlen < 6210:
        return 21
    elif insertlen < 22594:
        return 22
    else:
        return 23


def _get_copy_length_code(copylen):
    """GetCopyLengthCode."""
    if copylen < 10:
        return copylen - 2
    elif copylen < 134:
        nbits = _log2_floor_non_zero(copylen - 6) - 1
        return (nbits << 1) + ((copylen - 6) >> nbits) + 4
    elif copylen < 2118:
        return _log2_floor_non_zero(copylen - 70) + 12
    else:
        return 23


def _combine_length_codes(inscode, copycode, use_last_distance):
    """CombineLengthCodes."""
    bits64 = (copycode & 0x7) | ((inscode & 0x7) << 3)
    if use_last_distance and inscode < 8 and copycode < 16:
        return bits64 if copycode < 8 else (bits64 | 64)
    else:
        # Specification: 5 Encoding of ... (last table)
        # offset = 2 * index, where index is in range [0..8]
        offset = 2 * ((copycode >> 3) + 3 * (inscode >> 3))
        # All values in specification are K * 64,
        # where   K = [2, 3, 6, 4, 5, 8, 7, 9, 10],
        #     i + 1 = [1, 2, 3, 4, 5, 6, 7, 8,  9],
        # K - i - 1 = [1, 1, 3, 0, 0, 2, 0, 1,  2] = D.
        # All values in D require only 2 bits to encode.
        # Magic constant is shifted 6 bits left, to avoid final multiplication.
        offset = (offset << 5) + 0x40 + ((0x520D40 >> offset) & 0xC0)
        return offset | bits64


# kBrotliInsExtra / kBrotliCopyExtra (command.c)
K_INS_EXTRA = (0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10,
               12, 14, 24)
K_COPY_EXTRA = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8,
                9, 10, 24)

# Copy length code for lengths < 2118 (23 above).
_COPY_CODE_TABLE = tuple(_get_copy_length_code(l) if l >= 2 else 0
                         for l in range(2118))
# _CMD_CODE[use_last][inscode][copycode]
_CMD_CODE = tuple(tuple(tuple(_combine_length_codes(i, c, ul)
                              for c in range(24)) for i in range(24))
                  for ul in (False, True))


def _prefix_encode_copy_distance_code(distance_code, num_direct_codes,
                                      postfix_bits):
    """PrefixEncodeCopyDistance (prefix.h), returning only *code."""
    if distance_code < BROTLI_NUM_DISTANCE_SHORT_CODES + num_direct_codes:
        return distance_code
    dist = ((1 << (postfix_bits + 2)) +
            (distance_code - BROTLI_NUM_DISTANCE_SHORT_CODES -
             num_direct_codes))
    bucket = dist.bit_length() - 2
    postfix_mask = (1 << postfix_bits) - 1
    postfix = dist & postfix_mask
    prefix = (dist >> bucket) & 1
    nbits = bucket - postfix_bits
    return ((nbits << 10) |
            (BROTLI_NUM_DISTANCE_SHORT_CODES + num_direct_codes +
             ((2 * (nbits - 1) + prefix) << postfix_bits) + postfix))


# --- ZopfliNode ------------------------------------------------------------

class ZopfliNodes(object):
    """Array of ZopfliNode (struct-of-arrays)."""

    __slots__ = ("length", "distance", "dcode_insert_length", "u")

    def __init__(self, n):
        brotli_init_zopfli_nodes(self, n)


def brotli_init_zopfli_nodes(nodes, length):
    """BrotliInitZopfliNodes."""
    nodes.length = [1] * length
    nodes.distance = [0] * length
    nodes.dcode_insert_length = [0] * length
    nodes.u = [K_INFINITY] * length


def _zopfli_node_distance_code(nodes, i):
    """ZopfliNodeDistanceCode."""
    short_code = nodes.dcode_insert_length[i] >> 27
    if short_code == 0:
        return nodes.distance[i] + BROTLI_NUM_DISTANCE_SHORT_CODES - 1
    return short_code - 1


# --- ZopfliCostModel -------------------------------------------------------

class ZopfliCostModel(object):
    """Histogram based cost model for zopflification."""

    __slots__ = ("cost_cmd_", "cost_dist_", "distance_histogram_size",
                 "literal_costs_", "min_cost_cmd_", "num_bytes_",
                 "literal_histograms")

    def __init__(self, dist, num_bytes):
        # InitZopfliCostModel
        self.num_bytes_ = num_bytes
        self.literal_costs_ = [0.0] * (num_bytes + 2)
        self.cost_dist_ = [0.0] * dist.alphabet_size_limit
        self.distance_histogram_size = dist.alphabet_size_limit
        self.cost_cmd_ = [0.0] * BROTLI_NUM_COMMAND_SYMBOLS
        self.min_cost_cmd_ = 0.0
        self.literal_histograms = [0] * (3 * 256)


def _set_cost(histogram, histogram_size, literal_histogram, cost):
    """SetCost."""
    f = _F32
    s = 0
    for i in range(histogram_size):
        s += histogram[i]
    f[0] = fast_log2(s)
    log2sum = f[0]
    missing_symbol_sum = s
    if not literal_histogram:
        for i in range(histogram_size):
            if histogram[i] == 0:
                missing_symbol_sum += 1
    f[0] = fast_log2(missing_symbol_sum)
    f[0] = f[0] + 2
    missing_symbol_cost = f[0]
    for i in range(histogram_size):
        h = histogram[i]
        if h == 0:
            cost[i] = missing_symbol_cost
            continue
        # Shannon bits for this symbol.
        f[0] = fast_log2(h)
        f[0] = log2sum - f[0]
        c = f[0]
        # Cannot be coded with less than 1 bit
        if c < 1:
            c = 1.0
        cost[i] = c


def _accumulate_literal_costs(literal_costs, costs, num_bytes):
    """Kahan-style float prefix sum shared by both cost model setters:
         literal_carry += cost[i];
         literal_costs[i + 1] = literal_costs[i] + literal_carry;
         literal_carry -= literal_costs[i + 1] - literal_costs[i];"""
    f = _F32
    literal_carry = 0.0
    literal_costs[0] = 0.0
    prev = 0.0
    for i in range(num_bytes):
        f[0] = literal_carry + costs[i]
        literal_carry = f[0]
        f[0] = prev + literal_carry
        cur = f[0]
        literal_costs[i + 1] = cur
        f[0] = cur - prev
        f[0] = literal_carry - f[0]
        literal_carry = f[0]
        prev = cur


def zopfli_cost_model_set_from_commands(model, position, ringbuffer,
                                        ringbuffer_mask, commands, cmd_start,
                                        num_commands, last_insert_len):
    """ZopfliCostModelSetFromCommands (commands[cmd_start:cmd_start+n])."""
    pos = position - last_insert_len
    histogram_literal = [0] * BROTLI_NUM_LITERAL_SYMBOLS
    histogram_cmd = [0] * BROTLI_NUM_COMMAND_SYMBOLS
    histogram_dist = [0] * BROTLI_MAX_EFFECTIVE_DISTANCE_ALPHABET_SIZE
    cost_literal = [0.0] * BROTLI_NUM_LITERAL_SYMBOLS
    cost_cmd = model.cost_cmd_

    for i in range(cmd_start, cmd_start + num_commands):
        cmd = commands[i]
        inslength = cmd.insert_len
        copylength = cmd.copy_len & 0x1FFFFFF
        distcode = cmd.dist_prefix & 0x3FF
        cmdcode = cmd.cmd_prefix
        histogram_cmd[cmdcode] += 1
        if cmdcode >= 128:
            histogram_dist[distcode] += 1
        for j in range(inslength):
            histogram_literal[ringbuffer[(pos + j) & ringbuffer_mask]] += 1
        pos += inslength + copylength

    _set_cost(histogram_literal, BROTLI_NUM_LITERAL_SYMBOLS, True,
              cost_literal)
    _set_cost(histogram_cmd, BROTLI_NUM_COMMAND_SYMBOLS, False, cost_cmd)
    _set_cost(histogram_dist, model.distance_histogram_size, False,
              model.cost_dist_)

    min_cost_cmd = K_INFINITY
    for i in range(BROTLI_NUM_COMMAND_SYMBOLS):
        if cost_cmd[i] < min_cost_cmd:
            min_cost_cmd = cost_cmd[i]
    model.min_cost_cmd_ = min_cost_cmd

    num_bytes = model.num_bytes_
    costs = [cost_literal[ringbuffer[(position + i) & ringbuffer_mask]]
             for i in range(num_bytes)]
    _accumulate_literal_costs(model.literal_costs_, costs, num_bytes)


def zopfli_cost_model_set_from_literal_costs(model, position, ringbuffer,
                                             ringbuffer_mask):
    """ZopfliCostModelSetFromLiteralCosts."""
    num_bytes = model.num_bytes_
    costs = array("f", bytes(4 * num_bytes))
    brotli_estimate_bit_costs_for_literals(position, num_bytes,
                                           ringbuffer_mask, ringbuffer,
                                           model.literal_histograms, costs, 0)
    _accumulate_literal_costs(model.literal_costs_, costs, num_bytes)
    cost_cmd = model.cost_cmd_
    for i in range(BROTLI_NUM_COMMAND_SYMBOLS):
        cost_cmd[i] = _f32(fast_log2(11 + i))
    cost_dist = model.cost_dist_
    for i in range(model.distance_histogram_size):
        cost_dist[i] = _f32(fast_log2(20 + i))
    model.min_cost_cmd_ = _f32(fast_log2(11))


# --- StartPosQueue ---------------------------------------------------------
# PosData is a tuple (pos, distance_cache (4-tuple), costdiff, cost,
# backwards) where ``backwards`` caches the 16 candidate distances
# distance_cache[kDistanceCacheIndex[j]] + kDistanceCacheOffset[j] used by
# UpdateNodes (pure function of distance_cache).

class StartPosQueue(object):
    """Maintains the smallest 8 cost difference together with their
    positions."""

    __slots__ = ("q_", "idx_")

    def __init__(self):
        # InitStartPosQueue
        self.q_ = [None] * 8
        self.idx_ = 0

    def size(self):
        """StartPosQueueSize."""
        return self.idx_ if self.idx_ < 8 else 8

    def push(self, posdata):
        """StartPosQueuePush."""
        offset = ~self.idx_ & 7
        self.idx_ += 1
        ln = self.idx_ if self.idx_ < 8 else 8
        q = self.q_
        q[offset] = posdata
        # Restore the sorted order. In the list of |len| items at most
        # |len - 1| adjacent element comparisons / swaps are required.
        for _ in range(1, ln):
            a = offset & 7
            b = (offset + 1) & 7
            if q[a][2] > q[b][2]:
                q[a], q[b] = q[b], q[a]
            offset += 1

    def at(self, k):
        """StartPosQueueAt."""
        return self.q_[(k - self.idx_) & 7]


def _compute_minimum_copy_length(start_cost, u, num_bytes, pos):
    """ComputeMinimumCopyLength: returns the minimum possible copy length that
    can improve the cost of any future position."""
    # Compute the minimum possible cost of reaching any future position.
    min_cost = start_cost
    ln = 2
    next_len_bucket = 4
    next_len_offset = 10
    while pos + ln <= num_bytes and u[pos + ln] <= min_cost:
        # We already reached (pos + len) with no more cost than the minimum
        # possible cost of reaching anything from this pos, so there is no
        # point in looking for lengths <= len.
        ln += 1
        if ln == next_len_offset:
            # We reached the next copy length code bucket, so we add one more
            # extra bit to the minimum cost.
            min_cost = _f32(min_cost + 1.0)
            next_len_offset += next_len_bucket
            next_len_bucket *= 2
    return ln


def _compute_distance_shortcut(block_start, pos, max_backward_limit_, gap,
                               nodes):
    """ComputeDistanceShortcut.
    REQUIRES: nodes[pos].cost < kInfinity
    REQUIRES: nodes[0..pos] satisfies that "ZopfliNode array invariant"."""
    if pos == 0:
        return 0
    c_len = nodes.length[pos] & 0x1FFFFFF
    dil = nodes.dcode_insert_length[pos]
    dist = nodes.distance[pos]
    # Since |block_start + pos| is the end position of the command, the copy
    # part starts from |block_start + pos - c_len|. Distances that are greater
    # than this or greater than |max_backward_limit| + |gap| are static
    # dictionary references, and do not update the last distances.
    # Also distance code 0 (last distance) does not update the last distances.
    # (ZopfliNodeDistanceCode(&nodes[pos]) > 0 <=> short code != 1, since a
    # zero short code yields dist + 15 > 0.)
    if (dist + c_len <= block_start + pos + gap and
            dist <= max_backward_limit_ + gap and
            (dil >> 27) != 1):
        return pos
    return nodes.u[pos - c_len - (dil & 0x7FFFFFF)]


def _compute_distance_cache(pos, starting_dist_cache, nodes):
    """ComputeDistanceCache: returns the last four distances (as defined by
    Section 4. of the Spec) that would be used at (block_start + pos) if we
    used the shortest path of commands from block_start, computed from
    nodes[0..pos]."""
    dist_cache = []
    u = nodes.u
    n_dil = nodes.dcode_insert_length
    n_length = nodes.length
    n_distance = nodes.distance
    p = u[pos]
    idx = 0
    while idx < 4 and p > 0:
        dist_cache.append(n_distance[p])
        idx += 1
        # Because of prerequisite, p >= c_len + i_len >= 2.
        p = u[p - (n_length[p] & 0x1FFFFFF) - (n_dil[p] & 0x7FFFFFF)]
    if idx < 4:
        dist_cache.extend(starting_dist_cache[:4 - idx])
    return tuple(dist_cache)


def _evaluate_node(block_start, pos, max_backward_limit_, gap,
                   starting_dist_cache, model, queue, nodes):
    """EvaluateNode: maintains "ZopfliNode array invariant" and pushes node to
    the queue, if it is eligible."""
    u = nodes.u
    # Save cost, because ComputeDistanceCache invalidates it.
    node_cost = u[pos]
    u[pos] = _compute_distance_shortcut(block_start, pos, max_backward_limit_,
                                        gap, nodes)
    lc = model.literal_costs_
    f = _F32
    f[0] = lc[pos] - lc[0]
    lit = f[0]
    if node_cost <= lit:
        f[0] = node_cost - lit
        dc = _compute_distance_cache(pos, starting_dist_cache, nodes)
        d0 = dc[0]
        d1 = dc[1]
        backwards = (d0, d1, dc[2], dc[3], d0 - 1, d0 + 1, d0 - 2, d0 + 2,
                     d0 - 3, d0 + 3, d1 - 1, d1 + 1, d1 - 2, d1 + 2, d1 - 3,
                     d1 + 3)
        queue.push((pos, dc, f[0], node_cost, backwards))


# GetInsertLengthCode for insert lengths < 22594 (23 above).
_INS_CODE_TABLE = tuple(_get_insert_length_code(i) for i in range(22594))
# Last copy length with the same copy length code, for lengths < 2118.
_COPY_RUN_END = [0] * 2118
for _l in range(2117, 1, -1):
    _COPY_RUN_END[_l] = (_COPY_RUN_END[_l + 1]
                         if _l + 1 < 2118 and
                         _COPY_CODE_TABLE[_l + 1] == _COPY_CODE_TABLE[_l]
                         else (_l if _l + 1 < 2118 else 1 << 62))
_COPY_RUN_END = tuple(_COPY_RUN_END)
del _l
# PrefixEncodeCopyDistance symbol cache: (ndirect, npostfix) -> {code: sym}.
_DIST_SYMBOL_CACHE = {}


def _update_nodes(num_bytes, block_start, pos, ringbuffer, ringbuffer_mask,
                  params, max_backward_limit_, starting_dist_cache,
                  num_matches, m_dist, m_lac, match_off, model, queue, nodes):
    """UpdateNodes: returns longest copy length.

    Equivalent restructuring of the C loops (results are identical):
    * last-distance candidates: every rejection branch of the C code is a
      plain ``continue``; without compound dictionary (gap == 0) a candidate
      is examined iff 0 < backward <= min(dictionary_start + gap,
      max_distance) (non-positive ints become huge size_t values in C);
    * the cost of a copy only depends on its copy length code, so the length
      loops evaluate it once per run of lengths sharing a code;
    * the match length of a candidate distance does not depend on the queue
      entry, so it is cached per position."""
    stream_offset = params.stream_offset
    cur_ix = block_start + pos
    cur_ix_masked = cur_ix & ringbuffer_mask
    max_distance = cur_ix if cur_ix < max_backward_limit_ \
        else max_backward_limit_
    dictionary_start = cur_ix + stream_offset
    if dictionary_start > max_backward_limit_:
        dictionary_start = max_backward_limit_
    max_len = num_bytes - pos
    max_zopfli_len_ = (MAX_ZOPFLI_LEN_QUALITY_10 if params.quality <= 10
                       else MAX_ZOPFLI_LEN_QUALITY_11)
    max_iters = 1 if params.quality <= 10 else 5
    result = 0
    gap = 0  # params->dictionary.compound.total_size (no compound dict)

    _evaluate_node(block_start + stream_offset, pos, max_backward_limit_, gap,
                   starting_dist_cache, model, queue, nodes)

    f = _F32
    lc = model.literal_costs_
    cost_cmd = model.cost_cmd_
    cost_dist = model.cost_dist_
    u = nodes.u
    n_length = nodes.length
    n_distance = nodes.distance
    n_dil = nodes.dcode_insert_length
    copy_code_table = _COPY_CODE_TABLE
    copy_run_end = _COPY_RUN_END
    copy_extra = K_COPY_EXTRA
    rb = ringbuffer
    mask = ringbuffer_mask
    q = queue.q_
    qidx = queue.idx_

    posdata = q[-qidx & 7]
    f[0] = posdata[3] + model.min_cost_cmd_
    t = f[0]
    f[0] = lc[pos] - lc[posdata[0]]
    f[0] = t + f[0]
    min_cost = f[0]
    min_len = _compute_minimum_copy_length(min_cost, u, num_bytes, pos)
    f[0] = lc[pos] - lc[0]
    lit_cost_pos = f[0]

    # Candidate distances must satisfy 0 < backward <= cand_limit.
    cand_limit = dictionary_start + gap
    if max_distance < cand_limit:
        cand_limit = max_distance

    qsize = qidx if qidx < 8 else 8
    if qsize > max_iters:
        qsize = max_iters

    # Pre-filter the distinct last-distance candidates of all examined queue
    # entries.  best_len starts at bl0 = min_len - 1 for every entry and only
    # grows, and a candidate only has an effect if its match length exceeds
    # best_len (which implies the C "continuation" byte check passes) and
    # prev_ix + best_len <= mask.  Candidates failing at bl0 can never have an
    # effect; the match length of the others is cached (it does not depend on
    # the queue entry).
    match_len_cache = {}
    any_candidate = False
    bl0 = min_len - 1
    if bl0 < max_len and cur_ix_masked + bl0 <= mask:
        cont = rb[cur_ix_masked + bl0]
        if qsize == 1:
            cands = posdata[4]
        else:
            cands = set(posdata[4])
            for k in range(1, qsize):
                cands.update(q[(k - qidx) & 7][4])
        for backward in cands:
            if 0 < backward <= cand_limit:
                prev_ix = (cur_ix - backward) & mask
                if prev_ix + bl0 <= mask and rb[prev_ix + bl0] == cont:
                    ln = _find_match_length(rb, prev_ix, cur_ix_masked,
                                            max_len)
                    if ln > bl0:
                        match_len_cache[backward] = ln
                        any_candidate = True

    # Go over the command starting positions in order of increasing cost
    # difference.
    for k in range(qsize):
        posdata = q[(k - qidx) & 7]
        start = posdata[0]
        ins_bits = pos - start
        inscode = _INS_CODE_TABLE[ins_bits] if ins_bits < 22594 else 23
        f[0] = posdata[2] + K_INS_EXTRA[inscode]
        f[0] = f[0] + lit_cost_pos
        base_cost = f[0]
        cmd_codes = _CMD_CODE[0][inscode]

        # Look for last distance matches using the distance cache from this
        # starting position.
        best_len = bl0
        if any_candidate:
            get_len = match_len_cache.get
            j = -1
            for backward in posdata[4]:
                j += 1
                if best_len >= max_len or cur_ix_masked + best_len > mask:
                    break
                ln = get_len(backward)
                if ln is None or ln <= best_len:
                    continue
                prev_ix = (cur_ix - backward) & mask
                if prev_ix + best_len > mask:
                    continue
                f[0] = base_cost + cost_dist[j]
                dist_cost = f[0]
                codes = _CMD_CODE[1][inscode] if j == 0 else cmd_codes
                dil = ((j + 1) << 27) | ins_bits
                l = best_len + 1
                while l <= ln:
                    if l < 2118:
                        copycode = copy_code_table[l]
                        run_end = copy_run_end[l]
                        if run_end > ln:
                            run_end = ln
                    else:
                        copycode = 23
                        run_end = ln
                    cmdcode = codes[copycode]
                    f[0] = (base_cost if cmdcode < 128 else dist_cost) + \
                        copy_extra[copycode]
                    f[0] = f[0] + cost_cmd[cmdcode]
                    cost = f[0]
                    for nxt in range(pos + l, pos + run_end + 1):
                        if cost < u[nxt]:
                            # UpdateZopfliNode(nodes, pos, start, l, l,
                            #                  backward, j + 1, cost)
                            ll = nxt - pos
                            n_length[nxt] = ll | (9 << 25)
                            n_distance[nxt] = backward
                            n_dil[nxt] = dil
                            u[nxt] = cost
                            if ll > result:
                                result = ll
                    l = run_end + 1
                best_len = ln

        # At higher iterations look only for new last distance matches, since
        # looking only for new command start positions with the same distances
        # does not help much.
        if k >= 2:
            continue

        # Loop through all possible copy lengths at this position.
        ln = min_len
        for jj in range(match_off, match_off + num_matches):
            dist = m_dist[jj]
            lac = m_lac[jj]
            is_dictionary_match = dist > dictionary_start + gap
            # We already tried all possible last distance matches, so we can
            # use normal distance code here.
            dist_code = dist + BROTLI_NUM_DISTANCE_SHORT_CODES - 1
            sym_cache = _DIST_SYMBOL_CACHE.get(
                (params.dist.num_direct_distance_codes,
                 params.dist.distance_postfix_bits))
            if sym_cache is None:
                sym_cache = {}
                _DIST_SYMBOL_CACHE[(params.dist.num_direct_distance_codes,
                                    params.dist.distance_postfix_bits)] = \
                    sym_cache
            dist_symbol = sym_cache.get(dist_code)
            if dist_symbol is None:
                dist_symbol = _prefix_encode_copy_distance_code(
                    dist_code, params.dist.num_direct_distance_codes,
                    params.dist.distance_postfix_bits)
                if len(sym_cache) < (1 << 20):
                    sym_cache[dist_code] = dist_symbol
            f[0] = base_cost + (dist_symbol >> 10)
            f[0] = f[0] + cost_dist[dist_symbol & 0x3FF]
            dist_cost = f[0]

            # Try all copy lengths up until the maximum copy length
            # corresponding to this distance. If the distance refers to the
            # static dictionary, or the maximum length is long enough, try only
            # one maximum length.
            max_match_len = lac >> 5
            if ln < max_match_len and (is_dictionary_match or
                                       max_match_len > max_zopfli_len_):
                ln = max_match_len
            if ln > max_match_len:
                continue
            if is_dictionary_match:
                # BackwardMatchLengthCode; only len == max_match_len is tried.
                code = lac & 31
                len_code = code if code else max_match_len
                copycode = copy_code_table[len_code] if len_code < 2118 \
                    else 23
                cmdcode = cmd_codes[copycode]
                f[0] = dist_cost + copy_extra[copycode]
                f[0] = f[0] + cost_cmd[cmdcode]
                cost = f[0]
                nxt = pos + ln
                if cost < u[nxt]:
                    n_length[nxt] = ln | ((ln + 9 - len_code) << 25)
                    n_distance[nxt] = dist
                    n_dil[nxt] = ins_bits
                    u[nxt] = cost
                    if ln > result:
                        result = ln
                ln += 1
                continue
            while ln <= max_match_len:
                if ln < 2118:
                    copycode = copy_code_table[ln]
                    run_end = copy_run_end[ln]
                    if run_end > max_match_len:
                        run_end = max_match_len
                else:
                    copycode = 23
                    run_end = max_match_len
                cmdcode = cmd_codes[copycode]
                f[0] = dist_cost + copy_extra[copycode]
                f[0] = f[0] + cost_cmd[cmdcode]
                cost = f[0]
                for nxt in range(pos + ln, pos + run_end + 1):
                    if cost < u[nxt]:
                        # UpdateZopfliNode(nodes, pos, start, len, len, dist,
                        #                  0, cost)
                        ll = nxt - pos
                        n_length[nxt] = ll | (9 << 25)
                        n_distance[nxt] = dist
                        n_dil[nxt] = ins_bits
                        u[nxt] = cost
                        if ll > result:
                            result = ll
                ln = run_end + 1
    return result


def _find_match_length(s, i, j, limit):
    """FindMatchLengthWithLimit(&s[i], &s[j], limit)."""
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


def _compute_shortest_path_from_nodes(num_bytes, nodes):
    """ComputeShortestPathFromNodes."""
    n_dil = nodes.dcode_insert_length
    n_length = nodes.length
    u = nodes.u
    index = num_bytes
    num_commands = 0
    while (n_dil[index] & 0x7FFFFFF) == 0 and n_length[index] == 1:
        index -= 1
    u[index] = BROTLI_UINT32_MAX
    while index != 0:
        ln = (n_length[index] & 0x1FFFFFF) + (n_dil[index] & 0x7FFFFFF)
        index -= ln
        u[index] = ln
        num_commands += 1
    return num_commands


def brotli_zopfli_create_commands(num_bytes, block_start, nodes, dist_cache,
                                  last_insert_len, params, commands,
                                  num_literals):
    """BrotliZopfliCreateCommands.

    Appends the commands to ``commands``; ``dist_cache`` is updated in place.
    Returns (last_insert_len, num_literals).
    REQUIRES: nodes != NULL and len(nodes) >= num_bytes + 1"""
    stream_offset = params.stream_offset
    max_backward_limit_ = max_backward_limit(params.lgwin)
    pos = 0
    u = nodes.u
    offset = u[0]
    i = 0
    gap = 0
    while offset != BROTLI_UINT32_MAX:
        nxt = pos + offset
        copy_length = nodes.length[nxt] & 0x1FFFFFF
        insert_length = nodes.dcode_insert_length[nxt] & 0x7FFFFFF
        pos += insert_length
        offset = u[nxt]
        if i == 0:
            insert_length += last_insert_len
            last_insert_len = 0
        distance = nodes.distance[nxt]
        len_code = copy_length + 9 - (nodes.length[nxt] >> 25)
        dictionary_start = block_start + pos + stream_offset
        if dictionary_start > max_backward_limit_:
            dictionary_start = max_backward_limit_
        is_dictionary = distance > dictionary_start + gap
        dist_code = _zopfli_node_distance_code(nodes, nxt)
        commands.append(init_command(params.dist, insert_length, copy_length,
                                     len_code - copy_length, dist_code))
        if not is_dictionary and dist_code > 0:
            dist_cache[3] = dist_cache[2]
            dist_cache[2] = dist_cache[1]
            dist_cache[1] = dist_cache[0]
            dist_cache[0] = distance
        num_literals += insert_length
        pos += copy_length
        i += 1
    last_insert_len += num_bytes - pos
    return last_insert_len, num_literals


def _zopfli_iterate(num_bytes, position, ringbuffer, ringbuffer_mask, params,
                    gap, dist_cache, model, num_matches, m_dist, m_lac,
                    nodes):
    """ZopfliIterate."""
    stream_offset = params.stream_offset
    max_backward_limit_ = max_backward_limit(params.lgwin)
    max_zopfli_len_ = max_zopfli_len(params)
    queue = StartPosQueue()
    cur_match_pos = 0
    nodes.length[0] = 0
    nodes.u[0] = 0.0
    i = 0
    while i + 3 < num_bytes:
        nm = num_matches[i]
        skip = _update_nodes(num_bytes, position, i, ringbuffer,
                             ringbuffer_mask, params, max_backward_limit_,
                             dist_cache, nm, m_dist, m_lac, cur_match_pos,
                             model, queue, nodes)
        if skip < BROTLI_LONG_COPY_QUICK_STEP:
            skip = 0
        cur_match_pos += nm
        if nm == 1 and (m_lac[cur_match_pos - 1] >> 5) > max_zopfli_len_:
            skip = max(m_lac[cur_match_pos - 1] >> 5, skip)
        if skip > 1:
            skip -= 1
            while skip:
                i += 1
                if i + 3 >= num_bytes:
                    break
                _evaluate_node(position + stream_offset, i,
                               max_backward_limit_, gap, dist_cache, model,
                               queue, nodes)
                cur_match_pos += num_matches[i]
                skip -= 1
        i += 1
    return _compute_shortest_path_from_nodes(num_bytes, nodes)


def brotli_zopfli_compute_shortest_path(num_bytes, position, ringbuffer,
                                        ringbuffer_mask, literal_context_lut,
                                        params, dist_cache, hasher, nodes):
    """BrotliZopfliComputeShortestPath.

    Computes the shortest path of commands from position to at most
    position + num_bytes.  Returns the number of commands found.
    REQUIRES: nodes != NULL and len(nodes) >= num_bytes + 1"""
    # C passes &hasher->privat._H10; accept either the Hasher wrapper or H10.
    hasher = getattr(hasher, "privat", hasher)
    stream_offset = params.stream_offset
    max_backward_limit_ = max_backward_limit(params.lgwin)
    max_zopfli_len_ = max_zopfli_len(params)
    store_lookahead = hasher.store_lookahead()
    store_end = (position + num_bytes - store_lookahead + 1
                 if num_bytes >= store_lookahead else position)
    gap = 0
    nodes.length[0] = 0
    nodes.u[0] = 0.0
    model = ZopfliCostModel(params.dist, num_bytes)
    zopfli_cost_model_set_from_literal_costs(model, position, ringbuffer,
                                             ringbuffer_mask)
    queue = StartPosQueue()
    dictionary = None  # params->dictionary.contextual.dict[0] (built-in)
    max_distance_limit = params.dist.max_distance  # noqa: F841
    hash_type_length = hasher.hash_type_length()
    i = 0
    while i + hash_type_length - 1 < num_bytes:
        pos = position + i
        max_distance = pos if pos < max_backward_limit_ \
            else max_backward_limit_
        dictionary_start = pos + stream_offset
        if dictionary_start > max_backward_limit_:
            dictionary_start = max_backward_limit_
        m_dist = []
        m_lac = []
        num_matches = hasher.find_all_matches(
            dictionary, ringbuffer, ringbuffer_mask, pos, num_bytes - i,
            max_distance, dictionary_start + gap, params, m_dist, m_lac)
        if num_matches > 0 and (m_lac[num_matches - 1] >> 5) > max_zopfli_len_:
            m_dist[0] = m_dist[num_matches - 1]
            m_lac[0] = m_lac[num_matches - 1]
            num_matches = 1
        skip = _update_nodes(num_bytes, position, i, ringbuffer,
                             ringbuffer_mask, params, max_backward_limit_,
                             dist_cache, num_matches, m_dist, m_lac, 0,
                             model, queue, nodes)
        if skip < BROTLI_LONG_COPY_QUICK_STEP:
            skip = 0
        if num_matches == 1 and (m_lac[0] >> 5) > max_zopfli_len_:
            skip = max(m_lac[0] >> 5, skip)
        if skip > 1:
            # Add the tail of the copy to the hasher.
            hasher.store_range(ringbuffer, ringbuffer_mask, pos + 1,
                               min(pos + skip, store_end))
            skip -= 1
            while skip:
                i += 1
                if i + hash_type_length - 1 >= num_bytes:
                    break
                _evaluate_node(position + stream_offset, i,
                               max_backward_limit_, gap, dist_cache, model,
                               queue, nodes)
                skip -= 1
        i += 1
    return _compute_shortest_path_from_nodes(num_bytes, nodes)


def brotli_create_zopfli_backward_references(num_bytes, position, ringbuffer,
                                             ringbuffer_mask,
                                             literal_context_lut, params,
                                             hasher, dist_cache,
                                             last_insert_len, commands,
                                             num_commands, num_literals):
    """BrotliCreateZopfliBackwardReferences (quality 10).

    New commands are appended to ``commands``; ``dist_cache`` is updated in
    place.  Returns (last_insert_len, num_commands, num_literals)."""
    nodes = ZopfliNodes(num_bytes + 1)
    num_commands += brotli_zopfli_compute_shortest_path(
        num_bytes, position, ringbuffer, ringbuffer_mask, literal_context_lut,
        params, dist_cache, hasher, nodes)
    last_insert_len, num_literals = brotli_zopfli_create_commands(
        num_bytes, position, nodes, dist_cache, last_insert_len, params,
        commands, num_literals)
    return last_insert_len, num_commands, num_literals


def brotli_create_hq_zopfli_backward_references(num_bytes, position,
                                                ringbuffer, ringbuffer_mask,
                                                literal_context_lut, params,
                                                hasher, dist_cache,
                                                last_insert_len, commands,
                                                num_commands, num_literals):
    """BrotliCreateHqZopfliBackwardReferences (quality 11).

    New commands are appended to ``commands`` (whose length on entry is the
    index of the first new command); ``dist_cache`` is updated in place.
    Returns (last_insert_len, num_commands, num_literals)."""
    # C passes &hasher->privat._H10; accept either the Hasher wrapper or H10.
    hasher = getattr(hasher, "privat", hasher)
    stream_offset = params.stream_offset
    max_backward_limit_ = max_backward_limit(params.lgwin)
    num_matches = [0] * num_bytes
    store_lookahead = hasher.store_lookahead()
    store_end = (position + num_bytes - store_lookahead + 1
                 if num_bytes >= store_lookahead else position)
    cur_match_pos = 0
    gap = 0
    m_dist = []
    m_lac = []
    dictionary = None  # params->dictionary.contextual.dict[0] (built-in)
    hash_type_length = hasher.hash_type_length()
    find_all_matches = hasher.find_all_matches
    i = 0
    while i + hash_type_length - 1 < num_bytes:
        pos = position + i
        max_distance = pos if pos < max_backward_limit_ \
            else max_backward_limit_
        dictionary_start = pos + stream_offset
        if dictionary_start > max_backward_limit_:
            dictionary_start = max_backward_limit_
        max_length = num_bytes - i
        num_found_matches = find_all_matches(
            dictionary, ringbuffer, ringbuffer_mask, pos, max_length,
            max_distance, dictionary_start + gap, params, m_dist, m_lac)
        cur_match_end = cur_match_pos + num_found_matches
        num_matches[i] = num_found_matches
        if num_found_matches > 0:
            match_len = m_lac[cur_match_end - 1] >> 5
            if match_len > MAX_ZOPFLI_LEN_QUALITY_11:
                skip = match_len - 1
                m_dist[cur_match_pos] = m_dist[cur_match_end - 1]
                m_lac[cur_match_pos] = m_lac[cur_match_end - 1]
                cur_match_pos += 1
                del m_dist[cur_match_pos:]
                del m_lac[cur_match_pos:]
                num_matches[i] = 1
                # Add the tail of the copy to the hasher.
                hasher.store_range(ringbuffer, ringbuffer_mask, pos + 1,
                                   min(pos + match_len, store_end))
                for k in range(i + 1, i + 1 + skip):
                    num_matches[k] = 0
                i += skip
            else:
                cur_match_pos = cur_match_end
        i += 1

    orig_num_literals = num_literals
    orig_last_insert_len = last_insert_len
    orig_dist_cache = list(dist_cache[:4])
    orig_num_commands = num_commands
    orig_len_commands = len(commands)
    nodes = ZopfliNodes(num_bytes + 1)
    model = ZopfliCostModel(params.dist, num_bytes)
    for it in range(2):
        if it != 0:
            brotli_init_zopfli_nodes(nodes, num_bytes + 1)
        if it == 0:
            zopfli_cost_model_set_from_literal_costs(
                model, position, ringbuffer, ringbuffer_mask)
        else:
            zopfli_cost_model_set_from_commands(
                model, position, ringbuffer, ringbuffer_mask, commands,
                orig_len_commands, num_commands - orig_num_commands,
                orig_last_insert_len)
        num_commands = orig_num_commands
        num_literals = orig_num_literals
        last_insert_len = orig_last_insert_len
        dist_cache[:4] = orig_dist_cache
        del commands[orig_len_commands:]
        num_commands += _zopfli_iterate(num_bytes, position, ringbuffer,
                                        ringbuffer_mask, params, gap,
                                        dist_cache, model, num_matches,
                                        m_dist, m_lac, nodes)
        last_insert_len, num_literals = brotli_zopfli_create_commands(
            num_bytes, position, nodes, dist_cache, last_insert_len, params,
            commands, num_literals)
    return last_insert_len, num_commands, num_literals
