"""Block split point selection utilities
(port of c/enc/block_splitter.c and block_splitter_inc.h).

block_splitter_inc.h is instantiated in C for Literal (uint8_t data),
Command and Distance (uint16_t data) histograms; here the functions are
generic and take a histogram factory plus the alphabet size.

``BlockSplit`` objects (defined by the shared core, metablock.py) are used
through their ``num_types``, ``num_blocks``, ``types`` and ``lengths``
attributes (``types``/``lengths`` are Python lists).
"""

from operator import add as _add

from .bit_cost import brotli_population_cost
from .cluster import (brotli_histogram_bit_cost_distance,
                      brotli_histogram_combine)
from .fast_log import fast_log2
from .histogram import histogram_command, histogram_distance, histogram_literal

K_MAX_LITERAL_HISTOGRAMS = 100
K_MAX_COMMAND_HISTOGRAMS = 50
K_LITERAL_BLOCK_SWITCH_COST = 28.1
K_COMMAND_BLOCK_SWITCH_COST = 13.5
K_DISTANCE_BLOCK_SWITCH_COST = 14.6
K_LITERAL_STRIDE_LENGTH = 70
K_COMMAND_STRIDE_LENGTH = 40
K_DISTANCE_STRIDE_LENGTH = 40
K_SYMBOLS_PER_LITERAL_HISTOGRAM = 544
K_SYMBOLS_PER_COMMAND_HISTOGRAM = 530
K_SYMBOLS_PER_DISTANCE_HISTOGRAM = 544
K_MIN_LENGTH_FOR_BLOCK_SPLITTING = 128
K_ITER_MUL_FOR_REFINING = 2
K_MIN_ITERS_FOR_REFINING = 100

HISTOGRAMS_PER_BATCH = 64
CLUSTERS_PER_BATCH = 16

BROTLI_MAX_NUMBER_OF_BLOCK_TYPES = 256
HQ_ZOPFLIFICATION_QUALITY = 11

BROTLI_NUM_LITERAL_SYMBOLS = 256
BROTLI_NUM_COMMAND_SYMBOLS = 704
BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS = 544

K_INVALID_INDEX = 0xFFFFFFFF


def _count_literals(cmds, num_commands):
    """CountLiterals."""
    total_length = 0
    for i in range(num_commands):
        total_length += cmds[i].insert_len
    return total_length


def _copy_literals_to_byte_array(cmds, num_commands, data, offset, mask):
    """CopyLiteralsToByteArray: returns the literals as a bytes object."""
    out = []
    from_pos = offset & mask
    for i in range(num_commands):
        cmd = cmds[i]
        insert_len = cmd.insert_len
        if from_pos + insert_len > mask:
            head_size = mask + 1 - from_pos
            out.append(bytes(data[from_pos:from_pos + head_size]))
            from_pos = 0
            insert_len -= head_size
        if insert_len > 0:
            out.append(bytes(data[from_pos:from_pos + insert_len]))
        from_pos = (from_pos + insert_len +
                    (cmd.copy_len & 0x1FFFFFF)) & mask
    return b"".join(out)


def _my_rand(seed):
    """MyRand: returns the new seed (which is also the random value).
    Initial seed should be 7. In this case, loop length is (1 << 29)."""
    return (seed * 16807) & 0xFFFFFFFF


def _bit_cost(count):
    """BitCost."""
    return -2.0 if count == 0 else fast_log2(count)


def _histogram_add_vector(h, data, pos, n):
    """HistogramAddVector(h, data + pos, n)."""
    h.total_count += n
    d = h.data
    for k in range(pos, pos + n):
        d[data[k]] += 1


def _initial_entropy_codes(data, length, stride, num_histograms, histograms):
    """InitialEntropyCodes."""
    seed = 7
    block_length = length // num_histograms
    for h in histograms[:num_histograms]:
        h.clear()
    for i in range(num_histograms):
        pos = length * i // num_histograms
        if i != 0:
            seed = _my_rand(seed)
            pos += seed % block_length
        if pos + stride >= length:
            pos = length - stride - 1
        _histogram_add_vector(histograms[i], data, pos, stride)


def _random_sample(seed, data, length, stride, sample):
    """RandomSample: returns the new seed."""
    pos = 0
    if stride >= length:
        stride = length
    else:
        seed = _my_rand(seed)
        pos = seed % (length - stride + 1)
    _histogram_add_vector(sample, data, pos, stride)
    return seed


def _refine_entropy_codes(data, length, stride, num_histograms, histograms,
                          tmp):
    """RefineEntropyCodes."""
    iters = K_ITER_MUL_FOR_REFINING * length // stride + K_MIN_ITERS_FOR_REFINING
    seed = 7
    iters = ((iters + num_histograms - 1) // num_histograms) * num_histograms
    for it in range(iters):
        tmp.clear()
        seed = _random_sample(seed, data, length, stride, tmp)
        h = histograms[it % num_histograms]
        h.total_count += tmp.total_count
        h.data = list(map(_add, h.data, tmp.data))


def _find_blocks(data, length, block_switch_bitcost, num_histograms,
                 histograms, alphabet_size, block_id):
    """FindBlocks.

    Assigns a block id from the range [0, num_histograms) to each data
    element in data[0..length) and fills in block_id[0..length) with the
    assigned values.  Returns the number of blocks, i.e. one plus the number
    of block switches.

    The per-histogram inner loops of the C code are evaluated with list
    operations; the arithmetic (and its order) is identical."""
    num_blocks = 1
    # Trivial case: single histogram -> single block type.
    if num_histograms <= 1:
        for i in range(length):
            block_id[i] = 0
        return 1

    # Fill bitcost for each symbol of all histograms.
    # Non-existing symbol cost: 2 + log2(total_count).
    # Regular symbol cost: -log2(symbol_count / total_count).
    hs = histograms[:num_histograms]
    log_totals = [fast_log2(h.total_count) for h in hs]
    rows = [None] * alphabet_size
    for sym in set(data):
        rows[sym] = [log_totals[j] - _bit_cost(hs[j].data[sym])
                     for j in range(num_histograms)]

    # After each iteration of this loop, cost[k] will contain the difference
    # between the minimum cost of arriving at the current byte position using
    # entropy code k, and the minimum cost of arriving at the current byte
    # position. This difference is capped at the block switch cost, and if it
    # reaches block switch cost, it means that when we trace back from the
    # last position, we need to switch here.
    cost = [0.0] * num_histograms
    switch_signal = [0] * length
    prologue_length = 2000
    multiplier = 0.07 / 2000
    rng = range(num_histograms)
    for byte_ix in range(length):
        cost = list(map(_add, cost, rows[data[byte_ix]]))
        min_cost = min(cost)
        block_id[byte_ix] = cost.index(min_cost)
        block_switch_cost = block_switch_bitcost
        # More blocks for the beginning.
        if byte_ix < prologue_length:
            block_switch_cost *= 0.77 + multiplier * float(byte_ix)
        cost = [c - min_cost for c in cost]
        if max(cost) >= block_switch_cost:
            sig = 0
            for k in rng:
                if cost[k] >= block_switch_cost:
                    cost[k] = block_switch_cost
                    sig |= 1 << k
            switch_signal[byte_ix] = sig

    byte_ix = length - 1
    # Trace back from the last position and switch at the marked places.
    cur_id = block_id[byte_ix]
    while byte_ix > 0:
        byte_ix -= 1
        if (switch_signal[byte_ix] >> cur_id) & 1:
            if cur_id != block_id[byte_ix]:
                cur_id = block_id[byte_ix]
                num_blocks += 1
        block_id[byte_ix] = cur_id
    return num_blocks


def _remap_block_ids(block_ids, length, num_histograms):
    """RemapBlockIds."""
    k_invalid_id = 256
    new_id = [k_invalid_id] * num_histograms
    next_id = 0
    for i in range(length):
        b = block_ids[i]
        if new_id[b] == k_invalid_id:
            new_id[b] = next_id
            next_id += 1
    for i in range(length):
        block_ids[i] = new_id[block_ids[i]]
    return next_id


def _build_block_histograms(data, length, block_ids, num_histograms,
                            histograms):
    """BuildBlockHistograms."""
    for h in histograms[:num_histograms]:
        h.clear()
    datas = [h.data for h in histograms]
    counts = [0] * num_histograms
    for i in range(length):
        b = block_ids[i]
        datas[b][data[i]] += 1
        counts[b] += 1
    for j in range(num_histograms):
        histograms[j].total_count += counts[j]


def _cluster_blocks(data, length, num_blocks, block_ids, split, new_histogram):
    """ClusterBlocks: given the initial partitioning build partitioning with
    limited number of histograms (and block types)."""
    histogram_symbols = [0] * num_blocks
    expected_num_clusters = (CLUSTERS_PER_BATCH *
                             (num_blocks + HISTOGRAMS_PER_BATCH - 1) //
                             HISTOGRAMS_PER_BATCH)
    del expected_num_clusters  # only an allocation hint in C
    all_histograms = []
    cluster_size = []
    num_clusters = 0
    histograms = [new_histogram()
                  for _ in range(min(num_blocks, HISTOGRAMS_PER_BATCH))]
    max_num_pairs = HISTOGRAMS_PER_BATCH * HISTOGRAMS_PER_BATCH // 2
    pairs = []
    pos = 0
    sizes = [0] * HISTOGRAMS_PER_BATCH
    new_clusters = [0] * HISTOGRAMS_PER_BATCH
    symbols = [0] * HISTOGRAMS_PER_BATCH
    remap = [0] * HISTOGRAMS_PER_BATCH
    block_lengths = [0] * num_blocks
    tmp = [new_histogram(), new_histogram()]

    # Calculate block lengths (convert repeating values -> series length).
    block_idx = 0
    for i in range(length):
        block_lengths[block_idx] += 1
        if i + 1 == length or block_ids[i] != block_ids[i + 1]:
            block_idx += 1

    # Pre-cluster blocks (cluster batches).
    for i in range(0, num_blocks, HISTOGRAMS_PER_BATCH):
        num_to_combine = min(num_blocks - i, HISTOGRAMS_PER_BATCH)
        for j in range(num_to_combine):
            block_length = block_lengths[i + j]
            h = histograms[j]
            h.clear()
            d = h.data
            for k in range(pos, pos + block_length):
                d[data[k]] += 1
            h.total_count += block_length
            pos += block_length
            h.bit_cost = brotli_population_cost(h)
            new_clusters[j] = j
            symbols[j] = j
            sizes[j] = 1
        num_new_clusters = brotli_histogram_combine(
            histograms, tmp[0], sizes, symbols, 0, new_clusters, 0, pairs,
            num_to_combine, num_to_combine, HISTOGRAMS_PER_BATCH,
            max_num_pairs)
        for j in range(num_new_clusters):
            all_histograms.append(histograms[new_clusters[j]].copy())
            cluster_size.append(sizes[new_clusters[j]])
            remap[new_clusters[j]] = j
        for j in range(num_to_combine):
            histogram_symbols[i + j] = num_clusters + remap[symbols[j]]
        num_clusters += num_new_clusters
    del histograms

    # Final clustering.
    max_num_pairs = min(64 * num_clusters, (num_clusters // 2) * num_clusters)
    clusters = list(range(num_clusters))
    num_final_clusters = brotli_histogram_combine(
        all_histograms, tmp[0], cluster_size, histogram_symbols, 0, clusters,
        0, pairs, num_clusters, num_blocks, BROTLI_MAX_NUMBER_OF_BLOCK_TYPES,
        max_num_pairs)

    # Assign blocks to final histograms.
    new_index = [K_INVALID_INDEX] * num_clusters
    pos = 0
    next_index = 0
    t0 = tmp[0]
    t1 = tmp[1]
    dist = brotli_histogram_bit_cost_distance
    for i in range(num_blocks):
        t0.clear()
        d = t0.data
        bl = block_lengths[i]
        for k in range(pos, pos + bl):
            d[data[k]] += 1
        t0.total_count += bl
        pos += bl
        # Among equally good histograms prefer last used.
        # TODO(eustas): should we give a block-switch discount here?
        best_out = histogram_symbols[0] if i == 0 else histogram_symbols[i - 1]
        best_bits = dist(t0, all_histograms[best_out], t1)
        for j in range(num_final_clusters):
            cj = clusters[j]
            cur_bits = dist(t0, all_histograms[cj], t1)
            if cur_bits < best_bits:
                best_bits = cur_bits
                best_out = cj
        histogram_symbols[i] = best_out
        if new_index[best_out] == K_INVALID_INDEX:
            new_index[best_out] = next_index
            next_index += 1

    # Rewrite final assignment to block-split. There might be less blocks
    # than |num_blocks| due to clustering.
    types = []
    lengths = []
    cur_length = 0
    max_type = 0
    for i in range(num_blocks):
        cur_length += block_lengths[i]
        if (i + 1 == num_blocks or
                histogram_symbols[i] != histogram_symbols[i + 1]):
            id_ = new_index[histogram_symbols[i]] & 0xFF
            types.append(id_)
            lengths.append(cur_length)
            if id_ > max_type:
                max_type = id_
            cur_length = 0
    split.types = types
    split.lengths = lengths
    split.num_blocks = len(types)
    split.num_types = max_type + 1


def _split_byte_vector(data, length, symbols_per_histogram, max_histograms,
                       sampling_stride_length, block_switch_cost, params,
                       split, new_histogram, alphabet_size):
    """SplitByteVector: create BlockSplit (partitioning) given the limits,
    estimates and "effort" parameters."""
    # Calculate number of histograms; initial estimate is one histogram per
    # specified amount of symbols; however, this value is capped.
    num_histograms = length // symbols_per_histogram + 1
    if num_histograms > max_histograms:
        num_histograms = max_histograms

    # Corner case: no input.
    if length == 0:
        split.num_types = 1
        return

    if length < K_MIN_LENGTH_FOR_BLOCK_SPLITTING:
        split.num_types = 1
        nb = split.num_blocks
        del split.types[nb:]
        del split.lengths[nb:]
        split.types.append(0)
        split.lengths.append(length)
        split.num_blocks = nb + 1
        return
    histograms = [new_histogram() for _ in range(num_histograms + 1)]
    tmp = histograms[num_histograms]
    # Find good entropy codes.
    _initial_entropy_codes(data, length, sampling_stride_length,
                           num_histograms, histograms)
    _refine_entropy_codes(data, length, sampling_stride_length,
                          num_histograms, histograms, tmp)
    # Find a good path through literals with the good entropy codes.
    block_ids = [0] * length
    num_blocks = 0
    iters = 3 if params.quality < HQ_ZOPFLIFICATION_QUALITY else 10
    for _ in range(iters):
        num_blocks = _find_blocks(data, length, block_switch_cost,
                                  num_histograms, histograms, alphabet_size,
                                  block_ids)
        num_histograms = _remap_block_ids(block_ids, length, num_histograms)
        _build_block_histograms(data, length, block_ids, num_histograms,
                                histograms)
    _cluster_blocks(data, length, num_blocks, block_ids, split, new_histogram)


def brotli_split_block(cmds, num_commands, data, pos, mask, params,
                       literal_split, insert_and_copy_split, dist_split):
    """BrotliSplitBlock: extracts literals, command distance and prefix
    codes, then applies SplitByteVector to create partitioning."""
    # Create a continuous array of literals.
    literals = _copy_literals_to_byte_array(cmds, num_commands, data, pos,
                                            mask)
    literals_count = len(literals)
    # Create the block split on the array of literals.
    # Literal histograms can have alphabet size up to 256.
    # Though, to accommodate context modeling, less than half of maximum size
    # is allowed.
    _split_byte_vector(literals, literals_count,
                       K_SYMBOLS_PER_LITERAL_HISTOGRAM,
                       K_MAX_LITERAL_HISTOGRAMS, K_LITERAL_STRIDE_LENGTH,
                       K_LITERAL_BLOCK_SWITCH_COST, params, literal_split,
                       histogram_literal, BROTLI_NUM_LITERAL_SYMBOLS)

    # Compute prefix codes for commands.
    insert_and_copy_codes = [cmds[i].cmd_prefix for i in range(num_commands)]
    # Create the block split on the array of command prefixes.
    _split_byte_vector(insert_and_copy_codes, num_commands,
                       K_SYMBOLS_PER_COMMAND_HISTOGRAM,
                       K_MAX_COMMAND_HISTOGRAMS, K_COMMAND_STRIDE_LENGTH,
                       K_COMMAND_BLOCK_SWITCH_COST, params,
                       insert_and_copy_split, histogram_command,
                       BROTLI_NUM_COMMAND_SYMBOLS)

    # Create a continuous array of distance prefixes.
    distance_prefixes = []
    for i in range(num_commands):
        cmd = cmds[i]
        if (cmd.copy_len & 0x1FFFFFF) and cmd.cmd_prefix >= 128:
            distance_prefixes.append(cmd.dist_prefix & 0x3FF)
    # Create the block split on the array of distance prefixes.
    _split_byte_vector(distance_prefixes, len(distance_prefixes),
                       K_SYMBOLS_PER_DISTANCE_HISTOGRAM,
                       K_MAX_COMMAND_HISTOGRAMS, K_DISTANCE_STRIDE_LENGTH,
                       K_DISTANCE_BLOCK_SWITCH_COST, params, dist_split,
                       histogram_distance,
                       BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS)
