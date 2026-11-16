"""Functions for clustering similar histograms together
(port of c/enc/cluster.c, cluster.h and cluster_inc.h).

The C code instantiates cluster_inc.h for the Literal, Command and Distance
histogram types; the Python functions are generic and operate on
``histogram.Histogram`` objects (``data`` list, ``total_count``,
``bit_cost``).

HistogramPair values are represented as immutable tuples
``(idx1, idx2, cost_combo, cost_diff)``; a pairs "array" is a Python list
whose length is the C ``num_pairs``.
"""

from operator import add as _add

from .bit_cost import brotli_population_cost
from .fast_log import fast_log2

K_INVALID_INDEX = 0xFFFFFFFF


def _histogram_pair_is_less(p1, p2):
    """HistogramPairIsLess."""
    if p1[3] != p2[3]:
        return p1[3] > p2[3]
    return (p1[1] - p1[0]) > (p2[1] - p2[0])


def cluster_cost_diff(size_a, size_b):
    """ClusterCostDiff: entropy reduction of the context map when we combine
    two clusters."""
    size_c = size_a + size_b
    return (float(size_a) * fast_log2(size_a) +
            float(size_b) * fast_log2(size_b) -
            float(size_c) * fast_log2(size_c))


def _set_sum(tmp, h1, h2):
    """*tmp = *h1; HistogramAddHistogram(tmp, h2);"""
    tmp.data = list(map(_add, h1.data, h2.data))
    tmp.total_count = h1.total_count + h2.total_count
    tmp.bit_cost = h1.bit_cost


def _add_histogram(self, v):
    """HistogramAddHistogram(self, v)."""
    self.total_count += v.total_count
    self.data = list(map(_add, self.data, v.data))


def brotli_compare_and_push_to_queue(out, tmp, cluster_size, idx1, idx2,
                                     max_num_pairs, pairs):
    """BrotliCompareAndPushToQueue.

    Computes the bit cost reduction by combining out[idx1] and out[idx2] and
    if it is below a threshold, stores the pair (idx1, idx2) in the pairs
    queue (list, mutated in place)."""
    if idx1 == idx2:
        return
    if idx2 < idx1:
        idx1, idx2 = idx2, idx1
    h1 = out[idx1]
    h2 = out[idx2]
    cost_diff = 0.5 * cluster_cost_diff(cluster_size[idx1], cluster_size[idx2])
    cost_diff -= h1.bit_cost
    cost_diff -= h2.bit_cost
    is_good_pair = False
    cost_combo = 0.0
    if h1.total_count == 0:
        cost_combo = h2.bit_cost
        is_good_pair = True
    elif h2.total_count == 0:
        cost_combo = h1.bit_cost
        is_good_pair = True
    else:
        if not pairs:
            threshold = 1e99
        else:
            threshold = pairs[0][3]
            if threshold < 0.0:
                threshold = 0.0
        _set_sum(tmp, h1, h2)
        c = brotli_population_cost(tmp)
        if c < threshold - cost_diff:
            cost_combo = c
            is_good_pair = True
    if is_good_pair:
        cost_diff += cost_combo
        p = (idx1, idx2, cost_combo, cost_diff)
        num_pairs = len(pairs)
        if num_pairs > 0 and _histogram_pair_is_less(pairs[0], p):
            # Replace the top of the queue if needed.
            if num_pairs < max_num_pairs:
                pairs.append(pairs[0])
            pairs[0] = p
        elif num_pairs < max_num_pairs:
            pairs.append(p)


def brotli_histogram_combine(out, tmp, cluster_size, symbols, symbols_off,
                             clusters, clusters_off, pairs, num_clusters,
                             symbols_size, max_clusters, max_num_pairs):
    """BrotliHistogramCombine.

    ``symbols`` / ``clusters`` are lists; the C pointers are
    ``&symbols[symbols_off]`` and ``&clusters[clusters_off]``.  ``pairs`` is
    reset (num_pairs = 0) and used as the work queue.  Returns the new
    number of clusters."""
    cost_diff_threshold = 0.0
    min_cluster_size = 1
    del pairs[:]
    push = brotli_compare_and_push_to_queue
    # We maintain a vector of histogram pairs, with the property that the pair
    # with the maximum bit cost reduction is the first.
    for idx1 in range(num_clusters):
        c1 = clusters[clusters_off + idx1]
        for idx2 in range(idx1 + 1, num_clusters):
            push(out, tmp, cluster_size, c1, clusters[clusters_off + idx2],
                 max_num_pairs, pairs)

    while num_clusters > min_cluster_size:
        top = pairs[0]
        if top[3] >= cost_diff_threshold:
            cost_diff_threshold = 1e99
            min_cluster_size = max_clusters
            continue
        # Take the best pair from the top of heap.
        best_idx1 = top[0]
        best_idx2 = top[1]
        _add_histogram(out[best_idx1], out[best_idx2])
        out[best_idx1].bit_cost = top[2]
        cluster_size[best_idx1] += cluster_size[best_idx2]
        end = symbols_off + symbols_size
        for i in range(symbols_off, end):
            if symbols[i] == best_idx2:
                symbols[i] = best_idx1
        for i in range(num_clusters):
            if clusters[clusters_off + i] == best_idx2:
                # memmove(&clusters[i], &clusters[i + 1], ...)
                a = clusters_off + i
                b = clusters_off + num_clusters
                clusters[a:b - 1] = clusters[a + 1:b]
                break
        num_clusters -= 1
        # Remove pairs intersecting the just combined best pair.
        copy_to_idx = 0
        for i in range(len(pairs)):
            p = pairs[i]
            if (p[0] == best_idx1 or p[1] == best_idx1 or
                    p[0] == best_idx2 or p[1] == best_idx2):
                # Remove invalid pair from the queue.
                continue
            if _histogram_pair_is_less(pairs[0], p):
                # Replace the top of the queue if needed.
                front = pairs[0]
                pairs[0] = p
                pairs[copy_to_idx] = front
            else:
                pairs[copy_to_idx] = p
            copy_to_idx += 1
        del pairs[copy_to_idx:]

        # Push new pairs formed with the combined histogram to the heap.
        for i in range(num_clusters):
            push(out, tmp, cluster_size, best_idx1,
                 clusters[clusters_off + i], max_num_pairs, pairs)
    return num_clusters


def brotli_histogram_bit_cost_distance(histogram, candidate, tmp):
    """BrotliHistogramBitCostDistance: what is the bit cost of moving
    histogram from cur_symbol to candidate."""
    if histogram.total_count == 0:
        return 0.0
    _set_sum(tmp, histogram, candidate)
    return brotli_population_cost(tmp) - candidate.bit_cost


def brotli_histogram_remap(in_, in_size, clusters, num_clusters, out, tmp,
                           symbols):
    """BrotliHistogramRemap: find the best 'out' histogram for each of the
    'in' histograms.  Note: we assume that out[]->bit_cost_ is already
    up-to-date."""
    dist = brotli_histogram_bit_cost_distance
    for i in range(in_size):
        best_out = symbols[0] if i == 0 else symbols[i - 1]
        h = in_[i]
        best_bits = dist(h, out[best_out], tmp)
        for j in range(num_clusters):
            cj = clusters[j]
            cur_bits = dist(h, out[cj], tmp)
            if cur_bits < best_bits:
                best_bits = cur_bits
                best_out = cj
        symbols[i] = best_out

    # Recompute each out based on raw and symbols.
    for i in range(num_clusters):
        out[clusters[i]].clear()
    for i in range(in_size):
        _add_histogram(out[symbols[i]], in_[i])


def brotli_histogram_reindex(out, symbols, length):
    """BrotliHistogramReindex.

    Reorders elements of the out[0..length) array and changes values in
    symbols[0..length) so that the first occurrences of values in symbols
    come in consecutive increasing order.  Returns N, the number of unique
    values in symbols[]; out[0..N) holds the reordered histograms."""
    new_index = [K_INVALID_INDEX] * length
    next_index = 0
    for i in range(length):
        if new_index[symbols[i]] == K_INVALID_INDEX:
            new_index[symbols[i]] = next_index
            next_index += 1
    tmp = [None] * next_index
    next_index = 0
    for i in range(length):
        if new_index[symbols[i]] == next_index:
            tmp[next_index] = out[symbols[i]]
            next_index += 1
        symbols[i] = new_index[symbols[i]]
    for i in range(next_index):
        out[i] = tmp[i]
    return next_index


def brotli_cluster_histograms(in_, in_size, max_histograms, out,
                              histogram_symbols):
    """BrotliClusterHistograms.

    ``out`` is a list that receives the clustered histograms (it is replaced
    in place and truncated to the returned out_size); ``histogram_symbols``
    is a list of at least in_size entries receiving the context map.
    Returns out_size."""
    cluster_size = [1] * in_size
    clusters = [0] * in_size
    num_clusters = 0
    max_input_histograms = 64
    pairs_capacity = max_input_histograms * max_input_histograms // 2
    # For the first pass of clustering, we allow all pairs.
    pairs = []
    tmp = in_[0].copy() if in_size else None

    del out[:]
    for i in range(in_size):
        h = in_[i].copy()
        h.bit_cost = brotli_population_cost(in_[i])
        out.append(h)
        histogram_symbols[i] = i

    for i in range(0, in_size, max_input_histograms):
        num_to_combine = min(in_size - i, max_input_histograms)
        for j in range(num_to_combine):
            clusters[num_clusters + j] = i + j
        num_new_clusters = brotli_histogram_combine(
            out, tmp, cluster_size, histogram_symbols, i, clusters,
            num_clusters, pairs, num_to_combine, num_to_combine,
            max_histograms, pairs_capacity)
        num_clusters += num_new_clusters

    # For the second pass, we limit the total number of histogram pairs.
    # After this limit is reached, we only keep searching for the best pair.
    max_num_pairs = min(64 * num_clusters, (num_clusters // 2) * num_clusters)
    # Collapse similar histograms.
    num_clusters = brotli_histogram_combine(
        out, tmp, cluster_size, histogram_symbols, 0, clusters, 0, pairs,
        num_clusters, in_size, max_histograms, max_num_pairs)
    # Find the optimal map from original histograms to the final ones.
    brotli_histogram_remap(in_, in_size, clusters, num_clusters, out, tmp,
                           histogram_symbols)
    # Convert the context map to a canonical form.
    out_size = brotli_histogram_reindex(out, histogram_symbols, in_size)
    del out[out_size:]
    return out_size
