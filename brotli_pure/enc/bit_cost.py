"""Functions to estimate the bit cost of Huffman trees
(port of c/enc/bit_cost.h, bit_cost_inc.h, bit_cost.c).

All arithmetic is in double precision, exactly as in C (the log2 table holds
float32-rounded values, see fast_log.py).
"""

from .fast_log import fast_log2, _LOG2_TABLE

BROTLI_CODE_LENGTH_CODES = 18
BROTLI_REPEAT_ZERO_CODE_LENGTH = 17

_log2 = __import__('math').log2


def brotli_bits_entropy(population, size=None):
    """BrotliBitsEntropy (bit_cost.c)."""
    if size is None:
        size = len(population)
    tab = _LOG2_TABLE
    s = 0
    retval = 0.0
    for i in range(size):
        p = population[i]
        s += p
        if p < 256:
            retval -= p * tab[p]
        else:
            retval -= p * _log2(p)
    if s:
        retval += s * fast_log2(s)
    if retval < s:
        # At least one bit per literal is needed.
        retval = float(s)
    return retval


# Alias using the name requested by the encoder API contract.
bits_entropy = brotli_bits_entropy


def shannon_entropy(population, size=None):
    """ShannonEntropy from older brotli versions (not used by v1.2.1).

    Returns (retval, total) like the old C function (total was an
    out-parameter)."""
    if size is None:
        size = len(population)
    total = 0
    retval = 0.0
    for i in range(size):
        p = population[i]
        total += p
        retval -= p * fast_log2(p)
    if total:
        retval += total * fast_log2(total)
    return retval, total


def brotli_population_cost(histogram):
    """FN(BrotliPopulationCost) (bit_cost_inc.h); works for all three
    histogram kinds (the alphabet size is len(histogram.data))."""
    kOneSymbolHistogramCost = 12.0
    kTwoSymbolHistogramCost = 20.0
    kThreeSymbolHistogramCost = 28.0
    kFourSymbolHistogramCost = 37.0
    data = histogram.data
    data_size = len(data)
    if histogram.total_count == 0:
        return kOneSymbolHistogramCost
    s = []
    for i in range(data_size):
        if data[i] > 0:
            s.append(i)
            if len(s) > 4:
                break
    count = len(s)
    if count == 1:
        return kOneSymbolHistogramCost
    if count == 2:
        return kTwoSymbolHistogramCost + float(histogram.total_count)
    if count == 3:
        histo0 = data[s[0]]
        histo1 = data[s[1]]
        histo2 = data[s[2]]
        histomax = max(histo0, histo1, histo2)
        # uint32_t arithmetic in C.
        return kThreeSymbolHistogramCost + float(
            (2 * (histo0 + histo1 + histo2) - histomax) & 0xFFFFFFFF)
    if count == 4:
        histo = [data[s[0]], data[s[1]], data[s[2]], data[s[3]]]
        # Sort (same selection-swap sort as C; result is a descending sort).
        for i in range(4):
            for j in range(i + 1, 4):
                if histo[j] > histo[i]:
                    histo[j], histo[i] = histo[i], histo[j]
        h23 = histo[2] + histo[3]
        histomax = max(h23, histo[0])
        return kFourSymbolHistogramCost + float(
            (3 * h23 + 2 * (histo[0] + histo[1]) - histomax) & 0xFFFFFFFF)

    # In this loop we compute the entropy of the histogram and simultaneously
    # build a simplified histogram of the code length codes where we use the
    # zero repeat code 17, but we don't use the non-zero repeat code 16.
    tab = _LOG2_TABLE
    bits = 0.0
    max_depth = 1
    depth_histo = [0] * BROTLI_CODE_LENGTH_CODES
    log2total = fast_log2(histogram.total_count)
    i = 0
    while i < data_size:
        d = data[i]
        if d > 0:
            # -log2(P(symbol)) = log2(total_count) - log2(count(symbol))
            log2p = log2total - (tab[d] if d < 256 else _log2(d))
            # Approximate the bit depth by round(-log2(P(symbol)))
            depth = int(log2p + 0.5)
            bits += d * log2p
            if depth > 15:
                depth = 15
            if depth > max_depth:
                max_depth = depth
            depth_histo[depth] += 1
            i += 1
        else:
            # Compute the run length of zeros and add the appropriate number
            # of 0 and 17 code length codes to the code length code histogram.
            reps = 1
            k = i + 1
            while k < data_size and data[k] == 0:
                reps += 1
                k += 1
            i += reps
            if i == data_size:
                # Don't add any cost for the last zero run, since these are
                # encoded only implicitly.
                break
            if reps < 3:
                depth_histo[0] += reps
            else:
                reps -= 2
                while reps > 0:
                    depth_histo[BROTLI_REPEAT_ZERO_CODE_LENGTH] += 1
                    # Add the 3 extra bits for the 17 code length code.
                    bits += 3
                    reps >>= 3
    # Add the estimated encoding cost of the code length code histogram.
    bits += float(18 + 2 * max_depth)
    # Add the entropy of the code length code histogram.
    bits += brotli_bits_entropy(depth_histo, BROTLI_CODE_LENGTH_CODES)
    return bits


# C has three instantiations; they are identical up to the alphabet size.
brotli_population_cost_literal = brotli_population_cost
brotli_population_cost_command = brotli_population_cost
brotli_population_cost_distance = brotli_population_cost
population_cost = brotli_population_cost
