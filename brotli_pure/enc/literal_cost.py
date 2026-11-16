"""Literal cost model to allow backward reference replacement to be efficient
(port of c/enc/literal_cost.c / literal_cost.h)."""

from array import array

from .fast_log import fast_log2
from .utf8_util import brotli_is_mostly_utf8, kMinUTF8Ratio


def _utf8_position(last, c, clamp):
    """UTF8Position."""
    if c < 128:
        return 0  # Next one is the 'Byte 1' again.
    elif c >= 192:  # Next one is the 'Byte 2' of utf-8 encoding.
        return 1 if clamp > 1 else clamp
    else:
        # Let's decide over the last byte if this ends the sequence.
        if last < 0xE0:
            return 0  # Completed two or three byte coding.
        else:  # Next one is the 'Byte 3' of utf-8 encoding.
            return 2 if clamp > 2 else clamp


def _decide_multi_byte_stats_level(pos, length, mask, data):
    """DecideMultiByteStatsLevel."""
    counts = [0, 0, 0]
    max_utf8 = 1  # should be 2, but 1 compresses better.
    last_c = 0
    for i in range(length):
        c = data[(pos + i) & mask]
        counts[_utf8_position(last_c, c, 2)] += 1
        last_c = c
    if counts[2] < 500:
        max_utf8 = 1
    if counts[1] + counts[2] < 25:
        max_utf8 = 0
    return max_utf8


def _estimate_bit_costs_for_literals_utf8(pos, length, mask, data, histogram,
                                          cost):
    """EstimateBitCostsForLiteralsUTF8."""
    # max_utf8 is 0 (normal ASCII single byte modeling),
    # 1 (for 2-byte UTF-8 modeling), or 2 (for 3-byte UTF-8 modeling).
    max_utf8 = _decide_multi_byte_stats_level(pos, length, mask, data)
    window_half = 495
    in_window = min(window_half, length)
    in_window_utf8 = [0, 0, 0]
    for k in range(3 * 256):
        histogram[k] = 0
    utf8_pos_fn = _utf8_position
    log2 = fast_log2
    # Bootstrap histograms.
    last_c = 0
    utf8_pos = 0
    for i in range(in_window):
        c = data[(pos + i) & mask]
        histogram[256 * utf8_pos + c] += 1
        in_window_utf8[utf8_pos] += 1
        utf8_pos = utf8_pos_fn(last_c, c, max_utf8)
        last_c = c
    # Compute bit costs with sliding window.
    prologue_length = 2000
    multiplier = 0.35 / 2000
    out = [0.0] * length
    for i in range(length):
        if i >= window_half:
            # Remove a byte in the past.
            c = 0 if i < window_half + 1 else \
                data[(pos + i - window_half - 1) & mask]
            last_c = 0 if i < window_half + 2 else \
                data[(pos + i - window_half - 2) & mask]
            utf8_pos2 = utf8_pos_fn(last_c, c, max_utf8)
            histogram[256 * utf8_pos2 +
                      data[(pos + i - window_half) & mask]] -= 1
            in_window_utf8[utf8_pos2] -= 1
        if i + window_half < length:
            # Add a byte in the future.
            c = data[(pos + i + window_half - 1) & mask]
            last_c = data[(pos + i + window_half - 2) & mask]
            utf8_pos2 = utf8_pos_fn(last_c, c, max_utf8)
            histogram[256 * utf8_pos2 +
                      data[(pos + i + window_half) & mask]] += 1
            in_window_utf8[utf8_pos2] += 1
        c = 0 if i < 1 else data[(pos + i - 1) & mask]
        last_c = 0 if i < 2 else data[(pos + i - 2) & mask]
        utf8_pos = utf8_pos_fn(last_c, c, max_utf8)
        masked_pos = (pos + i) & mask
        histo = histogram[256 * utf8_pos + data[masked_pos]]
        if histo == 0:
            histo = 1
        lit_cost = log2(in_window_utf8[utf8_pos]) - log2(histo)
        lit_cost += 0.02905
        if lit_cost < 1.0:
            lit_cost *= 0.5
            lit_cost += 0.5
        # Make the first bytes more expensive -- seems to help, not sure why.
        # Perhaps because the entropy source is changing its properties
        # rapidly in the beginning of the file, perhaps because the beginning
        # of the data is a statistical "anomaly".
        if i < prologue_length:
            lit_cost += 0.35 + multiplier * float(i)
        out[i] = lit_cost
    return out


def brotli_estimate_bit_costs_for_literals(pos, length, mask, data,
                                           histogram, cost, cost_offset=0):
    """BrotliEstimateBitCostsForLiterals.

    Writes ``(float)`` bit costs of data[pos .. pos+length) into
    cost[cost_offset .. cost_offset+length) (``cost`` should be an
    ``array('f')`` or a list; values are rounded to float32 either way).
    ``histogram`` is scratch space of at least 3 * 256 entries."""
    if brotli_is_mostly_utf8(data, pos, mask, length, kMinUTF8Ratio):
        out = _estimate_bit_costs_for_literals_utf8(pos, length, mask, data,
                                                    histogram, cost)
    else:
        window_half = 2000
        in_window = min(window_half, length)
        for k in range(256):
            histogram[k] = 0
        # Bootstrap histogram.
        for i in range(in_window):
            histogram[data[(pos + i) & mask]] += 1
        log2 = fast_log2
        out = [0.0] * length
        # Compute bit costs with sliding window.
        for i in range(length):
            if i >= window_half:
                # Remove a byte in the past.
                histogram[data[(pos + i - window_half) & mask]] -= 1
                in_window -= 1
            if i + window_half < length:
                # Add a byte in the future.
                histogram[data[(pos + i + window_half) & mask]] += 1
                in_window += 1
            histo = histogram[data[(pos + i) & mask]]
            if histo == 0:
                histo = 1
            lit_cost = log2(in_window) - log2(histo)
            lit_cost += 0.029
            if lit_cost < 1.0:
                lit_cost *= 0.5
                lit_cost += 0.5
            out[i] = lit_cost
    # cost[i] = (float)lit_cost
    out32 = array("f", out)
    cost[cost_offset:cost_offset + length] = out32 if isinstance(
        cost, array) else out32.tolist()
