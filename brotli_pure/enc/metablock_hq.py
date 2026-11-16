"""Metablock builder for qualities >= 10 (MIN_QUALITY_FOR_HQ_BLOCK_SPLITTING).

Port of the BrotliBuildMetaBlock part of c/enc/metablock.c (including
ComputeDistanceCost and RecomputeDistancePrefixes).  The greedy builder,
BrotliInitDistanceParams, BlockSplit / MetaBlockSplit and
BrotliOptimizeHistograms live in metablock.py (shared encoder core).
"""

from .bit_cost import brotli_population_cost
from .block_splitter import brotli_split_block
from .cluster import brotli_cluster_histograms
from .command import command_restore_distance_code
from .histogram import (brotli_build_histograms_with_context,
                        histogram_command, histogram_distance,
                        histogram_literal)
from .metablock import brotli_init_distance_params
from .prefix import prefix_encode_copy_distance

BROTLI_MAX_NPOSTFIX = 3
BROTLI_LITERAL_CONTEXT_BITS = 6
BROTLI_DISTANCE_CONTEXT_BITS = 2


def _new_distance_params(template, npostfix, ndirect, large_window):
    """BrotliInitDistanceParams on a fresh copy of ``template``."""
    p = template.copy()
    brotli_init_distance_params(p, npostfix, ndirect, large_window)
    return p


def _recompute_distance_prefixes(cmds, num_commands, orig_params, new_params):
    """RecomputeDistancePrefixes."""
    if (orig_params.distance_postfix_bits ==
            new_params.distance_postfix_bits and
            orig_params.num_direct_distance_codes ==
            new_params.num_direct_distance_codes):
        return
    for i in range(num_commands):
        cmd = cmds[i]
        if (cmd.copy_len & 0x1FFFFFF) and cmd.cmd_prefix >= 128:
            cmd.dist_prefix, cmd.dist_extra = prefix_encode_copy_distance(
                command_restore_distance_code(cmd, orig_params),
                new_params.num_direct_distance_codes,
                new_params.distance_postfix_bits)


def _compute_distance_cost(cmds, num_commands, orig_params, new_params, tmp):
    """ComputeDistanceCost: returns (ok, cost)."""
    equal_params = False
    extra_bits = 0.0
    tmp.clear()
    if (orig_params.distance_postfix_bits ==
            new_params.distance_postfix_bits and
            orig_params.num_direct_distance_codes ==
            new_params.num_direct_distance_codes):
        equal_params = True
    data = tmp.data
    count = 0
    max_distance = new_params.max_distance
    ndirect = new_params.num_direct_distance_codes
    npostfix = new_params.distance_postfix_bits
    for i in range(num_commands):
        cmd = cmds[i]
        if (cmd.copy_len & 0x1FFFFFF) and cmd.cmd_prefix >= 128:
            if equal_params:
                dist_prefix = cmd.dist_prefix
            else:
                distance = command_restore_distance_code(cmd, orig_params)
                if distance > max_distance:
                    tmp.total_count += count
                    return False, 0.0
                dist_prefix, _ = prefix_encode_copy_distance(
                    distance, ndirect, npostfix)
            data[dist_prefix & 0x3FF] += 1
            count += 1
            extra_bits += dist_prefix >> 10
    tmp.total_count += count
    return True, brotli_population_cost(tmp) + extra_bits


def brotli_build_meta_block(ringbuffer, pos, mask, params, prev_byte,
                            prev_byte2, cmds, num_commands,
                            literal_context_mode, mb):
    """BrotliBuildMetaBlock.

    ``params.dist`` may be replaced by the best distance parameters found and
    the commands' distance prefixes are recomputed accordingly; ``mb`` (a
    MetaBlockSplit) is filled in."""
    # Histogram ids need to fit in one byte.
    k_max_number_of_histograms = 256
    literal_context_multiplier = 1
    ndirect_msb = 0
    check_orig = True
    best_dist_cost = 1e99
    orig_params = params.dist
    tmp = histogram_distance()

    for npostfix in range(BROTLI_MAX_NPOSTFIX + 1):
        while ndirect_msb < 16:
            ndirect = ndirect_msb << npostfix
            new_params = _new_distance_params(orig_params, npostfix, ndirect,
                                              params.large_window)
            if (npostfix == orig_params.distance_postfix_bits and
                    ndirect == orig_params.num_direct_distance_codes):
                check_orig = False
            ok, dist_cost = _compute_distance_cost(
                cmds, num_commands, orig_params, new_params, tmp)
            if not ok or dist_cost > best_dist_cost:
                break
            best_dist_cost = dist_cost
            params.dist = new_params
            ndirect_msb += 1
        if ndirect_msb > 0:
            ndirect_msb -= 1
        ndirect_msb //= 2
    if check_orig:
        _, dist_cost = _compute_distance_cost(cmds, num_commands, orig_params,
                                              orig_params, tmp)
        if dist_cost < best_dist_cost:
            # NB: currently unused; uncomment when more param tuning is added.
            # best_dist_cost = dist_cost;
            params.dist = orig_params
    _recompute_distance_prefixes(cmds, num_commands, orig_params, params.dist)

    brotli_split_block(cmds, num_commands, ringbuffer, pos, mask, params,
                       mb.literal_split, mb.command_split, mb.distance_split)

    literal_context_modes = None
    if not params.disable_literal_context_modeling:
        literal_context_multiplier = 1 << BROTLI_LITERAL_CONTEXT_BITS
        literal_context_modes = \
            [literal_context_mode] * mb.literal_split.num_types

    literal_histograms_size = \
        mb.literal_split.num_types * literal_context_multiplier
    literal_histograms = [histogram_literal()
                          for _ in range(literal_histograms_size)]

    distance_histograms_size = \
        mb.distance_split.num_types << BROTLI_DISTANCE_CONTEXT_BITS
    distance_histograms = [histogram_distance()
                           for _ in range(distance_histograms_size)]

    mb.command_histograms = [histogram_command()
                             for _ in range(mb.command_split.num_types)]

    brotli_build_histograms_with_context(
        cmds, num_commands, mb.literal_split, mb.command_split,
        mb.distance_split, ringbuffer, pos, mask, prev_byte, prev_byte2,
        literal_context_modes, literal_histograms, mb.command_histograms,
        distance_histograms)

    mb.literal_context_map = \
        [0] * (mb.literal_split.num_types << BROTLI_LITERAL_CONTEXT_BITS)

    # mb->literal_histograms_size = out_size (the list is truncated to it).
    mb.literal_histograms = []
    brotli_cluster_histograms(
        literal_histograms, literal_histograms_size,
        k_max_number_of_histograms, mb.literal_histograms,
        mb.literal_context_map)

    if params.disable_literal_context_modeling:
        # Distribute assignment to all contexts.
        i = mb.literal_split.num_types
        while i != 0:
            i -= 1
            for j in range(1 << BROTLI_LITERAL_CONTEXT_BITS):
                mb.literal_context_map[(i << BROTLI_LITERAL_CONTEXT_BITS) +
                                       j] = mb.literal_context_map[i]

    distance_context_map_size = \
        mb.distance_split.num_types << BROTLI_DISTANCE_CONTEXT_BITS
    mb.distance_context_map = [0] * distance_context_map_size

    # mb->distance_histograms_size = out_size (the list is truncated to it).
    mb.distance_histograms = []
    brotli_cluster_histograms(
        distance_histograms, distance_context_map_size,
        k_max_number_of_histograms, mb.distance_histograms,
        mb.distance_context_map)
