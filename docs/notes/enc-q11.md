# enc-q11: quality 10 / 11 encoder components

Port of the reference (google/brotli v1.2.1) quality-10/11 encoder parts:

| module | C source |
|---|---|
| `enc/static_dict.py` (+ `static_dict_lut.bin`, `tools/gen_static_dict_lut.py`) | `static_dict.c`, `static_dict_lut.h`, `static_dict_lut_inc.h` |
| `enc/hash_to_binary_tree.py` (H10) | `hash_to_binary_tree_inc.h` (+ BackwardMatch helpers of `hash.h`) |
| `enc/literal_cost.py` | `literal_cost.c` |
| `enc/utf8_util.py` | `utf8_util.c` |
| `enc/backward_references_hq.py` | `backward_references_hq.c` |
| `enc/cluster.py` | `cluster.c`, `cluster_inc.h` |
| `enc/block_splitter.py` | `block_splitter.c`, `block_splitter_inc.h` |
| `enc/metablock_hq.py` | `BrotliBuildMetaBlock`, `ComputeDistanceCost`, `RecomputeDistancePrefixes` of `metablock.c` |

## Verification

* `tools/enc_q11_harness/*.c` call the internal C functions directly
  (`BrotliFindAllStaticDictionaryMatches`, `FindAllMatchesH10`,
  `BrotliCreate{Hq,}ZopfliBackwardReferences`, `BrotliBuildMetaBlock`) on a
  linear buffer; their outputs were compared line by line with the Python
  ports on text, binary, random, repetitive and dictionary-heavy inputs
  (several windows, q10/q11, GENERIC and FONT distance parameters).  A small
  subset is kept as golden data for `tests/test_enc_q11_dev.py`.
* End to end: `brotli_pure.compress(quality=11/10)` is compared byte for byte
  with the C oracle (`Compressor.process + finish`).

## Floating point

`backward_references_hq.c` keeps the cost model (`cost_cmd_`, `cost_dist_`,
`literal_costs_`, `min_cost_cmd_`), node costs and all arithmetic on them in
`float`.  With SSE (x86-64, `FLT_EVAL_METHOD == 0`, no FMA in the default
target) each float operation rounds to binary32; Python emulates every
`a op b` as `f32(a op b)` (double rounding of a single +,- of two binary32
values through binary64 is exact).  `f32` = store/load through a one-element
`array('f')`.  Mixed expressions keep the C evaluation order, e.g.
`base_cost = (start_costdiff + (float)GetInsertExtra(inscode)) + literal_cost`
is two rounded additions.  Everything else (literal cost estimation before
the final `(float)` cast, block splitting, clustering, population costs) is
`double` and maps 1:1 to Python floats.

Note: an FMA-enabled build (e.g. `-march=haswell` with GCC's default
`-ffp-contract=fast`) could contract some `double` expressions such as
`0.35 + multiplier * (double)i` in `literal_cost.c` or the
`ClusterCostDiff` sum of products, which would change results in rare cases;
the oracle (default x86-64 target) does not do that.

## Implementation notes

* BackwardMatch arrays are two parallel lists (`distance`,
  `length_and_code`); ZopfliNode arrays are struct-of-arrays with a single
  `u` list mirroring the C union (float cost, later int shortcut / next).
* In `UpdateNodes` the per-length cost only depends on the copy length code,
  so it is computed once per code run (identical values, fewer float
  roundings in Python).
* H10's forest is an `array('I')` of `2 * num_nodes` entries as in C (for
  `lgwin = 24` and non-one-shot input that is 128 MiB, same as C).
* Shared / compound / contextual dictionaries are not supported (not reachable
  through the `compress()` API): `gap == 0`, `num_chunks == 0`,
  `context_based == 0`.

## Questionable behaviour / possible upstream issues found in the C code

1. `c/enc/backward_references_hq.c:468-469` (`UpdateNodes`):
   `uint8_t continuation = ringbuffer[cur_ix_masked + best_len];` is read
   *before* the bounds check `if (cur_ix_masked + best_len > ringbuffer_mask)
   break;`.  If the check fires, the read already happened one byte past the
   ring buffer mask.  It is harmless in practice (the ring buffer has a tail
   copy plus 7 slack bytes and the value is unused on that path), but the
   order is inverted with respect to the intent.

2. `c/enc/literal_cost.c:38-47` (`DecideMultiByteStatsLevel`): `max_utf8` is
   initialised to 1 ("should be 2, but 1 compresses better") and the
   following `if (counts[2] < 500) max_utf8 = 1;` is therefore dead code;
   the function can only return 0 or 1, so the third (3-byte UTF-8) histogram
   slice in `EstimateBitCostsForLiteralsUTF8` is never used, and the
   `counts[2]` statistic is computed for nothing.  Also `in_window`
   (line 62) is computed but only used to bound the bootstrap loop.

3. `c/enc/block_splitter_inc.h:25-26` (`InitialEntropyCodes`):
   `pos = length - stride - 1` underflows (size_t) if
   `length <= stride`; it is only safe because callers guarantee
   `length >= kMinLengthForBlockSplitting (128) > stride (70/40)`.  Fragile
   coupling between unrelated constants.

4. `c/enc/block_splitter.c:188-191`: the distance split uses
   `kMaxCommandHistograms` (50) as its histogram cap; there is no
   `kMaxDistanceHistograms`.  Probably intentional, but it looks like a
   copy/paste of the command parameters (the symbol/stride constants do have
   distance-specific versions).

5. `c/enc/cluster_inc.h` (`BrotliHistogramCombine`): the loop reads
   `pairs[0].cost_diff` without checking `num_pairs > 0`.  It is safe because
   `BrotliCompareAndPushToQueue` always accepts the first pair when the queue
   is empty (threshold `1e99`), so the queue is non-empty whenever
   `num_clusters >= 2`; but the invariant is implicit.

6. `c/enc/static_dict.c:526` (`BrotliFindAllStaticDictionaryMatches`,
   multi-dictionary branch): already marked upstream
   `TODO(lode): check for dist overflow`; not reachable with the built-in
   dictionary.

7. `c/enc/backward_references_hq.c:887`: the HQ match collection uses the
   constant `MAX_ZOPFLI_LEN_QUALITY_11` instead of `MaxZopfliLen(params)`;
   consistent only because the HQ path is exclusively used at quality 11.
