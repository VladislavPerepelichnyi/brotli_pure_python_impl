# brotli_pure encoder: internal API and conventions

Reference: google/brotli v1.2.1, `c/enc/`.  One Python module per C file
(template `*_inc.h` files become one module per template, or a class per
instantiation), functions named after their C counterparts in snake_case
(`BrotliPopulationCost` -> `brotli_population_cost`, `InitCommand` ->
`init_command`, ...), each with a docstring naming the C function.

## General conventions

* **Out-parameters** that are pointers to scalars become return values (a tuple
  in C parameter order if there are several).  In/out scalars (e.g.
  `size_t* last_insert_len`, `size_t* num_literals`) are passed in by value
  and the new values are returned.
* **Arrays / structs passed by pointer are mutated in place** (Python lists,
  `bytearray`s or objects).
* **Command arrays** are Python lists of `Command` objects; C's
  `num_commands` is `len(commands)`; functions that "write at
  `commands[num_commands]` and bump `num_commands`" append to the list.
* `MemoryManager*` arguments and OOM handling are dropped.
* Integer widths: Python ints are unbounded; where C relies on wrap-around
  (uint16 counters, `size_t` subtraction, uint32 products) the code masks
  explicitly.  Comments mark those places.
* Floating point: Python `float` is C `double`.  `kBrotliLog2Table` stores
  float32-rounded values (as the C `f`-suffixed literals do).  Modules that
  need C `float` storage must round explicitly (e.g. `array('f')` or
  `fast_log._to_float32`).
* The ring buffer / input `data` is a `bytearray` (`RingBuffer.buffer`, the C
  `buffer_` pointer, i.e. without the 2 leading bytes of `data_`).  Positions
  are absolute; index with `pos & mask` as in C.  The buffer has the C tail
  (copy of the first `tail_size` bytes after `size`) and 7 slack bytes, so
  4/8-byte loads past the masked position behave as in C.

## Modules (shared core)

| module | C source | contents |
|---|---|---|
| `fast_log.py` | fast_log.h/.c | `log2_floor_non_zero(n)`, `fast_log2(v)`, `fast_pow2(v)` (not in v1.2.1, provided anyway), `kBrotliLog2Table` |
| `histogram.py` | histogram.h, histogram_inc.h, histogram.c | `Histogram`, `histogram_literal/command/distance()`, `clear_histograms`, `BlockSplitIterator`, `brotli_build_histograms_with_context` |
| `bit_cost.py` | bit_cost.h, bit_cost_inc.h, bit_cost.c | `brotli_population_cost(h)` (+ `_literal/_command/_distance` aliases), `brotli_bits_entropy(population, size)` (= `bits_entropy`), `shannon_entropy` (old API, not used by v1.2.1; returns `(entropy, total)`) |
| `command.py` | command.h, command.c | `Command`, `init_command`, `new_command`, `init_insert_command`, `new_insert_command`, `command_copy_len`, `command_copy_len_code`, `command_restore_distance_code`, `command_distance_context`, `get_insert_length_code`, `get_copy_length_code`, `combine_length_codes`, `get_length_code`, `get_insert_base/extra`, `get_copy_base/extra`, `kBrotli{Ins,Copy}{Base,Extra}` |
| `prefix.py` | prefix.h | `prefix_encode_copy_distance(distance_code, num_direct_codes, postfix_bits) -> (code, extra_bits)` |
| `params.py` | params.h (+ constants.h distance macros) | `BrotliEncoderParams`, `BrotliHasherParams`, `BrotliDistanceParams`, `brotli_distance_alphabet_size`, `brotli_max_backward_limit`, `BROTLI_*` constants |
| `quality.py` | quality.h | quality constants, `sanitize_params`, `compute_lg_block`, `compute_rb_bits`, `max_metablock_size`, `max_zopfli_len`, `max_zopfli_candidates`, `literal_spree_length_for_sparse_search`, `choose_hasher`, `BROTLI_MAX_SIMD_QUALITY` (=6, as in the x86-64 GCC oracle build) |
| `entropy_encode.py` | entropy_encode.h/.c | `brotli_create_huffman_tree(data, length, tree_limit, tree, depth)` (`tree` scratch is ignored), `brotli_set_depth`, `brotli_optimize_huffman_counts_for_rle(length, counts, good_for_rle)`, `brotli_write_huffman_tree(depth, length, tree, extra_bits) -> tree_size` (appends to lists), `brotli_convert_bit_depths_to_symbols(depth, len, bits)` |
| `hash_base.py` | hash_base.h | `kHashMul32`, `kHashMul64`, `hash14(buf, off)`, `hash15(buf, off)`, `load32le`, `load64le` (struct `unpack_from`; return 1-tuples) |
| `find_match_length.py` | find_match_length.h | `find_match_length_with_limit(s1, i1, s2, i2, limit)` (pointers become `(buffer, offset)` pairs) |
| `hash.py` | hash.h | `HasherCommon`, `Hasher`, `HasherSearchResult`, `BackwardMatch` (+ `init_backward_match`, `init_dictionary_backward_match`, `backward_match_length`, `backward_match_length_code`), score helpers, `prepare_distance_cache`, `search_in_static_dictionary`, `hasher_setup`, `hasher_reset`, `init_or_stitch_to_previous_block`, `MAX_NUM_MATCHES_H10` |
| `encoder_dict.py` | encoder_dict.h/.c | `SharedEncoderDictionary` (`.compound`, `.contextual.dict[i]`, `.max_quality`), `BrotliEncoderDictionary`, `BrotliDictionary`, `brotli_get_dictionary()` |
| `dictionary_hash.py` | dictionary_hash.c/_inc.h | `kStaticDictionaryHashWords`, `kStaticDictionaryHashLengths` (from `dictionary_hash.bin`, see `tools/gen_dictionary_hash.py`) |
| `metablock.py` | metablock.h/.c (greedy + shared parts), metablock_inc.h, block_splitter.h (`BlockSplit`) | `BlockSplit`, `MetaBlockSplit`, `brotli_init_distance_params`, `brotli_calculate_distance_code_limit`, `brotli_build_meta_block_greedy`, `brotli_optimize_histograms`, `BlockSplitter`, `ContextBlockSplitter` |

## Key data structures

### `Command` (command.py)
`__slots__ = ('insert_len', 'copy_len', 'dist_extra', 'cmd_prefix',
'dist_prefix')` -- exactly the C fields without the trailing underscore and
with the same packing (`copy_len` low 25 bits = length, high 7 bits = signed
`len_code - len` delta; `dist_prefix` low 10 bits = symbol, high 6 bits = number
of extra bits).  `init_command(cmd, dist_params, insertlen, copylen,
copylen_code_delta, distance_code)` initialises in place and returns `cmd`;
`new_command(...)` allocates one.

### `Histogram` (histogram.py)
`__slots__ = ('data', 'total_count', 'bit_cost')`; `data` is a list of ints
of the alphabet size; `bit_cost` is initialised (and reset by `clear()`) to
`float('inf')` (C `HUGE_VAL` in `HistogramClear`).  Methods: `clear()`,
`add(val)`, `add_vector(seq, n)`, `add_histogram(other)` (rebinds `data` to
a new list), `copy()`, `assign(other)` (in-place `*self = *other`).  C struct
assignment `a[i] = b` can be written either `a[i] = b.copy()` or
`a[i].assign(b)`.  Sizes: literal 256, command 704,
distance `BROTLI_NUM_HISTOGRAM_DISTANCE_SYMBOLS` = 544.

### `BlockSplit` / `MetaBlockSplit` (metablock.py)
`BlockSplit`: `num_types`, `num_blocks`, `types` (list of ints), `lengths`
(list of ints).  As in C the lists may be longer than `num_blocks`; only the
first `num_blocks` entries are meaningful.
`MetaBlockSplit`: `literal_split`, `command_split`, `distance_split`,
`literal_context_map`, `distance_context_map` (lists; empty list == C size 0),
`literal_histograms`, `command_histograms`, `distance_histograms` (lists of
`Histogram`).  The C `*_size` fields are read-only properties returning the
list lengths, so producers must size the lists exactly
(`len(mb.literal_histograms) == literal_histograms_size`).

### Params
`BrotliEncoderParams`: `mode, quality, lgwin, lgblock, stream_offset,
size_hint, disable_literal_context_modeling, large_window, hasher
(BrotliHasherParams), dist (BrotliDistanceParams), dictionary
(SharedEncoderDictionary)`; `params.copy()` = C struct copy (dictionary
shared).  `BrotliHasherParams`: `type, bucket_bits, block_bits,
num_last_distances_to_check`.  `BrotliDistanceParams`:
`distance_postfix_bits, num_direct_distance_codes, alphabet_size_max,
alphabet_size_limit, max_distance`.

### Dictionary
`params.dictionary.contextual.dict[0]` is the `BrotliEncoderDictionary`
(`words` = `BrotliDictionary` with `data`, `offsets_by_length`,
`size_bits_by_length`; `num_transforms` = 121; `cutoff_transforms_count`,
`cutoff_transforms`; `hash_table_words`, `hash_table_lengths`).  Its
`buckets` / `dict_words` (static_dict_lut.h) are **None**: the quality-11
`static_dict.py` owns that LUT and should use its own tables.
`params.dictionary.compound.total_size == 0` / `num_chunks == 0` always
(no compound dictionaries); `contextual.context_based` is False.

## Hashers (hash.py and per-template modules)

A hasher type N is a class `HN` constructed as `HN(common, params)` (= C
`InitializeHN`; `common` is the `HasherCommon`, whose `params` is the
`BrotliHasherParams`).  Methods (C template functions):

```
hash_type_length()  store_lookahead()
prepare(one_shot, input_size, data)
store(data, mask, ix)
store_range(data, mask, ix_start, ix_end)
stitch_to_previous_block(num_bytes, position, ringbuffer, ringbuffer_mask)
prepare_distance_cache(distance_cache)
find_longest_match(dictionary, data, ring_buffer_mask, distance_cache,
                   cur_ix, max_length, max_backward, dictionary_distance,
                   max_distance, out)       # out: HasherSearchResult, mutated
common                                      # HasherCommon
```

`HasherCommon` fields: `params`, `is_setup`, `is_prepared`,
`dict_num_lookups`, `dict_num_matches`.  The C `Hasher` struct is `Hasher`
with `.common` and `.privat` (the `HN` instance, C `hasher->privat._HN`).
`hasher_setup(hasher, params, data, position, input_size, is_last)` runs
`choose_hasher` and instantiates the class on first use; type 10 is
`hash_to_binary_tree.H10` (lazy import, written by the q11 branch, which also
provides `find_all_matches(...)` etc.).  Ported here: H40/H41/H42
(`hash_forgetful_chain.py`), H58 (`hash_longest_match_simd.py`), H68
(`hash_longest_match64_simd.py`).

`HasherSearchResult`: `len, distance, score, len_code_delta`.
`BackwardMatch`: `distance, length_and_code`.

## Bit writing

The q11-owned modules do not write bits.  Internally (`write_bits.py`,
`brotli_bit_stream.py`) a `BitWriter` object replaces the C
`(size_t* storage_ix, uint8_t* storage)` pair.
