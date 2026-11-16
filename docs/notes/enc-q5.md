# enc-q5: notes on the reference C encoder (google/brotli v1.2.1)

Findings made while porting the shared encoder core and the quality-5 path.
File/line references are relative to `brotli_reference/c/`.

## Suspected bugs / questionable behaviour

1. **`BROTLI_NUM_DISTANCE_SYMBOLS` has its macro arguments swapped**
   (`common/constants.h:70-73`).
   `BROTLI_DISTANCE_ALPHABET_SIZE(NPOSTFIX, NDIRECT, MAXNBITS)` is invoked as
   `BROTLI_DISTANCE_ALPHABET_SIZE(BROTLI_MAX_NDIRECT, BROTLI_MAX_NPOSTFIX,
   BROTLI_LARGE_MAX_DISTANCE_BITS)`, i.e. with NDIRECT and NPOSTFIX
   exchanged.  The expansion contains `62U << (120 + 1)`, an undefined shift
   (GCC warns `-Wshift-count-overflow`; on x86-64/GCC the macro evaluates to
   **19**, not the 1128 claimed by the comment on line 70).  The macro is
   currently unused anywhere in the tree, so this is latent, but any future
   use would silently get a wrong alphabet size.  Verified with a small C
   program printing the macro (19) next to the correctly ordered expression
   (1128).

2. **`ExtendLastCommand` adds the copy-length-code delta without sign
   extension** (`enc/encode.c:942-946`).
   `Command.copy_len_` stores a signed 7-bit `len_code - len` delta in its top
   bits; `CommandCopyLenCode` (`enc/command.h:179-183`) sign-extends it, but
   `ExtendLastCommand` recomputes the command prefix with
   `(int)(copy_len_ & 0x1FFFFFF) + (int)(copy_len_ >> 25)`, which would turn a
   delta of -1 into +127.  It is harmless in v1.2.1 because only commands
   whose distance is a real (non-dictionary) backward distance are extended,
   and those always have delta 0; dictionary commands (the only ones with a
   non-zero, and then positive, delta) are never extended.  Still, the two
   decodings of the same field are inconsistent.  The Python port reproduces
   the C expression verbatim.

3. **Output depends on the build platform / compiler** (`common/platform.h:647-652`,
   `enc/quality.h:167-182`).  `BROTLI_MAX_SIMD_QUALITY` is 7 on MSVC x64
   (`_M_X64`), 6 on other 64-bit targets with `__builtin_ctzll`, and undefined
   on 32-bit targets.  `ChooseHasher` therefore picks H58/H68 (tag-SIMD
   hashers) or H5/H6 depending on the platform, and those hashers find
   different matches, so the compressed bytes for qualities 5-7 differ
   between e.g. Linux x86-64 GCC, Windows x64 MSVC and 32-bit builds (all
   outputs are valid).  The oracle used here (x86-64 Linux, GCC) uses
   H58/H68 at q5/q6 and H5/H6 at q7; the port hard-codes
   `BROTLI_MAX_SIMD_QUALITY = 6` (`brotli_pure/enc/quality.py`).  (Also note
   the `#define`s sit inside the body of `BrotliSuppressUnusedFunctions`,
   lines 647-653 -- legal, but easy to miss.)

4. **Tag-SIMD hashers forget a whole bucket every 65536 insertions**
   (`enc/hash_longest_match_simd_inc.h:226-230`,
   `enc/hash_longest_match64_simd_inc.h:253-257`).  The per-bucket counter
   `num_[key]` is a `uint16_t` that starts at 65535 and is decremented on
   every store; `n = 65535 - num[key]` is used as the number of valid slots.
   After exactly 65536 stores into one bucket the counter wraps back to 65535,
   `n` becomes 0 and `mask` becomes 0, so all (still valid) candidates of that
   bucket are ignored until the bucket is refilled.  Deterministic and only a
   small compression loss on degenerate inputs (reproduced by the
   `test_hasher_bucket_counter_wraparound` test input, which the port matches
   byte-for-byte); the classic H5/H6 hashers have the analogous wrap of their
   incrementing `uint16_t num_`.

5. **Stale comment / odd casts** (cosmetic):
   `enc/hash_longest_match_simd_inc.h:216` and siblings say "we require
   matches of len >4" while the code requires `len >= 4`;
   `enc/metablock_inc.h:128,130` cast `split->num_types` (a `size_t`) to
   `uint8_t` when storing it as a histogram index -- safe only because the
   preceding `num_types < BROTLI_MAX_NUMBER_OF_BLOCK_TYPES` check bounds it
   to 255.

6. **`BrotliOptimizeHuffmanCountsForRle` does 24.8 fixed-point math in
   `uint32_t`** (`enc/entropy_encode.c:324,329,352,354`): `256 * counts[i]`
   overflows for counts >= 2^24.  With metablocks of at most 2^24 bytes and
   the `nonzero_count >= 16` precondition this cannot happen in practice, so
   it is only a theoretical limit.  The port reproduces the uint32 wrap.

## Observations relevant to byte-exact reproduction

* `brotli.compress` (Python binding) and the `brotli` CLI use the streaming
  API (`BrotliEncoderCompressStream`), whereas the one-shot
  `BrotliEncoderCompress` (`enc/encode.c:1271`) has two differences: empty
  input is encoded as the single byte `0x06` (window 16) instead of the
  header for the requested window (e.g. `0x3b` for lgwin 22), and it falls
  back to `MakeUncompressedStream` when the result exceeds
  `BrotliEncoderMaxCompressedSize`.  So "the" brotli output for given
  parameters depends on which C entry point is used.  The port follows the
  binding (stream) behaviour.
* The size hint is not set by the binding; `UpdateSizeHint` derives it from
  the input that is available at the first `EncodeData` call.  With
  `process(data)` + `finish()` this is always `min(len(data), 2^30)`
  (for inputs >= one block the first `EncodeData` happens during PROCESS
  with all remaining input still pending).  The size hint selects H68 over
  H58 (>= 1 MiB with lgwin >= 19) and enables the 13-context "complex" static
  context map (>= 1 MiB), so splitting the same input into several
  `process()` calls of < 64 KiB can change the output.
* Inputs that are an exact multiple of the input block size (64 KiB at q5)
  are fully encoded during PROCESS (with `is_last = 0`) and FINISH then emits
  a separate empty last meta-block (`ISLAST, ISEMPTY`).
* The hashers read up to 7 bytes past the current input end; the encoder
  zeroes those bytes during the first ring-buffer lap
  (`CopyInputToRingBuffer`), so the output is deterministic.  The port
  allocates zeroed buffers, which is equivalent.
* Where C leaves memory uninitialised (hasher buckets, block-split arrays,
  histogram slots) the port initialises it; all such memory is provably
  masked/cleared before it is read, so this does not change results.
