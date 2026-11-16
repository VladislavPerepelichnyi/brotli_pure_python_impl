# Findings in the reference implementation (google/brotli v1.2.1)

Collected while porting `c/` to pure Python. Line numbers refer to tag v1.2.1
(commit `db24a98`). Full explanations and reproducers are in
[`notes/decoder.md`](notes/decoder.md), [`notes/enc-q5.md`](notes/enc-q5.md),
[`notes/enc-q11.md`](notes/enc-q11.md) and the crafted test cases in `tests/`.

**None of these makes the C library produce wrong output.** The port reproduces the
C behaviour in every case, so its output stays byte-identical to the reference.

## Real defects

| # | Location | Finding | Effect |
|---|----------|---------|--------|
| 1 | `c/enc/compress_fragment_two_pass.c:359` | Copy-paste bug: the hash for position `ip-1` is computed with `HashBytesAtOffset(..., 0, ...)`, whereas line 418 and `compress_fragment.c` use offset 2. | Quality 1 misses matches; fixing it shrinks output by 0.22 % (lgwin 10) to 1.16 % (lgwin 15) on the testdata files. |
| 2 | `c/common/constants.h:70-73` | `BROTLI_NUM_DISTANCE_SYMBOLS` passes its arguments to `BROTLI_DISTANCE_ALPHABET_SIZE` in the wrong order. | An undefined shift: GCC evaluates it to 19 instead of the documented 1128. Currently unused. |
| 3 | `c/common/transform.c:179-193` (`ToUpperCase`), uppercase-all loop at about line 264 | Uppercase transforms can write 1–2 bytes past the end of the word. This happens for 10 dictionary words, e.g. len 4 idx 436 `b"zh:\xe5"` with transforms 44/68/83. | A latent out-of-bounds write. It is harmless only because of the decoder ring buffer's slack (`kRingBufferWriteAheadSlack`, `decode.c:45`). |
| 4 | `c/dec/decode.c:2274` | The guard `len == 0 && distance_code <= 120` can never fire with the built-in dictionary: emptying transforms need distance > 34816. | Zero-length dictionary "copies" are always accepted (RFC 7932 allows them); the check is dead code. |
| 5 | `c/dec/decode.c:585/642/666/706/911` | `space` is unsigned and wraps for an over-subscribed code-length code; decoding continues and fails only at the end. | In streaming mode, a truncated over-subscribed code reports "needs more input" instead of an error. |
| 6 | H58/H68 SIMD hashers (`c/enc/hash_longest_match*_simd_inc.h`) | The `uint16` bucket counter wraps after 65,536 inserts, so the hasher forgets the whole bucket. | Deterministic. Slightly worse matches on very large inputs at q5/q6. |

## Fragile code / latent issues

| # | Location | Finding |
|---|----------|---------|
| 7 | `c/enc/encode.c:942-946` | `ExtendLastCommand` adds the 7-bit copy-length delta without sign extension (unlike `CommandCopyLenCode`). Harmless today. |
| 8 | `c/enc/backward_references_hq.c:468-469` | `continuation` is read from the ring buffer before the bounds check (inverted order). |
| 9 | `c/enc/block_splitter_inc.h:25-26` | `length - stride - 1` would underflow for short inputs; it is safe only because a separate constant guarantees length ≥ 128. |
| 10 | `c/enc/cluster_inc.h` (`BrotliHistogramCombine`) | Reads `pairs[0]` without checking that the queue is non-empty; this relies on an invariant that is never stated. |
| 11 | `c/enc/static_dict.c:526` | Upstream `TODO(lode): check for dist overflow` is still open. |
| 12 | `c/enc/entropy_encode.c` (`OptimizeHuffmanCountsForRle`) | Theoretical `uint32` overflow; also a `uint8_t` cast of `num_types`. |
| 13 | `c/dec/decode.c:1190` | The `FORMAT_BLOCK_SWITCH` error is returned without the `BROTLI_FAILURE()` macro, unlike every other error site. |

## Dead code / inconsistencies

| # | Location | Finding |
|---|----------|---------|
| 14 | `c/enc/literal_cost.c:38-47,62` | `max_utf8` starts at 1, so the `counts[2] < 500` branch is dead and the 3-byte UTF-8 histogram is never used; `in_window` only bounds the bootstrap loop. |
| 15 | `c/enc/block_splitter.c:188-191` | The distance split reuses `kMaxCommandHistograms`; there is no distance-specific limit. |
| 16 | `c/enc/backward_references_hq.c:887` | Uses the constant `MAX_ZOPFLI_LEN_QUALITY_11` instead of `MaxZopfliLen(params)`. |
| 17 | `c/dec/decode.c:1084` | The context-map prefix peeks 5 bits where only 1 can be used. Harmless. |

## API / portability observations

| # | Location | Finding |
|---|----------|---------|
| 18 | `c/common/platform.h:647-652` | `BROTLI_MAX_SIMD_QUALITY` is platform dependent (7 on MSVC x64, 6 on GCC x86-64, undefined on 32-bit), so **native q5–q7 output differs between builds**. This port reproduces GCC/x86-64 (H58/H68 at q5). |
| 19 | `python/_brotli.c:410` | Parameters are parsed as unsigned bytes (`"|bbbb"`): `quality=12` raises `brotli.error`, `-1`/`256` raise `OverflowError`, `5.0` raises `TypeError`, and `True` is silently accepted as 1, although the docstring promises `brotli.error`. The port mirrors this. |
| 20 | `python/brotli.py` | `brotli.compress(data, 5)` sets **mode** = 5, not quality, so it fails with "invalid mode". This follows from the signature but is easy to misuse. |
| 21 | `python/_brotli.c:877` | `brotli.decompress` rejects any bytes after the first complete stream, so two concatenated valid streams fail. This is by design; the port matches it. |
| 22 | `c/enc/encode.c` | One-shot `BrotliEncoderCompress` produces different bytes from the streaming API used by the Python binding and the CLI: empty input becomes `0x06`, and it has an uncompressed fallback that the stream API lacks. The port follows the streaming API. |
| 23 | q10/q11 floating point | Exactness assumes a build without FMA contraction (true for the default x86-64 build); an FMA-enabled build could round some double expressions differently. |
