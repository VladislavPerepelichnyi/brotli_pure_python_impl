# Decoder notes (`brotli_pure.decompress`)

Port of `c/dec/*` and `c/common/transform.c` from google/brotli v1.2.1.

## File map

| Python | C |
| --- | --- |
| `brotli_pure/common/transforms.py` | `c/common/transform.c`, `transform.h` (121 RFC transforms, `ToUpperCase`, `Shift`, `BrotliTransformDictionaryWord`) |
| `brotli_pure/dec/bit_reader.py` | `c/dec/bit_reader.h`, `bit_reader.c` |
| `brotli_pure/dec/huffman.py` | `c/dec/huffman.c`, `huffman.h` (8-bit root tables + 2nd-level tables, simple codes, code-length codes) |
| `brotli_pure/dec/prefix.py` | `c/dec/prefix.c` (`kCmdLut`, built like `BrotliDecoderInitCmdLut`; verified equal to `prefix_inc.h`), `_kBrotliPrefixCodeRanges` |
| `brotli_pure/dec/state.py` | `c/dec/state.h`, `state.c` |
| `brotli_pure/dec/decode.py` | `c/dec/decode.c` |

## Design decisions

* **Straight-line driver.** `BrotliDecoderDecompressStream` is a resumable
  state machine; with the whole input available every stage runs to
  completion (`_decompress_stream`).  The individual algorithms are the C
  ones (same function names in snake_case).
* **Linear output instead of a ring buffer.** All ring-buffer reads in C are
  limited to `distance <= max_distance = min(bytes produced, 2**WBITS - 16)`
  and to the two context bytes before `pos` (zero-initialised before the
  first byte), so the ring buffer contents at those positions always equal a
  linear output buffer.  Hence `max_distance = min(len(out), max_backward)`.
* **Bit reader / truncation.** The accumulator is refilled 4 bytes at a
  time; reading past the end yields zero bits.  Consuming bits past the end
  is detected on each refill, at byte-alignment points and at the end of the
  stream.  This is exactly the condition under which C returns
  `NEEDS_MORE_INPUT`, which the binding reports as an error.
* **Early failure.** When an insert/copy/dictionary word makes
  `meta_block_remaining_len` negative, C continues and fails at the next
  check (`WriteRingBuffer`: `FORMAT_BLOCK_LENGTH_1`, or `METABLOCK_DONE`:
  `FORMAT_BLOCK_LENGTH_2`); every path from there fails, so we fail
  immediately.  This also bounds the work for corrupt inputs.
* **Error semantics of the oracle** (`python/_brotli.c`, `brotli_decompress`):
  success requires `BROTLI_DECODER_RESULT_SUCCESS` *and* `available_in == 0`.
  So truncated input, any trailing byte after the last meta-block (including
  a second concatenated stream), non-zero padding bits (before
  uncompressed/metadata blocks and after the last meta-block), and
  large-window streams (`WBITS` signature `0010001`, the binding never sets
  `BROTLI_DECODER_PARAM_LARGE_WINDOW`) all raise `BrotliError`.
  Non-buffer arguments raise `TypeError`, like the binding.
* **Safety net.** Internal Python exceptions (`IndexError`, ...) are converted
  into `BrotliError`; tests run with `BROTLI_PURE_STRICT=1`, which disables
  the conversion, to make sure none occurs.

## Verification

`tests/test_decoder_dev.py`: all reference vectors, the 45 Java `SynthTest`
streams, oracle round trips (all qualities x windows 10..24 x modes, lgblock,
multi-MB multi-meta-block inputs), hand-crafted streams (window bits 10..24,
metadata incl. last-metadata, padding bits, exuberant nibbles, every
transform for words of every length, window-edge distances), mutation fuzz
and structure-aware fuzz (valid random headers with random Huffman codes /
context maps / block types / NPOSTFIX / NDIRECT followed by random command
bits; every prefix is compared).  Additional offline runs: 200k mutation
cases and ~600k structured cases, no discrepancy.

## Observations about the reference implementation

1. **`c/dec/decode.c:2274` -- the zero-length-word check is dead code for the
   built-in dictionary.**
   `if (len == 0 && s->distance_code <= 120) return FORMAT_TRANSFORM;`
   With the RFC dictionary a transformed word is empty only for transforms
   such as OmitFirst4..9 / OmitLast4..9 applied to short words; the smallest
   such transform index is 34, and words of length 4..9 have
   `size_bits_by_length >= 10`, so the distance is at least
   `max_distance + 1 + (34 << 10) > 120`.  The check therefore never fires,
   and zero-length dictionary "copies" (e.g. word "time" with transform 64,
   OmitLast9) are silently accepted, producing nothing.  RFC 7932 does not
   forbid them, so this is not a conformance bug, but the guard (120 is the
   maximal NDIRECT) does not protect against what it apparently intends to.
   Such commands still consume input bits (a zero-bit distance code can only
   yield distances <= max(max_distance, 16) + 3 or direct distances <= 120,
   i.e. transform index 0..3), so no zero-progress loop is possible with the
   built-in dictionary; with custom shared dictionaries/transforms it might be.
   Our port keeps the check verbatim.

2. **`c/dec/decode.c:585, 642, 666, 706, 911` -- over-subscribed code lengths
   are detected late.**  `space` is an unsigned `brotli_reg_t`; when a code
   length over-subscribes the Kraft sum, `space -= 32768 >> code_len`
   wraps around to a huge value, the loop condition `space > 0` stays true and
   the decoder keeps reading code lengths until `alphabet_size` symbols were
   read, and only then fails with `FORMAT_HUFFMAN_SPACE`.  For the one-shot
   API the result is the same (error), but in streaming mode an
   over-subscribed, truncated code yields `NEEDS_MORE_INPUT` instead of an
   error.  The port stops at `space <= 0` (equivalent outcome).

3. **`c/dec/decode.c:1084` -- context map prefix peeks 5 bits.**
   `BrotliSafeGetBits(br, 5, ...)` is required even when only the 1-bit RLE
   flag is used.  Harmless: the following `ReadHuffmanCode` always needs at
   least 4 more bits, so no valid stream is rejected.

4. **`c/dec/decode.c:1190` -- inconsistent error reporting.**
   `DecodeBlockTypeAndLength` returns `BROTLI_DECODER_ERROR_FORMAT_BLOCK_SWITCH`
   without `BROTLI_FAILURE()` (no `BROTLI_DUMP()` in debug builds), unlike
   every other error site.  It is only reachable for streams that are already
   invalid (more than 2**24 symbols of a single-block-type category, which
   requires overrunning the meta-block length).

5. **`python/_brotli.c:877` -- concatenated streams are rejected.**
   `brotli.decompress` fails if any input remains after the first complete
   stream (`available_in > 0`), so `decompress(a + b)` of two valid streams
   raises.  Not a bug, but worth knowing; the port matches it.

6. **RFC 7932 vs. C transform table**: identical (checked by decoding every
   transform of words of every length through the oracle).  The C header
   also defines `SHIFT_FIRST` / `SHIFT_ALL` transform types that do not exist
   in RFC 7932 (used only by shared dictionaries); they are ported
   (`transforms.shift`) but unreachable through the public API.
