# brotli_pure functional test-suite

Functional (not performance) tests for `brotli_pure`, the pure-Python port of
google/brotli v1.2.1. The suite tests against the format spec (RFC 7932) and
against the **oracle**, the C-backed `brotli` Python module built from the
same v1.2.1 sources. It does not test against assumptions about the
implementation.

## Running

```sh
PY=/home/vlad/thesis/wasm/pure_python_impl/brotli_pure_python/.venv/bin/python   # has brotli, pytest, hypothesis

$PY -m pytest tests                       # default fast run
BROTLI_PURE_SLOW=1 $PY -m pytest tests    # full run (or: --run-slow)
$PY -m pytest tests -m "not oracle"       # only what works without the oracle
BROTLI_PURE_IMPL=/path/to/other/checkout $PY -m pytest tests   # test another tree
tools/run_branch_tests.sh decoder [pytest args...]             # test a branch's committed state
```

Results come out as pass, fail or **skip**. Unimplemented parts are skipped
with an explicit reason, such as `brotli_pure.dec.decode not implemented yet
(module missing)` or `quality 11 not supported (brotli_pure.error: ...)`. They
never count as a pass. Set `BROTLI_PURE_STRICT=1` to turn those skips into
failures for qualities 5 and 11 and for the decoder. Use it on the merged
main branch.

### Environment knobs

| variable | meaning |
|---|---|
| `BROTLI_PURE_SLOW=1` | also run tests marked `slow`: large inputs, full parameter matrices, exhaustive fuzzing and the q11 corpus |
| `BROTLI_PURE_STRICT=1` | "not implemented" fails instead of skipping |
| `BROTLI_PURE_IMPL=<dir>` | import `brotli_pure` from `<dir>` (another worktree or snapshot) instead of this checkout |
| `BROTLI_TESTDATA=<dir>` | reference test vectors (default `../../brotli_reference/tests/testdata`). Small vectors are also copied into `tests/data/` |
| `BROTLI_REFERENCE=<dir>` | reference checkout, used to find `c/common/dictionary.bin` for the crafted-stream model. Falls back to the package copy, verified by SHA-256 |
| `BROTLI_PURE_HYPOTHESIS_PROFILE` | `fast` (15 examples), `default` (40, derandomized) or `thorough` (400, the default when slow) |
| `BROTLI_PURE_TIMEOUT_SCALE` | multiplier for the hang-detection deadlines (SIGALRM) |

Markers (declared in `pyproject.toml`): `slow` and `oracle`. Oracle-dependent
tests skip cleanly when `brotli` cannot be imported.

## What each module covers

| module | coverage |
|---|---|
| `test_api.py` | Exports, signatures and defaults. Buffer types (bytes, bytearray, memoryview, slices, `array`). Return type is `bytes`. The input is not mutated. `TypeError` for non-buffers. Parameter validation value-by-value against the oracle: `brotli_pure.error` where the oracle raises `brotli.error`; Overflow/Value/`brotli_pure.error` where the oracle raises `OverflowError`; `TypeError` for non-integers. Valid edge parameters. Optional qualities (0-4, 6-10) must either round-trip or raise. |
| `test_decompress.py` | All reference vectors (`*.compressed*`). The corpus compressed by the oracle at **every quality 0..11** × lgwin {10,16,22,24} with all modes cycled (full lgwin 10..24 × modes when slow). Medium inputs that cross block boundaries, large inputs (slow), all 256 single bytes, lgblock 16..24, and reference originals re-compressed. |
| `test_decompress_crafted.py` | About 340 hand-crafted streams from `crafted_cases.py`. Every WBITS value and encoding, the large-window marker, empty and metadata-only streams. Uncompressed metablocks with 4/5/6-nibble MLEN and exuberant nibbles. Metadata with every MSKIPBYTES, the reserved bit, non-zero padding, and ISLAST metadata. Simple codes with 1-4 symbols and both tree shapes. Complex codes with HSKIP 0/2/3, RLE codes 16/17, 15-bit codes and incomplete or oversubscribed codes. Every insert and copy length code, and overlapping copies. Implicit, all 16 short and explicit distances, including the initial ring buffer 4/11/15/16. NPOSTFIX 0..3 × NDIRECT, max-distance and max+1 at every WBITS. Static dictionary with **all 121 transforms**, zero-length results, UTF-8 uppercase overruns, references after the window is full, and the "dictionary references don't touch the distance cache" rule. Invalid distances, transforms and MLEN. Truncation at every byte. Random crafted streams. |
| `test_decompress_invalid.py` | Differential against the oracle: truncation at every byte, every single-bit flip (or 400 seeded flips), byte substitutions/insertions/deletions, trailing garbage, all 1-byte inputs, 2-byte inputs (all 65536 when slow) and random strings. Corrupt input must raise **only** `brotli_pure.error` and must never hang. |
| `test_compress_q5.py` | `compress(quality=5)`. Each combination is reported three ways: `oracle_roundtrip`, `own_roundtrip` and `byte_exact` (identical bytes to `brotli.compress`). Covers the small corpus × lgwin {10,16,22,24} with all modes, lgwin 10..24, lgblock variants, inputs crossing 64 KiB and all single bytes. Slow: lgwin × modes, the medium and large corpora. |
| `test_compress_q11.py` | Same as q5 for quality 11, scaled for pure-Python speed. Default: tiny corpus (3 KiB or less), every lgwin on one sample, every mode, lgblock and single bytes. Slow: the small corpus, lgwin × modes, the medium corpus (crossing the 256 KiB block) and large inputs. |
| `test_roundtrip_property.py` | Hypothesis tests. q5/q11 round-trip plus byte-exactness on generated data. Decoding oracle output for random quality/lgwin/mode/lgblock. Differential decoding of mutated streams and random bytes. Random crafted streams checked against the model. |
| `test_harness.py` | Self-checks of the test infrastructure. Every crafted case and 300 random crafted streams agree with the oracle, which validates the expectations. Also checks the bit writer, the distance and length code tables, the transform table, the dictionary, corpus determinism (hash pinned) and that the test vectors are present. |

## Support code

- `support.py`: implementation probes (skip vs strict), oracle access, the
  SIGALRM `deadline`, and `check_decode_matches_oracle()` (the core
  differential property).
- `corpus.py`: deterministic, seeded corpus in tiers `small` / `tiny` /
  `medium` / `large`, plus `single_bytes()` and the reference test vectors.
  Run `python -m tests.corpus` to list it.
- `bitstream.py`: a test-only bit writer and mini-encoder for crafted streams,
  plus an **independent reference model** of decoder output: back-references,
  the distance ring buffer, the dictionary and transforms with C semantics.
- `crafted_cases.py`: the named crafted streams.
- `rfc_transforms.py`: the 121 RFC 7932 transforms. It is generated from the
  reference C by `tools/gen_transforms.py`.
- `compress_common.py`: shared checks for the q5/q11 modules. Each
  compression is run once and cached.

Implementation bugs and oracle quirks that these tests found are recorded in
`tests/FINDINGS.md`.
