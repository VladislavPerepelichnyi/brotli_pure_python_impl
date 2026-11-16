# brotli_pure_python_impl

A pure-Python (standard library only) port of Google's Brotli compression format.
It follows the reference C implementation [google/brotli](https://github.com/google/brotli)
at tag **v1.2.1**, function by function.

```python
import brotli_pure
packed = brotli_pure.compress(data, quality=5)     # or quality=11 (the default)
assert brotli_pure.decompress(packed) == data
```

The API mirrors the C-backed `brotli` PyPI module:

- `compress(data, mode=MODE_GENERIC, quality=11, lgwin=22, lgblock=0)`
- `decompress(data)`
- `error`

## Status

| Function | Status |
|----------|--------|
| `decompress(data)` | Complete RFC 7932 decoder. Same output and same accept/reject decisions as `brotli.decompress`, checked on about 800k differential fuzz cases. |
| `compress(data, quality=5)` | **Byte-identical** to `brotli.compress(..., quality=5)` for lgwin 10–24 and all modes. |
| `compress(data, quality=11)` | **Byte-identical** to `brotli.compress(..., quality=11)` for lgwin 10–24 and all modes. |
| other qualities (0–4, 6–10) | Also ported and byte-identical. |

"Byte-identical" means the same output as the reference Python binding built from
v1.2.1 with GCC on x86-64. Native q5–q7 output itself depends on the platform; see
[docs/UPSTREAM_FINDINGS.md](docs/UPSTREAM_FINDINGS.md) #18.

Not supported: custom, shared or compound dictionaries; metadata-block emission; and
large-window (lgwin > 24) streams. The reference Python binding does not expose these
either.

## Layout

| Path | Port of |
|------|---------|
| `brotli_pure/common/` | `c/common/`: dictionary, transforms, context tables, constants |
| `brotli_pure/dec/`    | `c/dec/`: bit reader, Huffman tables, prefix tables, decoder |
| `brotli_pure/enc/`    | `c/enc/`: one module per C file (hashers, Zopfli, block splitter, clustering, metablock, bit stream, `encode.py` state machine) |
| `tools/`              | generators for the large tables (from the C sources), C harnesses used for verification, branch test runner |
| `tests/`              | functional tests (pytest + hypothesis); see [tests/README.md](tests/README.md) |
| `docs/ENCODER_API.md` | internal encoder conventions |
| `docs/UPSTREAM_FINDINGS.md` | bugs and questionable behaviour found in the reference C code |
| `docs/notes/`         | per-component porting notes |

## Tests

```sh
python3 -m venv .venv
.venv/bin/pip install pytest hypothesis
# oracle: the C reference Python binding built from google/brotli v1.2.1
git clone --depth 1 --branch v1.2.1 https://github.com/google/brotli.git ../brotli_reference
.venv/bin/pip install ../brotli_reference

.venv/bin/python -m pytest tests                         # fast run (~2 min)
BROTLI_PURE_SLOW=1 .venv/bin/python -m pytest tests      # full run (large inputs, all lgwin)
```

Tests that need the oracle are skipped when it is not installed. The reference test
vectors are taken from `../brotli_reference/tests/testdata`; set `BROTLI_TESTDATA` to
use a different location.
