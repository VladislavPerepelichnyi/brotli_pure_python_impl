# brotli_pure_python_impl

A pure-Python (stdlib only) port of Google's Brotli compression format,
following the reference C implementation [google/brotli](https://github.com/google/brotli)
at tag **v1.2.1**.

```python
import brotli_pure
packed = brotli_pure.compress(data, quality=5)    # or quality=11 (default)
assert brotli_pure.decompress(packed) == data
```

## Layout

| Path | Port of |
|------|---------|
| `brotli_pure/common/` | `c/common/` (dictionary, transforms, constants) |
| `brotli_pure/dec/`    | `c/dec/` (decoder) |
| `brotli_pure/enc/`    | `c/enc/` (encoder; quality 5 and 11 paths) |
| `tests/`              | functional tests (pytest) |

## Tests

```sh
python3 -m venv .venv
.venv/bin/pip install pytest hypothesis
# optional oracle: the C reference Python binding built from google/brotli v1.2.1
.venv/bin/pip install path/to/google/brotli
.venv/bin/python -m pytest
```
