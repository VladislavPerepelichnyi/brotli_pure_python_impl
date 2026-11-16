"""Public API surface of brotli_pure vs the oracle ``brotli`` module.

Signatures and defaults, accepted argument types (bytes / bytearray /
memoryview / other C-contiguous buffers), return types, the error type, and
parameter validation (invalid quality / lgwin / lgblock / mode) compared
value-by-value with what the oracle does.
"""

import array
import inspect

import pytest

import brotli_pure
from brotli_pure.common.errors import BrotliError
from tests import corpus, support


# --------------------------------------------------------------------------
# Static surface (works without any implementation)
# --------------------------------------------------------------------------

def test_exports():
    for name in ("compress", "decompress", "error", "MODE_GENERIC", "MODE_TEXT", "MODE_FONT"):
        assert hasattr(brotli_pure, name), name
    assert set(brotli_pure.__all__) >= {"compress", "decompress", "error"}


def test_mode_constants():
    assert (brotli_pure.MODE_GENERIC, brotli_pure.MODE_TEXT, brotli_pure.MODE_FONT) == (0, 1, 2)


@pytest.mark.oracle
def test_mode_constants_match_oracle():
    o = support.need_oracle()
    assert brotli_pure.MODE_GENERIC == o.MODE_GENERIC
    assert brotli_pure.MODE_TEXT == o.MODE_TEXT
    assert brotli_pure.MODE_FONT == o.MODE_FONT


def test_error_type():
    assert brotli_pure.error is BrotliError
    assert issubclass(brotli_pure.error, Exception)


def test_compress_signature():
    sig = inspect.signature(brotli_pure.compress)
    params = list(sig.parameters.values())
    assert [p.name for p in params[1:]] == ["mode", "quality", "lgwin", "lgblock"]
    defaults = {p.name: p.default for p in params[1:]}
    assert defaults == {"mode": 0, "quality": 11, "lgwin": 22, "lgblock": 0}
    assert params[0].default is inspect.Parameter.empty


def test_decompress_signature():
    params = list(inspect.signature(brotli_pure.decompress).parameters.values())
    assert len(params) >= 1 and params[0].default is inspect.Parameter.empty
    assert all(p.default is not inspect.Parameter.empty for p in params[1:])


# --------------------------------------------------------------------------
# decompress argument handling
# --------------------------------------------------------------------------

STREAM_HELLO = bytes.fromhex("0b028068656c6c6f03")  # oracle.compress(b"hello", quality=5)


def _buffer_variants(data):
    yield "bytes", bytes(data)
    yield "bytearray", bytearray(data)
    yield "memoryview", memoryview(data)
    yield "memoryview_slice", memoryview(b"XX" + data + b"YY")[2:-2]
    yield "array_B", array.array("B", data)


def test_decompress_buffer_types():
    support.need_decoder()
    for label, buf in _buffer_variants(STREAM_HELLO):
        out = brotli_pure.decompress(buf)
        assert out == b"hello", label
        assert type(out) is bytes, label


def test_decompress_does_not_modify_input():
    support.need_decoder()
    buf = bytearray(STREAM_HELLO)
    brotli_pure.decompress(buf)
    assert buf == STREAM_HELLO


@pytest.mark.oracle
def test_decompress_noncontiguous_like_oracle():
    support.need_decoder()
    o = support.need_oracle()
    mv = memoryview(bytes(STREAM_HELLO) * 2)[::2]
    with pytest.raises(TypeError):
        o.decompress(mv)
    with pytest.raises(TypeError):
        brotli_pure.decompress(mv)


@pytest.mark.parametrize("bad", ["abc", None, 5, 1.5, [1, 2]], ids=repr)
def test_decompress_non_buffer_type_error(bad):
    support.need_decoder()
    with pytest.raises(TypeError):
        brotli_pure.decompress(bad)


# --------------------------------------------------------------------------
# compress argument handling
# --------------------------------------------------------------------------

def test_compress_buffer_types():
    support.need_encoder(5)
    data = corpus.small()["english_text_1500"]
    ref = brotli_pure.compress(data, quality=5)
    assert type(ref) is bytes
    for label, buf in _buffer_variants(data):
        out = brotli_pure.compress(buf, quality=5)
        assert type(out) is bytes, label
        assert out == ref, label


def test_compress_does_not_modify_input():
    support.need_encoder(5)
    buf = bytearray(corpus.small()["english_text_1500"])
    before = bytes(buf)
    brotli_pure.compress(buf, quality=5)
    assert buf == before


@pytest.mark.parametrize("bad", ["abc", None, 5, 1.5], ids=repr)
def test_compress_non_buffer_type_error(bad):
    support.need_encoder_module()
    with pytest.raises(TypeError):
        brotli_pure.compress(bad, quality=5)


@pytest.mark.oracle
def test_compress_positional_argument_order():
    support.need_encoder(5)
    data = corpus.small()["english_text_1500"]
    a = brotli_pure.compress(data, brotli_pure.MODE_TEXT, 5, 18, 0)
    b = brotli_pure.compress(data, mode=brotli_pure.MODE_TEXT, quality=5, lgwin=18, lgblock=0)
    assert a == b
    o = support.need_oracle()
    assert o.decompress(a) == data


def test_compress_default_quality_is_11():
    support.need_encoder(11)
    data = b"abcabcabc hello hello"
    assert brotli_pure.compress(data) == brotli_pure.compress(data, quality=11, lgwin=22)


# --------------------------------------------------------------------------
# Parameter validation vs oracle
# --------------------------------------------------------------------------

# Values the oracle rejects with brotli.error (range checks in _brotli.c).
INVALID = (
    [("quality", q) for q in (12, 13, 20, 100, 255)]
    + [("lgwin", w) for w in (0, 1, 9, 25, 26, 30, 255)]
    + [("lgblock", b) for b in (1, 2, 15, 25, 30, 255)]
    + [("mode", m) for m in (3, 4, 255)]
)
# Values the oracle rejects while parsing the argument (unsigned char):
# OverflowError for out-of-byte-range ints, TypeError for non-ints.
OVERFLOW = [("quality", -1), ("quality", 256), ("lgwin", -1), ("lgwin", 1 << 40),
            ("lgblock", -1), ("mode", -1), ("mode", 256)]
WRONG_TYPE = [("quality", 5.0), ("quality", "5"), ("quality", None), ("lgwin", 22.0),
              ("mode", "0"), ("lgblock", None)]


def _ids(cases):
    return ["%s=%r" % c for c in cases]


@pytest.mark.parametrize("param,value", INVALID, ids=_ids(INVALID))
@pytest.mark.parametrize("data", [b"", b"some data"], ids=["empty", "data"])
def test_invalid_parameter_raises_error(param, value, data):
    support.need_encoder_module()
    kwargs = {"quality": 5, param: value}
    if support.oracle is not None:
        with pytest.raises(support.oracle.error):
            support.oracle.compress(data, **kwargs)
    support.assert_raises_brotli_error(brotli_pure.compress, data, what="compress(%r)" % kwargs,
                                       **kwargs)


@pytest.mark.parametrize("param,value", OVERFLOW, ids=_ids(OVERFLOW))
def test_out_of_byte_range_parameter_rejected(param, value):
    """Oracle raises OverflowError; we accept OverflowError, ValueError or
    brotli_pure.error, but it must be rejected (not wrapped around)."""
    support.need_encoder_module()
    kwargs = {"quality": 5, param: value}
    if support.oracle is not None:
        with pytest.raises(OverflowError):
            support.oracle.compress(b"x", **kwargs)
    with pytest.raises((OverflowError, ValueError, BrotliError)):
        brotli_pure.compress(b"x", **kwargs)


@pytest.mark.parametrize("param,value", WRONG_TYPE, ids=_ids(WRONG_TYPE))
def test_wrong_type_parameter_type_error(param, value):
    support.need_encoder_module()
    kwargs = {"quality": 5, param: value}
    if support.oracle is not None:
        with pytest.raises(TypeError):
            support.oracle.compress(b"x", **kwargs)
    with pytest.raises(TypeError):
        brotli_pure.compress(b"x", **kwargs)


VALID_EDGES = ([dict(lgwin=w) for w in (10, 24)] + [dict(lgblock=b) for b in (0, 16, 24)]
               + [dict(mode=m) for m in (0, 1, 2)])


@pytest.mark.oracle
@pytest.mark.parametrize("quality", (5, 11))
@pytest.mark.parametrize("kwargs", VALID_EDGES, ids=[repr(k) for k in VALID_EDGES])
def test_valid_edge_parameters_accepted(kwargs, quality):
    support.need_encoder(quality)
    o = support.need_oracle()
    data = corpus.small()["short_quickfox"] * 3
    out = brotli_pure.compress(data, quality=quality, **kwargs)
    assert o.decompress(out) == data


@pytest.mark.oracle
@pytest.mark.parametrize("quality", [q for q in range(12) if q not in support.REQUIRED_QUALITIES])
def test_optional_quality_roundtrips_or_raises(quality):
    """Qualities other than 5/11 may be unsupported, but then they must raise
    brotli_pure.error or NotImplementedError; if supported they must produce
    valid streams."""
    support.need_encoder_module()
    o = support.need_oracle()
    data = corpus.small()["english_text_1500"]
    try:
        out = brotli_pure.compress(data, quality=quality)
    except (BrotliError, NotImplementedError):
        return
    except ModuleNotFoundError as e:
        if not (e.name or "").startswith("brotli_pure."):
            raise
        msg = ("quality %d raises ModuleNotFoundError(%s); an unsupported quality should raise "
               "brotli_pure.error" % (quality, e.name))
        if support.STRICT:
            pytest.fail(msg)
        pytest.skip(msg)
    assert type(out) is bytes
    assert o.decompress(out) == data


@pytest.mark.oracle
@pytest.mark.parametrize("quality", [q for q in range(12) if q not in support.REQUIRED_QUALITIES])
@pytest.mark.parametrize("name", ["empty", "short_quickfox", "english_text_1500", "random_1000",
                                  "dict_dense_text_2000", "records_2400"])
def test_optional_quality_byte_exact_if_supported(quality, name):
    """If an optional quality is implemented, it should be as exact as 5/11."""
    support.need_encoder(quality)
    o = support.need_exact_oracle()
    data = corpus.small()[name]
    for lgwin in (10, 22):
        out = brotli_pure.compress(data, quality=quality, lgwin=lgwin)
        ref = o.compress(data, quality=quality, lgwin=lgwin)
        assert out == ref, "q=%d lgwin=%d: %s" % (quality, lgwin, support.first_diff(out, ref))
