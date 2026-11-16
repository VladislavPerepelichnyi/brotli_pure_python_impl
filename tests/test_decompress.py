"""brotli_pure.decompress on valid streams.

* reference test vectors (tests/testdata of google/brotli)
* the corpus compressed by the oracle at every quality 0..11, several window
  sizes and all modes (covers block switching, context maps, NPOSTFIX/NDIRECT,
  dictionary references, ... as produced by the real encoder)
* lgblock variations, every single-byte input, boundary-crossing inputs
"""

import functools
import itertools
import os

import pytest

import brotli_pure
from tests import corpus, support

QUALITIES = list(range(12))
DEFAULT_LGWINS = (10, 16, 22, 24)
VECTOR_SLOW_THRESHOLD = 160_000  # original size above which a vector is "slow"


@pytest.fixture(autouse=True)
def _decoder():
    support.need_decoder()


@functools.lru_cache(maxsize=4096)
def oracle_compress(name, tier, quality, lgwin, mode, lgblock=0):
    data = getattr(corpus, tier)()[name]
    return support.oracle.compress(data, mode=mode, quality=quality, lgwin=lgwin, lgblock=lgblock)


def check(data, stream, label):
    got = support.ours_decompress(stream, len(data))
    assert type(got) is bytes, "decompress returned %r" % type(got)
    if got != data:
        raise AssertionError("%s: %s" % (label, support.first_diff(got, data)))


# --------------------------------------------------------------------------
# Reference vectors
# --------------------------------------------------------------------------

def _vector_params():
    out = []
    for label, cpath, opath in corpus.testdata_compressed():
        marks = [pytest.mark.slow] if os.path.getsize(opath) > VECTOR_SLOW_THRESHOLD else []
        out.append(pytest.param(cpath, opath, id=label, marks=marks))
    if not out:
        out.append(pytest.param(None, None, id="no-testdata",
                                marks=pytest.mark.skip(reason="no reference test vectors found")))
    return out


@pytest.mark.parametrize("cpath,opath", _vector_params())
def test_reference_vector(cpath, opath):
    data = corpus.read(opath)
    stream = corpus.read(cpath)
    check(data, stream, cpath)


# --------------------------------------------------------------------------
# Oracle-compressed corpus
# --------------------------------------------------------------------------

def _mode_for(name, quality, lgwin):
    # Deterministic cycling so the default matrix covers all three modes.
    return (support.default_mode_for(name) + quality + lgwin) % 3


@pytest.mark.oracle
@pytest.mark.parametrize("quality", QUALITIES)
@pytest.mark.parametrize("name", sorted(corpus.small()))
def test_oracle_small_corpus(name, quality):
    support.need_oracle()
    data = corpus.small()[name]
    for lgwin in DEFAULT_LGWINS:
        mode = _mode_for(name, quality, lgwin)
        stream = oracle_compress(name, "small", quality, lgwin, mode)
        check(data, stream, "q=%d lgwin=%d mode=%d" % (quality, lgwin, mode))


@pytest.mark.slow
@pytest.mark.oracle
@pytest.mark.parametrize("quality", QUALITIES)
@pytest.mark.parametrize("name", sorted(corpus.small()))
def test_oracle_small_corpus_full_matrix(name, quality):
    """All lgwin 10..24 x all modes."""
    support.need_oracle()
    data = corpus.small()[name]
    for lgwin, mode in itertools.product(range(10, 25), range(3)):
        stream = oracle_compress(name, "small", quality, lgwin, mode)
        check(data, stream, "q=%d lgwin=%d mode=%d" % (quality, lgwin, mode))


MEDIUM_DEFAULT = ("zeros_65537", "english_text_65536p1000", "boundary_match_64k",
                  "long_repeat_gap_70000")


def _medium_params():
    out = []
    for name in sorted(corpus.medium()):
        for q in (0, 1, 2, 4, 5, 9, 10, 11):
            default = name in MEDIUM_DEFAULT and q in (1, 5, 11)
            out.append(pytest.param(name, q, id="%s-q%d" % (name, q),
                                    marks=[] if default else [pytest.mark.slow]))
    return out


@pytest.mark.oracle
@pytest.mark.parametrize("name,quality", _medium_params())
def test_oracle_medium_corpus(name, quality):
    support.need_oracle()
    data = corpus.medium()[name]
    lgwins = (22,) if not support.SLOW else (16, 18, 22, 24)
    for lgwin in lgwins:
        stream = oracle_compress(name, "medium", quality, lgwin, support.default_mode_for(name))
        check(data, stream, "q=%d lgwin=%d" % (quality, lgwin))


@pytest.mark.slow
@pytest.mark.oracle
@pytest.mark.parametrize("quality", (1, 5, 9, 11))
@pytest.mark.parametrize("name", sorted(corpus.large()))
def test_oracle_large_corpus(name, quality):
    support.need_oracle()
    data = corpus.large()[name]
    for lgwin in (20, 24):
        stream = oracle_compress(name, "large", quality, lgwin, 0)
        check(data, stream, "q=%d lgwin=%d" % (quality, lgwin))


@pytest.mark.oracle
@pytest.mark.parametrize("quality", (0, 1, 2, 5, 9, 10, 11))
def test_oracle_single_bytes(quality):
    support.need_oracle()
    for b in corpus.single_bytes():
        stream = support.oracle.compress(b, quality=quality)
        check(b, stream, "byte %r q=%d" % (b, quality))


@pytest.mark.oracle
@pytest.mark.parametrize("lgblock", (16, 17, 18, 20, 24))
@pytest.mark.parametrize("quality", (5, 11))
def test_oracle_lgblock(quality, lgblock):
    support.need_oracle()
    name = "mixed_15000" if quality == 11 else "english_text_12000"
    data = corpus.small()[name]
    stream = oracle_compress(name, "small", quality, 22, 0, lgblock)
    check(data, stream, "q=%d lgblock=%d" % (quality, lgblock))


@pytest.mark.slow
@pytest.mark.oracle
@pytest.mark.parametrize("lgblock", (16, 18, 24))
@pytest.mark.parametrize("quality", (2, 5, 11))
def test_oracle_lgblock_medium(quality, lgblock):
    support.need_oracle()
    name = "mixed_200000"
    data = corpus.medium()[name]
    stream = oracle_compress(name, "medium", quality, 24, 0, lgblock)
    check(data, stream, "q=%d lgblock=%d" % (quality, lgblock))


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(corpus.testdata_originals()) or ["none"])
def test_oracle_reference_originals_q5_q11(name):
    """Reference originals re-compressed by the oracle (small files only by default)."""
    support.need_oracle()
    paths = corpus.testdata_originals()
    if name not in paths:
        pytest.skip("no testdata")
    data = corpus.read(paths[name])
    if len(data) > 64_000 and not support.SLOW:
        pytest.skip("large file: set BROTLI_PURE_SLOW=1")
    if len(data) > 2_000_000:
        pytest.skip("too large for a pure-Python decode test")
    for q in (5, 11):
        stream = support.oracle.compress(data, quality=q)
        check(data, stream, "q=%d" % q)


def test_decompress_returns_new_bytes_object():
    out = brotli_pure.decompress(b"\x06")
    assert out == b"" and type(out) is bytes
