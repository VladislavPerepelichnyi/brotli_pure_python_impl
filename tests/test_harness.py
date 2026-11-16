"""Self-checks of the test infrastructure (no brotli_pure code involved).

These tests make sure the *expectations* used elsewhere are trustworthy:
the crafted-stream model in bitstream.py is compared with the oracle, the
corpus is deterministic, and the small helper encoders/decoders agree.
"""

import hashlib
import random

import pytest

from tests import bitstream as bs
from tests import corpus, support
from tests.crafted_cases import CASES


def _params(cases):
    out = []
    for c in cases:
        marks = [pytest.mark.oracle]
        if c.slow:
            marks.append(pytest.mark.slow)
        out.append(pytest.param(c, id=c.name, marks=marks))
    return out


@pytest.mark.parametrize("case", _params(CASES))
def test_crafted_case_matches_oracle(case):
    """The model's verdict (bytes or 'invalid') equals the C decoder's."""
    oracle = support.need_oracle()
    if case.needs_dictionary and bs.DICTIONARY is None:
        pytest.skip("static dictionary not found")
    stream, expected = case.build()
    ok, got = support.oracle_decompress(stream)
    if expected is None:
        assert not ok, "model says invalid but oracle decodes %d bytes" % len(got)
    else:
        assert ok, "model expects %d bytes but oracle rejects" % len(expected)
        support.assert_same_bytes(got, expected)
    assert oracle is not None


@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(0, 300, 10))
def test_random_crafted_streams_match_oracle(seed):
    support.need_oracle()
    for s in range(seed, seed + 10):
        stream, expected, desc = bs.random_stream(random.Random(s))
        ok, got = support.oracle_decompress(stream)
        assert ok == (expected is not None), (s, desc)
        if ok:
            support.assert_same_bytes(got, expected, "seed %d (%s)" % (s, desc))


def test_bitwriter_lsb_first():
    w = bs.BitWriter()
    w.write(1, 1)
    w.write(0b10, 2)
    w.write(0x1F, 5)
    w.write(0xABC, 12)
    assert w.getvalue() == bytes([0b11111101, 0xBC, 0x0A])
    w = bs.BitWriter()
    w.write(0, 1)
    w.write(1, 1)
    w.write(1, 1)
    assert w.getvalue() == b"\x06"  # the canonical empty stream


def test_wbits_header_encodings():
    for wb in range(10, 25):
        b = bs.StreamBuilder(wb)
        stream, exp = b.finish()
        # decode the header by hand per RFC 7932 section 9.1
        v = int.from_bytes(stream, "little")
        if v & 1 == 0:
            got = 16
        elif (v >> 1) & 7:
            got = 17 + ((v >> 1) & 7)
        elif (v >> 4) & 7:
            got = 8 + ((v >> 4) & 7)
        else:
            got = 17
        assert got == wb


@pytest.mark.parametrize("npostfix", range(4))
@pytest.mark.parametrize("ndirect_field", (0, 1, 15))
def test_distance_code_roundtrip(npostfix, ndirect_field):
    ndirect = ndirect_field << npostfix
    size = bs.distance_alphabet_size(npostfix, ndirect)
    for d in list(range(1, 3000)) + [1 << 20, (1 << 24) - 16, (1 << 26) - 100]:
        sym, extra, nbits = bs.encode_distance(d, npostfix, ndirect)
        assert sym < size
        assert 0 <= extra < (1 << nbits) or nbits == 0
        assert bs.decode_distance_symbol(sym, extra, npostfix, ndirect) == d


def test_length_codes_cover_ranges():
    for n in list(range(0, 3000)) + [22594, 22594 + (1 << 24) - 1]:
        c, e, nb = bs.insert_code(n)
        assert bs.INS_BASE[c] + e == n and e < (1 << nb) or (nb == 0 and e == 0)
    for n in list(range(2, 3000)) + [2118 + (1 << 24) - 1]:
        c, e, nb = bs.copy_code(n)
        assert bs.COPY_BASE[c] + e == n


def test_staircase_lengths_complete():
    for n in (2, 3, 5, 17, 64, 256, 704):
        lengths = bs.staircase_lengths(range(n))
        assert bs.kraft_ok(lengths)
        assert max(lengths.values()) <= 15


def test_rfc_transform_table_shape():
    assert len(bs.TRANSFORMS) == 121
    assert bs.TRANSFORMS[0] == (b"", 0, b"")
    assert bs.TRANSFORMS[9] == (b"", 10, b"")       # uppercase first
    assert bs.TRANSFORMS[44] == (b"", 11, b"")      # uppercase all
    assert bs.TRANSFORMS[5] == (b"", 0, b" the ")


def test_dictionary_available():
    if bs.DICTIONARY is None:
        pytest.skip("dictionary.bin not found (set BROTLI_REFERENCE)")
    assert len(bs.DICTIONARY) == bs.DICT_SIZE == 122784
    assert bs.dict_word(4, 0) == b"time"


# Corpus determinism.  If you intentionally change corpus.py, update the hash.
SMALL_CORPUS_SHA256 = "158eb750f1537e438dc07025aec8ccaeaf8ffb280a9860209f19ba1e561a9f6b"


def test_corpus_deterministic():
    a = corpus.small()
    h = hashlib.sha256()
    for k in sorted(a):
        h.update(k.encode() + b"\0" + a[k])
    corpus.small.cache_clear()
    b = corpus.small()
    assert a == b
    assert len(a["english_text_12000"]) == 12000
    assert len(corpus.single_bytes()) == 256
    if SMALL_CORPUS_SHA256:
        assert h.hexdigest() == SMALL_CORPUS_SHA256, "corpus changed (update the hash if intended)"


def test_testdata_found():
    if not corpus.testdata_compressed():
        pytest.skip("no reference test vectors found (BROTLI_TESTDATA)")
    names = {n for n, _, _ in corpus.testdata_compressed()}
    assert "10x10y.compressed" in names
