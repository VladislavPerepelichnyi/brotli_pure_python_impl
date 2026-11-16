"""Property-based tests (hypothesis).

Example counts come from the hypothesis profile chosen in conftest.py
(``default``: 40, derandomized; ``thorough`` with BROTLI_PURE_SLOW=1;
override with BROTLI_PURE_HYPOTHESIS_PROFILE).

* compress(q5 / q11) round-trips through both decoders and is byte-exact
* decompress of oracle output for random data and random parameters
* differential decoding of mutated streams and of random bytes
* decoding of random hand-crafted streams vs the bitstream model
"""

import random

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import brotli_pure  # noqa: E402
from tests import bitstream as bs  # noqa: E402
from tests import corpus, support  # noqa: E402

WORDS = [w.encode() for w in corpus.ENGLISH_WORDS[:60] + corpus.DICT_WORDS[:60]]


def _repeat(args):
    unit, n = args
    return unit * n


data_strategy = st.one_of(
    st.binary(max_size=2000),
    st.text(max_size=600).map(lambda s: s.encode("utf-8")),
    st.lists(st.sampled_from(WORDS + [b" ", b". ", b", ", b"\n", b"The ", b" the "]),
             max_size=400).map(b"".join),
    st.tuples(st.binary(min_size=1, max_size=12), st.integers(1, 300)).map(_repeat),
    st.lists(st.sampled_from([b"\x00", b"\xff", b"ab", b"\x01\x02\x03\x04"]), max_size=500)
    .map(b"".join),
)
small_data_strategy = st.one_of(
    st.binary(max_size=300),
    st.lists(st.sampled_from(WORDS + [b" ", b". "]), max_size=60).map(b"".join),
    st.tuples(st.binary(min_size=1, max_size=6), st.integers(1, 60)).map(_repeat),
)


def _check_compress(data, quality, lgwin, mode):
    out = brotli_pure.compress(data, quality=quality, lgwin=lgwin, mode=mode)
    assert type(out) is bytes
    if support.oracle is not None:
        assert support.oracle.decompress(out) == data, "oracle cannot decode our output"
    if support.decoder_available():
        assert support.ours_decompress(out, len(data)) == data, "own decoder round-trip failed"
    if support.oracle is not None and support.ORACLE_VERSION == support.EXPECTED_ORACLE_VERSION:
        ref = support.oracle.compress(data, quality=quality, lgwin=lgwin, mode=mode)
        assert out == ref, "not byte-exact vs oracle: %s" % support.first_diff(out, ref)


@given(data=data_strategy, lgwin=st.integers(10, 24), mode=st.integers(0, 2))
def test_q5_roundtrip_and_exact(data, lgwin, mode):
    support.need_encoder(5)
    _check_compress(data, 5, lgwin, mode)


@settings(max_examples=max(5, settings().max_examples // 3))
@given(data=small_data_strategy, lgwin=st.integers(10, 24), mode=st.integers(0, 2))
def test_q11_roundtrip_and_exact(data, lgwin, mode):
    support.need_encoder(11)
    _check_compress(data, 11, lgwin, mode)


@pytest.mark.oracle
@given(data=data_strategy, quality=st.integers(0, 11), lgwin=st.integers(10, 24),
       mode=st.integers(0, 2), lgblock=st.sampled_from([0, 16, 18, 24]))
def test_decompress_oracle_output(data, quality, lgwin, mode, lgblock):
    support.need_decoder()
    support.need_oracle()
    stream = support.oracle.compress(data, quality=quality, lgwin=lgwin, mode=mode, lgblock=lgblock)
    got = support.ours_decompress(stream, len(data))
    support.assert_same_bytes(got, data)


@pytest.mark.oracle
@given(data=st.binary(min_size=1, max_size=600), quality=st.integers(0, 11),
       mutations=st.lists(st.tuples(st.integers(0, 10_000), st.integers(0, 255),
                                    st.sampled_from(["set", "xor", "ins", "del"])),
                          min_size=1, max_size=4),
       cut=st.one_of(st.none(), st.integers(0, 10_000)))
def test_decompress_mutated_streams_differential(data, quality, mutations, cut):
    support.need_decoder()
    support.need_oracle()
    s = bytearray(support.oracle.compress(data, quality=quality))
    for pos, val, op in mutations:
        if not s:
            break
        p = pos % len(s)
        if op == "set":
            s[p] = val
        elif op == "xor":
            s[p] ^= (val or 1)
        elif op == "ins":
            s.insert(p, val)
        elif len(s) > 1:
            del s[p]
    if cut is not None and s:
        s = s[:cut % (len(s) + 1)]
    support.check_decode_matches_oracle(bytes(s), max_output=1 << 20)


@pytest.mark.oracle
@given(st.binary(max_size=128))
def test_decompress_random_bytes_differential(blob):
    support.need_decoder()
    support.need_oracle()
    support.check_decode_matches_oracle(blob, max_output=1 << 20)


@given(seed=st.integers(0, 2**32 - 1))
def test_decompress_random_crafted_stream(seed):
    support.need_decoder()
    stream, expected, desc = bs.random_stream(random.Random(seed))
    assume(expected is not None)
    if bs.DICTIONARY is None:
        assume("C" not in desc)
    got = support.ours_decompress(stream, len(expected))
    support.assert_same_bytes(got, expected, desc)
    if support.oracle is not None:  # keep the model honest
        assert support.oracle.decompress(stream) == expected
