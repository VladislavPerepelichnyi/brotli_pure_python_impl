"""brotli_pure.decompress on hand-crafted streams (tests/crafted_cases.py).

Expectations come from the independent model in bitstream.py; the same
cases are cross-checked against the oracle in test_harness.py, so a failure
here is a decoder bug, not a test bug (unless test_harness fails too).

Covers: every WBITS value and encoding, the large-window marker, empty and
metadata-only streams, uncompressed metablocks (4/5/6 nibble MLEN, exuberant
nibbles), metadata blocks (all MSKIPBYTES sizes, reserved bit, padding),
simple prefix codes (1-4 symbols, both 4-symbol tree shapes), complex codes
(HSKIP 0/2/3, run-length codes 16/17, 15-bit codes), every insert and copy
length code, overlapping copies, implicit/short/explicit distances incl. the
initial ring buffer, NPOSTFIX 0..3 x NDIRECT, window-limit distances for
every WBITS, dictionary references with all 121 transforms (zero-length
results, UTF-8 uppercase overruns, after the window is full, ring-buffer
interaction), and a set of invalid constructions.
"""

import random

import pytest

from tests import bitstream as bs
from tests import support
from tests.crafted_cases import CASES


@pytest.fixture(autouse=True)
def _decoder():
    support.need_decoder()


def _params(cases):
    out = []
    for c in cases:
        marks = [pytest.mark.slow] if c.slow else []
        out.append(pytest.param(c, id=c.name, marks=marks))
    return out


def _run_case(case):
    if case.needs_dictionary and bs.DICTIONARY is None:
        pytest.skip("static dictionary not found (set BROTLI_REFERENCE)")
    stream, expected = case.build()
    if expected is None:
        support.assert_raises_brotli_error(support.ours_decompress, stream,
                                           what="decompress(%s)" % case.name)
    else:
        got = support.ours_decompress(stream, len(expected))
        assert type(got) is bytes
        support.assert_same_bytes(got, expected, case.name)


@pytest.mark.parametrize("case", _params(CASES))
def test_crafted(case):
    _run_case(case)


VALID_SMALL = [c for c in CASES
               if not c.slow and not c.name.startswith(("window", "random_crafted", "dict_transform"))]


@pytest.mark.parametrize("case", _params([c for c in VALID_SMALL if c.name in (
    "uncompressed_then_compressed_backref", "metadata_between_compressed", "short_codes_all_16",
    "literals_complex_hskip2_repeat1", "short_codes_initial_ring_buffer", "dict_ref_between_backrefs_ring_buffer",
    "implicit_distance_commands", "metadata_is_last", "literals_simple_4sym_tree1",
    "dict_uppercase_utf8_overrun_last_in_stream", "empty_wbits17", "many_small_metablocks")]))
def test_crafted_truncated_every_byte(case):
    """Every strict prefix of a valid stream must be rejected (oracle-checked
    when available)."""
    if case.needs_dictionary and bs.DICTIONARY is None:
        pytest.skip("static dictionary not found")
    stream, expected = case.build()
    assert expected is not None
    for n in range(len(stream)):
        prefix = stream[:n]
        if support.oracle is not None:
            support.check_decode_matches_oracle(prefix)
        else:
            support.assert_raises_brotli_error(support.ours_decompress, prefix,
                                               what="decompress(prefix %d)" % n)


@pytest.mark.parametrize("seed", range(0, 200, 20))
def test_random_crafted_streams(seed):
    """Random streams from the mini-encoder; expectation from the model."""
    for s in range(seed, seed + 20):
        stream, expected, desc = bs.random_stream(random.Random(50_000 + s))
        if expected is None or (bs.DICTIONARY is None and "C" in desc):
            continue
        got = support.ours_decompress(stream, len(expected))
        support.assert_same_bytes(got, expected, "seed %d (%s)" % (50_000 + s, desc))


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(0, 2000, 100))
def test_random_crafted_streams_many(seed):
    for s in range(seed, seed + 100):
        stream, expected, desc = bs.random_stream(random.Random(90_000 + s), max_out=20_000)
        if expected is None:
            continue
        got = support.ours_decompress(stream, len(expected))
        support.assert_same_bytes(got, expected, "seed %d (%s)" % (90_000 + s, desc))
