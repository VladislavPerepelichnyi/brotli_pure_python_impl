"""brotli_pure.decompress on corrupt / invalid input, differential vs oracle.

Property checked everywhere (support.check_decode_matches_oracle):
  oracle accepts -> identical output;  oracle rejects -> brotli_pure.error,
  never another exception type, never a hang (SIGALRM deadline).

Inputs: truncation at every byte, every single-bit flip (small streams) or
seeded random flips, byte substitutions/insertions/deletions, trailing
garbage, all 1-byte inputs, 2-byte inputs, random byte strings, and wrong
argument types.
"""

import functools
import random

import pytest

import brotli_pure
from tests import corpus, support

pytestmark = [pytest.mark.oracle]

MAX_FUZZ_OUTPUT = 1 << 20  # skip mutants that legitimately expand hugely


@pytest.fixture(autouse=True)
def _decoder_and_oracle():
    support.need_decoder()
    support.need_oracle()


# name -> (corpus sample, quality, lgwin)
SAMPLES = {
    "hello_q11": ("short_hello_world", 11, 22),
    "quickfox_q5": ("short_quickfox", 5, 16),
    "english_q0": ("english_text_1500", 0, 18),
    "english_q5": ("english_text_1500", 5, 22),
    "english_q11": ("english_text_1500", 11, 22),
    "dict_q11": ("dict_dense_text_2000", 11, 24),
    "random_q5_uncompressed": ("random_255", 5, 22),
    "json_q9": ("json_text_6000", 9, 20),
    "records_q11": ("records_5000", 11, 22),
    "zeros_q1": ("zeros_1000", 1, 10),
    "empty_q5": ("empty", 5, 22),
}


@functools.lru_cache(maxsize=None)
def sample_stream(key):
    name, q, lgwin = SAMPLES[key]
    data = corpus.small()[name]
    return support.oracle.compress(data, quality=q, lgwin=lgwin, mode=support.default_mode_for(name))


def _vectors():
    out = []
    for label, cpath, opath in corpus.testdata_compressed():
        if label.startswith(("x.compressed", "empty.compressed.0", "10x10y", "64x", "quickfox.",
                             "ukkonooa", "xyzzy", "monkey")):
            out.append(label)
    return out


@pytest.mark.parametrize("key", sorted(SAMPLES))
def test_truncated_every_byte(key):
    stream = sample_stream(key)
    for n in range(len(stream)):
        support.check_decode_matches_oracle(stream[:n])


@pytest.mark.parametrize("label", _vectors() or ["none"])
def test_reference_vector_truncated_every_byte(label):
    paths = {l: c for l, c, _ in corpus.testdata_compressed()}
    if label not in paths:
        pytest.skip("no testdata")
    stream = corpus.read(paths[label])
    for n in range(len(stream)):
        support.check_decode_matches_oracle(stream[:n])


@pytest.mark.parametrize("key", sorted(SAMPLES))
def test_bit_flips(key):
    stream = sample_stream(key)
    nbits = len(stream) * 8
    if nbits <= 800 or support.SLOW:
        positions = range(nbits)
    else:
        positions = random.Random(key).sample(range(nbits), 400)
    for p in positions:
        mutated = bytearray(stream)
        mutated[p >> 3] ^= 1 << (p & 7)
        support.check_decode_matches_oracle(bytes(mutated), max_output=MAX_FUZZ_OUTPUT)


@pytest.mark.parametrize("key", sorted(SAMPLES))
def test_byte_mutations(key):
    stream = sample_stream(key)
    r = random.Random("mut-" + key)
    count = 1500 if support.SLOW else 150
    for _ in range(count):
        m = bytearray(stream)
        op = r.random()
        if op < 0.4 and m:
            m[r.randrange(len(m))] = r.randrange(256)
        elif op < 0.6:
            m.insert(r.randrange(len(m) + 1), r.randrange(256))
        elif op < 0.8 and m:
            del m[r.randrange(len(m))]
        else:
            for _ in range(r.randint(2, 6)):
                if m:
                    m[r.randrange(len(m))] = r.randrange(256)
        support.check_decode_matches_oracle(bytes(m), max_output=MAX_FUZZ_OUTPUT)


TRAILERS = [b"\x00", b"\x01", b"\xff", b"a", b"\x06", b"\x00" * 16]


@pytest.mark.parametrize("key", ["hello_q11", "english_q5", "random_q5_uncompressed", "empty_q5"])
def test_trailing_garbage(key):
    stream = sample_stream(key)
    for t in TRAILERS + [stream]:
        ok, _ = support.check_decode_matches_oracle(stream + t)
        assert not ok, "oracle accepted trailing data (unexpected)"


def test_reference_python_binding_garbage_case():
    """From brotli's own python/tests/decompress_test.py: compress(b'a') + b'a'."""
    stream = support.oracle.compress(b"a") + b"a"
    support.assert_raises_brotli_error(brotli_pure.decompress, stream)


def test_all_single_byte_inputs():
    for b in range(256):
        support.check_decode_matches_oracle(bytes([b]))


def test_two_byte_inputs():
    values = range(65536) if support.SLOW else random.Random(2).sample(range(65536), 3000)
    for v in values:
        support.check_decode_matches_oracle(v.to_bytes(2, "little"))


def test_random_byte_strings():
    r = random.Random(3)
    count = 20000 if support.SLOW else 1500
    for _ in range(count):
        n = r.choice((3, 4, 5, 8, 16, 33, 64, 200))
        support.check_decode_matches_oracle(r.randbytes(n), max_output=MAX_FUZZ_OUTPUT)


def test_empty_input_is_error():
    support.assert_raises_brotli_error(brotli_pure.decompress, b"")


@pytest.mark.parametrize("bad", ["abc", None, 5, 1.5, [1, 2]], ids=repr)
def test_non_buffer_argument_raises_type_error(bad):
    # Oracle: TypeError("brotli: data must be a C-contiguous buffer").
    with pytest.raises(TypeError):
        brotli_pure.decompress(bad)


def test_error_is_exception_subclass_and_catchable():
    try:
        brotli_pure.decompress(b"\xff\xff\xff")
    except brotli_pure.error as e:
        assert isinstance(e, Exception)
    else:
        pytest.fail("no error raised")
