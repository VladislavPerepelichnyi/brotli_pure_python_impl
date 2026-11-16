"""Decoder development tests for brotli_pure.decompress.

Compares the pure-Python decoder against the reference test vectors and
against the C-backed ``brotli`` module (the oracle, v1.2.1), including a
differential fuzz test.

Environment variables:
    BROTLI_TESTDATA          directory with the reference test vectors
                             (default: ../brotli_reference/tests/testdata)
    BROTLI_REFERENCE         reference source tree (for java/.../SynthTest.java
                             and out/brotli); default ../brotli_reference
    BROTLI_FUZZ_ITERATIONS   number of fuzz cases per seed (default 2500)
    BROTLI_FUZZ_SEEDS        number of fuzz seeds (default 4)
"""

import glob
import os
import random
import re
import subprocess
import tempfile

import pytest

import brotli_pure
from brotli_pure.common import transforms as tf
from brotli_pure.common.dictionary import (DATA as DICT, OFFSETS_BY_LENGTH,
                                           SIZE_BITS_BY_LENGTH)
from brotli_pure.dec.prefix import CMD_LUT

brotli = pytest.importorskip("brotli")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_REF = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "brotli_reference"))
if not os.path.isdir(_DEFAULT_REF):
    _DEFAULT_REF = "/home/vlad/thesis/wasm/pure_python_impl/brotli_reference"
REFERENCE = os.environ.get("BROTLI_REFERENCE", _DEFAULT_REF)
TESTDATA = os.environ.get("BROTLI_TESTDATA",
                          os.path.join(REFERENCE, "tests", "testdata"))

FUZZ_ITERATIONS = int(os.environ.get("BROTLI_FUZZ_ITERATIONS", "2500"))
FUZZ_SEEDS = int(os.environ.get("BROTLI_FUZZ_SEEDS", "4"))


def oracle(data):
    """Returns the oracle's output, or None if it raises brotli.error."""
    try:
        return brotli.decompress(data)
    except brotli.error:
        return None


def ours(data):
    try:
        return brotli_pure.decompress(data)
    except brotli_pure.error:
        return None


def assert_same(data):
    exp = oracle(data)
    got = ours(data)
    if exp is None:
        assert got is None, "oracle rejects, we returned %d bytes" % len(got)
    else:
        assert got is not None, "oracle accepts (%d bytes), we raised" % len(exp)
        assert got == exp
    return exp


# ---------------------------------------------------------------------------
# Reference test vectors
# ---------------------------------------------------------------------------

def _vector_files():
    if not os.path.isdir(TESTDATA):
        return []
    return sorted(glob.glob(os.path.join(TESTDATA, "*.compressed*")))


@pytest.mark.skipif(not os.path.isdir(TESTDATA), reason="no test vectors")
@pytest.mark.parametrize("path", _vector_files(), ids=os.path.basename)
def test_reference_vectors(path):
    original = path[:path.index(".compressed")]
    with open(path, "rb") as f:
        compressed = f.read()
    with open(original, "rb") as f:
        expected = f.read()
    assert brotli_pure.decompress(compressed) == expected
    assert oracle(compressed) == expected


def _synth_vectors():
    path = os.path.join(REFERENCE, "java", "org", "brotli", "dec",
                        "SynthTest.java")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        src = f.read()
    out = []
    for name, body in re.findall(
            r"public void (\w+)\(\) \{\s*(?:assumeTrue\(false\);\s*)?"
            r"byte\[\] compressed = \{(.*?)\};", src, re.S):
        data = bytes(int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{2})", body))
        out.append(pytest.param(data, id=name))
    return out


@pytest.mark.parametrize("data", _synth_vectors())
def test_java_synth_vectors(data):
    """Hand-crafted edge-case streams from the Java decoder's SynthTest."""
    assert_same(data)


# ---------------------------------------------------------------------------
# Round trips through the oracle compressor
# ---------------------------------------------------------------------------

def _make_inputs():
    rng = random.Random(1234)
    words = [b"the", b"quick", b"brown", b"fox", b"jumps", b"over", b"lazy",
             b"dog", b"brotli", b"compression", b"Python", b"window", b"\n"]
    text = b" ".join(rng.choice(words) for _ in range(6000))
    utf8 = ("Grüße, мир! 你好，世界。 ¿Qué tal? ✓ ★ ∑ — Ελληνικά עברית "
            * 150).encode("utf-8")
    binary = b"".join(
        (i * 2654435761 & 0xFFFFFFFF).to_bytes(4, "little") +
        (i & 0xFFFF).to_bytes(2, "big") for i in range(5000))
    return {
        "empty": b"",
        "one": b"x",
        "tiny": b"abc",
        "zeros": bytes(70000),
        "random": bytes(rng.getrandbits(8) for _ in range(20000)),
        "text": text,
        "repeat": b"0123456789abcdef!" * 2000,
        "utf8": utf8,
        "binary": binary,
    }


INPUTS = _make_inputs()


@pytest.mark.parametrize("name", sorted(INPUTS))
@pytest.mark.parametrize("quality", range(12))
def test_roundtrip_qualities(name, quality):
    data = INPUTS[name]
    for lgwin in (10, 16, 22, 24):
        mode = (quality + lgwin) % 3
        c = brotli.compress(data, quality=quality, lgwin=lgwin, mode=mode)
        assert brotli_pure.decompress(c) == data, (lgwin, mode)


@pytest.mark.parametrize("mode", [brotli.MODE_GENERIC, brotli.MODE_TEXT,
                                  brotli.MODE_FONT])
@pytest.mark.parametrize("lgwin", [10, 11, 12, 15, 18, 20, 24])
def test_roundtrip_modes_windows(mode, lgwin):
    data = INPUTS["text"] + INPUTS["binary"] + INPUTS["utf8"]
    for quality in (1, 5, 9, 11):
        c = brotli.compress(data, quality=quality, lgwin=lgwin, mode=mode)
        assert brotli_pure.decompress(c) == data


@pytest.mark.parametrize("lgblock", [16, 18, 24])
def test_roundtrip_lgblock(lgblock):
    data = INPUTS["text"] * 3 + INPUTS["random"]
    for quality in (4, 9, 11):
        c = brotli.compress(data, quality=quality, lgblock=lgblock)
        assert brotli_pure.decompress(c) == data


def test_large_multi_metablock():
    """Several MB with small windows and fast qualities -> many meta-blocks,
    window wrap-around, uncompressed meta-blocks."""
    rng = random.Random(99)
    chunks = []
    for i in range(40):
        kind = i % 4
        if kind == 0:
            chunks.append(bytes(rng.getrandbits(8) for _ in range(30000)))
        elif kind == 1:
            chunks.append(INPUTS["text"][:50000])
        elif kind == 2:
            chunks.append(bytes([i]) * 40000)
        else:
            chunks.append(INPUTS["utf8"][:20000])
    data = b"".join(chunks)
    for quality, lgwin in ((0, 10), (1, 16), (3, 18), (5, 24)):
        c = brotli.compress(data, quality=quality, lgwin=lgwin)
        assert brotli_pure.decompress(c) == data


def test_input_types():
    c = brotli.compress(b"hello hello hello world")
    assert brotli_pure.decompress(c) == b"hello hello hello world"
    assert brotli_pure.decompress(bytearray(c)) == b"hello hello hello world"
    assert brotli_pure.decompress(memoryview(c)) == b"hello hello hello world"
    assert brotli_pure.decompress(memoryview(b"xx" + c)[2:]) == \
        b"hello hello hello world"
    assert isinstance(brotli_pure.decompress(bytearray(c)), bytes)
    with pytest.raises(TypeError):
        brotli_pure.decompress("not bytes")
    with pytest.raises(TypeError):
        brotli.decompress("not bytes")


def test_error_is_brotli_error():
    assert brotli_pure.error is brotli_pure.BrotliError
    with pytest.raises(brotli_pure.error):
        brotli_pure.decompress(b"")
    with pytest.raises(brotli_pure.error):
        brotli_pure.decompress(b"\xff\xff\xff\xff")


# ---------------------------------------------------------------------------
# Hand-crafted streams
# ---------------------------------------------------------------------------

class BitWriter(object):
    """LSB-first bit writer, as the Brotli format requires."""

    def __init__(self):
        self.acc = 0
        self.nbits = 0

    def write(self, nbits, value):
        assert 0 <= value < (1 << nbits) or nbits == 0
        self.acc |= value << self.nbits
        self.nbits += nbits
        return self

    def align(self, pad=0):
        n = -self.nbits % 8
        return self.write(n, pad & ((1 << n) - 1))

    def write_bytes(self, data):
        assert self.nbits % 8 == 0
        for b in data:
            self.write(8, b)
        return self

    def getvalue(self):
        n = (self.nbits + 7) // 8
        return self.acc.to_bytes(n, "little") if n else b""


def write_wbits(w, wbits):
    if wbits == 16:
        w.write(1, 0)
    elif wbits >= 18:
        w.write(1, 1).write(3, wbits - 17)
    elif wbits == 17:
        w.write(1, 1).write(3, 0).write(3, 0)
    else:
        w.write(1, 1).write(3, 0).write(3, wbits - 8)


def write_mlen(w, mlen, is_last, uncompressed=False, nibbles=None):
    w.write(1, 1 if is_last else 0)
    if is_last:
        w.write(1, 0)  # ISEMPTY
    if nibbles is None:
        nibbles = max(4, ((mlen - 1).bit_length() + 3) // 4)
    w.write(2, nibbles - 4)
    w.write(4 * nibbles, mlen - 1)
    if not is_last:
        w.write(1, 1 if uncompressed else 0)


def write_last_empty(w):
    w.write(1, 1).write(1, 1)


def write_metadata(w, payload, reserved=0, skip_bytes=None):
    w.write(1, 0)       # ISLAST
    w.write(2, 3)       # MNIBBLES == 0 -> metadata
    w.write(1, reserved)
    if not payload:
        w.write(2, 0)
    else:
        n = len(payload) - 1
        nbytes = skip_bytes or max(1, (n.bit_length() + 7) // 8)
        w.write(2, nbytes)
        w.write(8 * nbytes, n)
    w.align()
    w.write_bytes(payload)


def stream(wbits=16):
    w = BitWriter()
    write_wbits(w, wbits)
    return w


@pytest.mark.parametrize("wbits", range(10, 25))
def test_window_bits_headers(wbits):
    w = stream(wbits)
    write_mlen(w, 5, False, uncompressed=True)
    w.align().write_bytes(b"hello")
    write_last_empty(w)
    data = w.getvalue()
    assert assert_same(data) == b"hello"


def test_large_window_header_rejected():
    # WBITS signature 0010001 + 6 bits (large-window brotli) is rejected
    # because the binding does not enable BROTLI_DECODER_PARAM_LARGE_WINDOW.
    for lw in (10, 24, 25, 30):
        w = BitWriter().write(1, 1).write(3, 0).write(3, 1).write(1, 0)
        w.write(6, lw)
        write_last_empty(w)
        assert assert_same(w.getvalue()) is None


def test_large_window_cli_stream_rejected():
    exe = os.path.join(REFERENCE, "out", "brotli")
    if not os.access(exe, os.X_OK):
        pytest.skip("reference CLI not built")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.bin")
        with open(src, "wb") as f:
            f.write(INPUTS["text"])
        for lw in (22, 25, 30):
            dst = os.path.join(d, "out%d.br" % lw)
            subprocess.check_call([exe, "-f", "-q", "5",
                                   "--large_window=%d" % lw, "-o", dst, src])
            with open(dst, "rb") as f:
                result = assert_same(f.read())
            # The CLI only emits the large-window format for lgwin > 24.
            if lw > 24:
                assert result is None
            else:
                assert result == INPUTS["text"]


def test_metadata_blocks():
    w = stream()
    write_metadata(w, b"")
    write_metadata(w, b"some metadata")
    write_metadata(w, bytes(300))
    write_mlen(w, 3, False, uncompressed=True)
    w.align().write_bytes(b"abc")
    write_metadata(w, b"\x00\x01")
    write_last_empty(w)
    assert assert_same(w.getvalue()) == b"abc"


def test_metadata_errors():
    w = stream()
    write_metadata(w, b"x", reserved=1)
    write_last_empty(w)
    assert assert_same(w.getvalue()) is None
    # MSKIPBYTES = 2 but the high byte is zero: "exuberant" length.
    w = stream()
    write_metadata(w, b"xyz", skip_bytes=2)
    write_last_empty(w)
    assert assert_same(w.getvalue()) is None
    # Truncated metadata payload.
    w = stream()
    write_metadata(w, b"0123456789")
    write_last_empty(w)
    assert assert_same(w.getvalue()[:-3]) is None


def test_last_metadata_block():
    # ISLAST=1, ISEMPTY=0, MNIBBLES=0: a metadata meta-block can be last.
    w = stream()
    w.write(1, 1).write(1, 0).write(2, 3).write(1, 0).write(2, 1).write(8, 2)
    w.align().write_bytes(b"md!")
    assert assert_same(w.getvalue()) == b""


def test_padding_bits():
    assert assert_same(b"\x06") == b""
    for bit in range(3, 8):
        assert assert_same(bytes([0x06 | (1 << bit)])) is None
    # Non-zero padding before an uncompressed meta-block (PADDING_1).
    w = stream()
    write_mlen(w, 2, False, uncompressed=True)
    good = BitWriter()
    write_wbits(good, 16)
    write_mlen(good, 2, False, uncompressed=True)
    pad = -good.nbits % 8
    assert pad > 0
    w.align(pad=1).write_bytes(b"hi")
    write_last_empty(w)
    assert assert_same(w.getvalue()) is None
    good.align().write_bytes(b"hi")
    write_last_empty(good)
    assert assert_same(good.getvalue()) == b"hi"


def test_exuberant_nibble_and_trailing_data():
    w = stream()
    write_mlen(w, 3, False, uncompressed=True, nibbles=5)
    w.align().write_bytes(b"abc")
    write_last_empty(w)
    assert assert_same(w.getvalue()) is None
    c = brotli.compress(b"payload" * 10)
    assert assert_same(c) == b"payload" * 10
    assert assert_same(c + b"\x00") is None
    assert assert_same(c + c) is None
    assert assert_same(c[:-1]) is None


def test_uncompressed_blocks_and_window_wrap():
    rng = random.Random(5)
    w = stream(10)
    expected = b""
    for _ in range(12):
        chunk = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 900)))
        write_mlen(w, len(chunk), False, uncompressed=True)
        w.align().write_bytes(chunk)
        expected += chunk
    write_last_empty(w)
    assert assert_same(w.getvalue()) == expected


# -- streams made of static dictionary references ---------------------------

def _distance_symbol(distance):
    """(symbol, nbits, extra) for NPOSTFIX = 0, NDIRECT = 0."""
    for sym in range(16, 64):
        i = sym - 16
        bits = 1 + (i >> 1)
        offset = ((2 + (i & 1)) << bits) - 4 + 1
        if offset <= distance < offset + (1 << bits):
            return sym, bits, distance - offset
    raise ValueError(distance)


def _command_symbol(copy_len):
    """Insert-and-copy symbol with insert length 0 and explicit distance."""
    for sym, (ie, ce, dc, _ctx, io, co) in enumerate(CMD_LUT):
        if ie == 0 and io == 0 and dc == -1 and co <= copy_len < co + (1 << ce):
            return sym, ce, copy_len - co
    raise ValueError(copy_len)


def _dictionary_stream(word_len, refs, wbits=24):
    """One meta-block emitting the given (word_idx, transform_idx) references
    of length ``word_len``; returns (stream, expected output per our model)."""
    sym, copy_nbits, copy_extra = _command_symbol(word_len)
    shift = SIZE_BITS_BY_LENGTH[word_len]
    expected = bytearray()
    body = BitWriter()
    for word_idx, transform_idx in refs:
        max_distance = min(len(expected), (1 << wbits) - 16)
        distance = max_distance + 1 + ((transform_idx << shift) | word_idx)
        dsym, dbits, dextra = _distance_symbol(distance)
        body.write(copy_nbits, copy_extra)
        for k in range(5, -1, -1):  # 6-bit canonical code, MSB first
            body.write(1, (dsym >> k) & 1)
        body.write(dbits, dextra)
        off = OFFSETS_BY_LENGTH[word_len] + word_idx * word_len
        expected += tf.transform_word(DICT[off:off + word_len], transform_idx)

    w = stream(wbits)
    write_mlen(w, len(expected), True)
    w.write(1, 0).write(1, 0).write(1, 0)  # NBLTYPESL/I/D = 1
    w.write(2, 0).write(4, 0)              # NPOSTFIX = 0, NDIRECT = 0
    w.write(2, 0)                          # context mode of block type 0
    w.write(1, 0).write(1, 0)              # NTREESL = NTREESD = 1
    # Literal code: simple, 1 symbol ('a'), never used.
    w.write(2, 1).write(2, 0).write(8, ord("a"))
    # Command code: simple, 1 symbol (10-bit symbols for alphabet 704).
    w.write(2, 1).write(2, 0).write(10, sym)
    # Distance code: complex code, all 64 symbols with length 6.  The code
    # length code has a single used symbol (6) => 0-bit codes.
    w.write(2, 0)  # HSKIP = 0
    static_code = {0: (2, 0), 1: (4, 7), 2: (3, 3), 3: (2, 2), 4: (2, 1),
                   5: (4, 15)}
    for idx in (1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        n, v = static_code[1 if idx == 6 else 0]
        w.write(n, v)
    w.write(body.nbits, body.acc)
    w.align()
    return w.getvalue(), bytes(expected)


def _copy_stream(wbits, prefix, copy_len, distances):
    """Uncompressed ``prefix`` meta-block, then one compressed meta-block of
    commands (insert 0, ``copy_len``, explicit distance).  Distances above
    the current maximum are static dictionary references.  Returns
    (stream, expected output per our model)."""
    sym, copy_nbits, copy_extra = _command_symbol(copy_len)
    expected = bytearray(prefix)
    body = BitWriter()
    for distance in distances:
        max_distance = min(len(expected), (1 << wbits) - 16)
        dsym, dbits, dextra = _distance_symbol(distance)
        body.write(copy_nbits, copy_extra)
        for k in range(5, -1, -1):
            body.write(1, (dsym >> k) & 1)
        body.write(dbits, dextra)
        if distance <= max_distance:
            for _ in range(copy_len):
                expected.append(expected[-distance])
        else:
            address = distance - max_distance - 1
            shift = SIZE_BITS_BY_LENGTH[copy_len]
            word_idx = address & ((1 << shift) - 1)
            off = OFFSETS_BY_LENGTH[copy_len] + word_idx * copy_len
            expected += tf.transform_word(DICT[off:off + copy_len],
                                          address >> shift)
    w = stream(wbits)
    if prefix:
        write_mlen(w, len(prefix), False, uncompressed=True)
        w.align().write_bytes(prefix)
    write_mlen(w, len(expected) - len(prefix), True)
    w.write(3, 0).write(6, 0).write(2, 0).write(2, 0)
    w.write(2, 1).write(2, 0).write(8, ord("a"))
    w.write(2, 1).write(2, 0).write(10, sym)
    w.write(2, 0)
    static_code = {0: (2, 0), 1: (4, 7)}
    for idx in (1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        w.write(*static_code[1 if idx == 6 else 0])
    w.write(body.nbits, body.acc)
    return w.align().getvalue(), bytes(expected)


@pytest.mark.parametrize("wbits", [10, 11, 12, 16])
def test_window_edge_distances(wbits):
    """Back-references right at the window limit (2**wbits - 16) vs. the
    first dictionary reference just beyond it, after the window wrapped."""
    rng = random.Random(wbits)
    prefix = bytes(rng.getrandbits(8) for _ in range(5000))
    limit = min(len(prefix), (1 << wbits) - 16)
    distances = [limit, limit - 1, 1, 2, 3, 4, 5, limit + 1, limit,
                 limit + 2 + (9 << 10), 7, limit + 1 + (3 << 10) + 77]
    for copy_len in (4, 5, 9):
        data, expected = _copy_stream(wbits, prefix, copy_len, distances)
        assert assert_same(data) == expected
    # One past the maximum, with a copy length that has no dictionary words.
    data, _ = _copy_stream(wbits, prefix, 3, [limit + 1])
    assert assert_same(data) is None


@pytest.mark.parametrize("word_len", range(4, 25))
def test_dictionary_all_transforms(word_len):
    """Every transform applied to several dictionary words of each length,
    including words with multi-byte UTF-8 (UppercaseFirst/All paths)."""
    rng = random.Random(word_len)
    nwords = 1 << SIZE_BITS_BY_LENGTH[word_len]
    candidates = [rng.randrange(nwords) for _ in range(3)]
    # Prefer words that start with or contain bytes >= 0xC0 as well.
    base = OFFSETS_BY_LENGTH[word_len]
    utf = [i for i in range(nwords)
           if max(DICT[base + i * word_len:base + (i + 1) * word_len]) >= 0xC0]
    candidates += utf[:2] + utf[-2:]
    refs = [(wi, t) for wi in candidates for t in range(121)]
    refs.append((candidates[0], 0))  # end with a non-empty word
    data, expected = _dictionary_stream(word_len, refs)
    assert assert_same(data) == expected


def test_zero_length_dictionary_words():
    """OmitLast9 on a 4-byte word yields an empty word; C accepts it (its
    `len == 0 && distance <= 120` check cannot trigger here)."""
    data, expected = _dictionary_stream(4, [(5, 64), (5, 64), (9, 54), (7, 0)])
    assert assert_same(data) == expected


def test_dictionary_reference_errors():
    # transform index >= 121 -> error.
    data, _ = _dictionary_stream(4, [(0, 0)])
    assert assert_same(data) is not None
    for word_len in (4, 10, 24):
        shift = SIZE_BITS_BY_LENGTH[word_len]
        sym, cnb, cex = _command_symbol(word_len)
        dsym, dbits, dextra = _distance_symbol(1 + (121 << shift))
        w = stream(24)
        write_mlen(w, word_len, True)
        w.write(3, 0).write(6, 0).write(2, 0).write(2, 0)
        w.write(2, 1).write(2, 0).write(8, 97)
        w.write(2, 1).write(2, 0).write(10, sym)
        w.write(2, 1).write(2, 0).write(6, dsym)  # 1-symbol distance code
        w.write(cnb, cex).write(dbits, dextra)
        assert assert_same(w.align().getvalue()) is None


def test_transforms_api():
    assert tf.NUM_RFC_TRANSFORMS == 121
    assert tf.transform_word(b"time", 0) == b"time"
    assert tf.transform_word(b"time", 1) == b"time "
    assert tf.transform_word(b"time", 4) == b"Time "
    assert tf.transform_word(b"time", 9) == b"Time"
    assert tf.transform_word(b"time", 44) == b"TIME"
    assert tf.transform_word(b"time", 3) == b"ime"
    assert tf.transform_word(b"time", 12) == b"tim"
    assert tf.transform_word(b"time", 64) == b""
    assert tf.transform_word("ölig".encode(), 44) == bytes(
        [0xC3, 0xB6 ^ 32]) + b"LIG"
    dst = bytearray(b"xy")
    n = tf.transform_dictionary_word(dst, b"time", 4, tf.RFC_TRANSFORMS, 2)
    assert dst == b"xy time " and n == 6
    # Shift transforms (only used by custom shared dictionaries).
    word = bytearray(b"a\xc3\xa9\xe2\x82\xac\xf0\x9f\x98\x80")
    assert tf.shift(word, 0, len(word), 1) == 1 and word[0] == ord("b")


# ---------------------------------------------------------------------------
# Differential fuzzing
# ---------------------------------------------------------------------------

def _fuzz_corpus(rng):
    corpus = []
    texts = [INPUTS["text"][:3000], INPUTS["utf8"][:1500],
             INPUTS["binary"][:2000], INPUTS["random"][:600], b"", b"a",
             b"abc" * 300, INPUTS["repeat"][:1000]]
    for t in texts:
        for q in (0, 1, 2, 4, 5, 9, 10, 11):
            for lgwin in (10, 16, 22):
                corpus.append(brotli.compress(t, quality=q, lgwin=lgwin,
                                              mode=rng.randrange(3)))
    for path in _vector_files():
        if os.path.getsize(path) < 3000:
            with open(path, "rb") as f:
                corpus.append(f.read())
    for p in _synth_vectors():
        if len(p.values[0]) < 3000:
            corpus.append(p.values[0])
    return corpus


def _mutate(rng, c):
    c = bytearray(c)
    k = rng.randrange(7)
    if k == 0 and c:
        for _ in range(rng.randint(1, 4)):
            i = rng.randrange(len(c))
            c[i] ^= 1 << rng.randrange(8)
    elif k == 1 and c:
        # Flip a bit near the end (headers survive, body/padding changes).
        i = rng.randrange(max(0, len(c) - 8), len(c))
        c[i] ^= 1 << rng.randrange(8)
    elif k == 2:
        c = c[:rng.randrange(len(c) + 1)]
    elif k == 3:
        c += bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 3)))
    elif k == 4 and c:
        i = rng.randrange(len(c))
        del c[i:i + rng.randint(1, 4)]
    elif k == 5 and c:
        i = rng.randrange(len(c))
        c[i] = rng.getrandbits(8)
    else:
        i = rng.randrange(len(c) + 1)
        c[i:i] = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 4)))
    return bytes(c)


@pytest.mark.parametrize("seed", range(FUZZ_SEEDS))
def test_differential_fuzz(seed):
    rng = random.Random(1000 + seed)
    corpus = _fuzz_corpus(rng)
    for _ in range(FUZZ_ITERATIONS):
        data = _mutate(rng, rng.choice(corpus))
        exp = oracle(data)
        try:
            got = brotli_pure.decompress(data)
        except brotli_pure.error:
            got = None
        assert got == exp, data.hex()


def test_differential_fuzz_random_bytes():
    rng = random.Random(77)
    for _ in range(FUZZ_ITERATIONS * 2):
        data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 40)))
        assert ours(data) == oracle(data), data.hex()


# ---------------------------------------------------------------------------
# Structure-aware fuzzing: valid headers with random Huffman codes, context
# maps (RLE / IMTF), block types and distance parameters, followed by random
# command bits.  Every prefix of the stream is compared with the oracle.
# ---------------------------------------------------------------------------

def _write_simple_code(w, rng, alphabet_size, symbols=None):
    """Writes a random simple prefix code; returns {symbol: (nbits, bits)}
    with ``bits`` in stream (LSB-first) order."""
    if symbols is None:
        nsym = rng.randint(1, min(4, alphabet_size))
        symbols = rng.sample(range(alphabet_size), nsym)
    nsym = len(symbols)
    max_bits = (alphabet_size - 1).bit_length()
    w.write(2, 1).write(2, nsym - 1)
    for sym in symbols:
        w.write(max_bits, sym)
    v = list(symbols)
    if nsym == 1:
        return {v[0]: (0, 0)}
    if nsym == 2:
        a, b = sorted(v)
        return {a: (1, 0), b: (1, 1)}
    if nsym == 3:
        lo, hi = sorted(v[1:])
        return {v[0]: (1, 0), lo: (2, 1), hi: (2, 3)}
    tree_select = rng.randrange(2)
    w.write(1, tree_select)
    if tree_select == 0:
        s = sorted(v)
        return {s[0]: (2, 0), s[1]: (2, 2), s[2]: (2, 1), s[3]: (2, 3)}
    lo, hi = sorted(v[2:])
    return {v[0]: (1, 0), v[1]: (2, 1), lo: (3, 3), hi: (3, 7)}


def _write_context_map(w, rng, size, ntrees):
    """Writes a context map of ``size`` entries referring to ``ntrees``
    trees, with random RLE parameter and IMTF flag."""
    if ntrees == 1:
        w.write(1, 0)  # DecodeVarLenUint8 == 0
        return
    n = ntrees - 1
    nb = n.bit_length() - 1
    w.write(1, 1).write(3, nb)
    if nb:
        w.write(nb, n - (1 << nb))
    rle = rng.randrange(4)  # max run length prefix (0 = no RLE)
    if rle:
        w.write(1, 1).write(4, rle - 1)
    else:
        w.write(1, 0)
    alphabet = ntrees + rle
    syms = rng.sample(range(alphabet), rng.randint(1, min(4, alphabet)))
    if rng.random() < 0.9 and not [x for x in syms if x == 0 or x > rle]:
        syms[0] = 0
    codes = _write_simple_code(w, rng, alphabet, syms)
    plain = [x for x in syms if x == 0 or x > rle]
    i = 0
    while i < size:
        sym = rng.choice(syms)
        if 0 < sym <= rle:
            reps = rng.randrange(1 << sym)
            if i + (1 << sym) + reps > size and plain and rng.random() < 0.95:
                sym = rng.choice(plain)
        w.write(*codes[sym])
        if 0 < sym <= rle:
            w.write(sym, reps)
            i += (1 << sym) + reps
        else:
            i += 1
    w.write(1, rng.randrange(2))  # IMTF


_SHORT_COMMANDS = [i for i, e in enumerate(CMD_LUT) if e[4] <= 2 and e[5] <= 4]


def _random_structured_stream(rng):
    w = BitWriter()
    wbits = rng.randint(10, 24)
    write_wbits(w, wbits)
    # Leading uncompressed / metadata meta-blocks (the uncompressed data
    # gives back-references something to point to).
    for k in range(rng.randrange(4)):
        if k == 0 or rng.random() < 0.5:
            chunk = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 300)))
            write_mlen(w, len(chunk), False, uncompressed=True)
            w.align().write_bytes(chunk)
        else:
            write_metadata(w, bytes(rng.randint(0, 5)))
    mlen = rng.choice((1, 2, 5, 17, 64, 300, 1000, 3000))
    write_mlen(w, mlen, True)
    nbt = []
    for _ in range(3):
        n = rng.choice((1, 1, 2, 3, 5))
        nbt.append(n)
        if n == 1:
            w.write(1, 0)
            continue
        k = n - 1
        b = k.bit_length() - 1
        w.write(1, 1).write(3, b)
        if b:
            w.write(b, k - (1 << b))
        _write_simple_code(w, rng, n + 2)
        len_codes = _write_simple_code(w, rng, 26)
        sym = rng.choice(list(len_codes))
        w.write(*len_codes[sym])
        nbits = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7,
                 8, 9, 10, 11, 12, 13, 24)[sym]
        w.write(nbits, rng.getrandbits(nbits))
    npostfix = rng.randrange(4)
    ndirect = rng.randrange(16)
    w.write(2, npostfix).write(4, ndirect)
    for _ in range(nbt[0]):
        w.write(2, rng.randrange(4))
    nlit = rng.choice((1, 1, 2, 3, 4))
    _write_context_map(w, rng, nbt[0] * 64, nlit)
    ndist = rng.choice((1, 1, 2, 3))
    _write_context_map(w, rng, nbt[2] * 4, ndist)
    dist_alphabet = 16 + (ndirect << npostfix) + (48 << npostfix)
    for _ in range(nlit):
        _write_simple_code(w, rng, 256)
    for _ in range(nbt[1]):
        # Mostly short insert/copy lengths, so that meta-blocks can complete.
        if rng.random() < 0.8:
            syms = rng.sample(_SHORT_COMMANDS, rng.randint(1, 4))
        else:
            syms = rng.sample(range(704), rng.randint(1, 4))
        _write_simple_code(w, rng, 704, syms)
    for _ in range(ndist):
        # Mostly "last distance" codes and short direct distances.
        if rng.random() < 0.7:
            pool = list(range(16 + (ndirect << npostfix)))
            syms = rng.sample(pool, rng.randint(1, min(4, len(pool))))
            _write_simple_code(w, rng, dist_alphabet, syms)
        else:
            _write_simple_code(w, rng, dist_alphabet)
    header_len = (w.nbits + 7) // 8
    w.write(8 * rng.randint(4, 40), 0)  # reserve, then fill with random bits
    data = bytearray(w.getvalue())
    for i in range(header_len, len(data)):
        data[i] = rng.getrandbits(8) if rng.random() < 0.8 else 0
    return bytes(data), header_len


@pytest.mark.parametrize("seed", range(FUZZ_SEEDS))
def test_structured_fuzz(seed):
    rng = random.Random(5000 + seed)
    successes = 0
    for _ in range(FUZZ_ITERATIONS // 25):
        data, header_len = _random_structured_stream(rng)
        for n in range(max(0, header_len - 2), len(data) + 1):
            exp = assert_same(data[:n])
            successes += exp is not None
    assert successes > 0
