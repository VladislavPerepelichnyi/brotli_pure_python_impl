"""Development tests for the quality 10 / 11 encoder components (enc-q11).

* Component tests compare against golden outputs of the reference C
  functions (tests/data/enc_q11_golden.bin, regenerated with
  tools/enc_q11_harness/gen_golden.sh): static dictionary matches, H10
  matches, Zopfli command lists, BrotliBuildMetaBlock results.
* End-to-end tests compare ``brotli_pure.compress(quality=11 / 10)`` with the
  C oracle (``brotli`` module built from google/brotli v1.2.1) byte for byte
  and round-trip through the oracle decoder.

Large inputs are marked ``slow`` and only run when the environment variable
``BROTLI_PURE_SLOW`` is set (pure-Python quality 11 is ~50-100 us / byte).
"""

import base64
import json
import os
import random
import zlib

import pytest

import brotli_pure

HERE = os.path.dirname(os.path.abspath(__file__))
TESTDATA = os.path.join(HERE, "..", "..", "..", "brotli_reference", "tests",
                        "testdata")
if not os.path.isdir(TESTDATA):
    TESTDATA = os.environ.get("BROTLI_TESTDATA", TESTDATA)

SLOW = bool(os.environ.get("BROTLI_PURE_SLOW"))
slow = pytest.mark.skipif(not SLOW, reason="set BROTLI_PURE_SLOW=1 to run")


def _golden():
    with open(os.path.join(HERE, "data", "enc_q11_golden.bin"), "rb") as f:
        return json.loads(zlib.decompress(f.read()).decode())


GOLDEN = _golden()
G1 = base64.b64decode(GOLDEN["g1"])
G2 = base64.b64decode(GOLDEN["g2"])
CAP = 1 << 24


def _lines(s):
    return s.rstrip("\n").split("\n")


def _params(quality, lgwin, font):
    from brotli_pure.enc.metablock import brotli_init_distance_params
    from brotli_pure.enc.params import BrotliEncoderParams
    p = BrotliEncoderParams()
    p.quality = quality
    p.lgwin = lgwin
    p.mode = 2 if font else 0
    brotli_init_distance_params(p.dist, 1 if font else 0, 12 if font else 0,
                                False)
    return p


def _buffer(data):
    buf = bytearray(CAP + 1024)
    buf[:len(data)] = data
    return buf


# --- component tests --------------------------------------------------------

@pytest.mark.parametrize("minlen", [4, 9])
def test_static_dict_matches_golden(minlen):
    from brotli_pure.enc.static_dict import (
        K_INVALID_MATCH, brotli_find_all_static_dictionary_matches)
    data = G1
    n = len(data)
    buf = data + bytes(64)
    out = []
    for p in range(n - 3):
        m = [K_INVALID_MATCH] * 38
        r = brotli_find_all_static_dictionary_matches(None, buf, p, minlen,
                                                      n - p, m)
        out.append("%d %d%s" % (p, 1 if r else 0, "".join(
            " %d:%d" % (i, m[i]) for i in range(38) if m[i] != K_INVALID_MATCH)))
    assert out == _lines(GOLDEN["static_dict_g1_%d" % minlen])


def test_static_dict_lut_shape():
    from brotli_pure.enc import static_dict as sd
    assert len(sd.K_STATIC_DICTIONARY_BUCKETS) == 32768
    assert len(sd.DICT_WORD_LEN) == len(sd.DICT_WORD_IDX) == 31705
    # Every bucket chain ends with an item carrying the end bit (0x80).
    for b in sd.K_STATIC_DICTIONARY_BUCKETS:
        if b:
            i = b
            while not sd.DICT_WORD_LEN[i] & 0x80:
                i += 1
            assert 4 <= sd.DICT_WORD_LEN[i] & 0x1F <= 24


def test_h10_find_all_matches_golden():
    from brotli_pure.enc.hash_to_binary_tree import H10
    data = G2
    n = len(data)
    buf = _buffer(data)
    mask = CAP - 1
    p = _params(11, 16, False)
    h = H10(None, p, True, n)
    h.prepare(True, n, buf)
    mbl = (1 << 16) - 16
    store_end = n - 128 + 1
    out = []
    i = 0
    while i + 3 < n:
        md = min(i, mbl)
        dd, ll = [], []
        num = h.find_all_matches(None, buf, mask, i, n - i, md, md, p, dd, ll)
        out.append("%d:%s" % (i, "".join(" %d/%d" % x for x in zip(dd, ll))))
        if num and ll[-1] >> 5 > 325:
            h.store_range(buf, mask, i + 1, min(i + (ll[-1] >> 5), store_end))
            i += (ll[-1] >> 5) - 1
        i += 1
    assert out == _lines(GOLDEN["h10_g2_w16_q11"])


def _run_zopfli(data, lgwin, quality, font):
    from brotli_pure.enc import backward_references_hq as bh
    from brotli_pure.enc.hash import Hasher, hasher_setup
    n = len(data)
    buf = _buffer(data)
    p = _params(quality, lgwin, font)
    h = Hasher()
    hasher_setup(h, p, buf, 0, n, True)
    dc = [4, 11, 15, 16] + [0] * 12
    cmds = []
    fn = (bh.brotli_create_hq_zopfli_backward_references if quality == 11
          else bh.brotli_create_zopfli_backward_references)
    lil, nc, nl = fn(n, 0, buf, CAP - 1, None, p, h, dc, 0, cmds, 0, 0)
    return p, buf, cmds, (nc, lil, nl, dc)


@pytest.mark.parametrize("key,lgwin,quality,font", [
    ("zopfli_g2_w16_q11", 16, 11, False),
    ("zopfli_g2_w12_q10", 12, 10, False),
    ("zopfli_g2_w18_q11_font", 18, 11, True),
])
def test_zopfli_commands_golden(key, lgwin, quality, font):
    _, _, cmds, (nc, lil, nl, dc) = _run_zopfli(G2, lgwin, quality, font)
    out = ["%d %d %d %d %d" % (c.insert_len, c.copy_len, c.dist_extra,
                               c.cmd_prefix, c.dist_prefix) for c in cmds]
    out.append("end %d %d %d %d %d %d %d" % ((nc, lil, nl) + tuple(dc[:4])))
    assert out == _lines(GOLDEN[key])


@pytest.mark.parametrize("key,lgwin,font,cmode", [
    ("metablock_g2_w16_q11_m0_c2", 16, False, 2),
    ("metablock_g2_w18_q11_m1_c3", 18, True, 3),
])
def test_build_meta_block_golden(key, lgwin, font, cmode):
    from brotli_pure.enc.command import init_insert_command
    from brotli_pure.enc.metablock import MetaBlockSplit
    from brotli_pure.enc.metablock_hq import brotli_build_meta_block
    p, buf, cmds, (nc, lil, nl, dc) = _run_zopfli(G2, lgwin, 11, font)
    if lil > 0:
        cmds.append(init_insert_command(lil))
    mb = MetaBlockSplit()
    brotli_build_meta_block(buf, 0, CAP - 1, p, 0, 0, cmds, len(cmds), cmode,
                            mb)
    o = ["dist %d %d %d %d" % (p.dist.distance_postfix_bits,
                               p.dist.num_direct_distance_codes,
                               p.dist.alphabet_size_limit,
                               p.dist.max_distance)]
    o += ["%d %d %d %d %d" % (c.insert_len, c.copy_len, c.dist_extra,
                              c.cmd_prefix, c.dist_prefix) for c in cmds]
    for name, s in (("lit", mb.literal_split), ("cmd", mb.command_split),
                    ("dst", mb.distance_split)):
        o.append("%s %d %d:%s" % (name, s.num_types, s.num_blocks, "".join(
            " %d/%d" % (s.types[i], s.lengths[i])
            for i in range(s.num_blocks))))
    o.append("lcm" + "".join(" %d" % x for x in mb.literal_context_map))
    o.append("dcm" + "".join(" %d" % x for x in mb.distance_context_map))
    assert mb.literal_histograms_size == len(mb.literal_histograms)
    assert mb.distance_histograms_size == len(mb.distance_histograms)
    assert mb.command_histograms_size == len(mb.command_histograms)
    for name, hs in (("lh", mb.literal_histograms),
                     ("ch", mb.command_histograms),
                     ("dh", mb.distance_histograms)):
        o.append("%s %d" % (name, len(hs)))
        for hh in hs:
            o.append("%d:%s" % (hh.total_count, "".join(
                " %d=%d" % (k, v) for k, v in enumerate(hh.data) if v)))
    assert o == _lines(GOLDEN[key])


def _is_mostly_utf8_ref(data, min_fraction):
    """Straightforward transcription of BrotliIsMostlyUTF8."""
    from brotli_pure.enc.utf8_util import brotli_parse_as_utf8
    size_utf8 = 0
    i = 0
    while i < len(data):
        symbol, n = brotli_parse_as_utf8(data, i, len(data) - i)
        i += n
        if symbol < 0x110000:
            size_utf8 += n
    return size_utf8 > min_fraction * len(data)


def test_is_mostly_utf8():
    from brotli_pure.enc.utf8_util import brotli_is_mostly_utf8, kMinUTF8Ratio
    rnd = random.Random(3)
    samples = [b"", b"\x00", b"abc", "żółć".encode(), b"\xc3\x28",
               "😀x".encode(), b"\xf4\x90\x80\x80", b"\xed\xa0\x80" * 3]
    for _ in range(200):
        n = rnd.randrange(1, 60)
        samples.append(bytes(rnd.choice(b"a\x00\x80\xbf\xc3\xe2\xf0\x9f\xa9")
                             for _ in range(n)))
    for s in samples:
        for frac in (kMinUTF8Ratio, 0.5, 0.0):
            # Also exercise the ring-buffer wrap-around path.
            mask = 63
            buf = bytearray(64 + 16)
            pos = 50
            for k, b in enumerate(s[:64]):
                buf[(pos + k) & mask] = b
            ln = min(len(s), 64)
            assert brotli_is_mostly_utf8(buf, pos, mask, ln, frac) == \
                _is_mostly_utf8_ref(s[:64], frac)


def test_literal_costs_are_float32():
    from array import array
    from brotli_pure.enc.literal_cost import \
        brotli_estimate_bit_costs_for_literals
    data = bytearray(G2)
    costs = [0.0] * len(data)
    brotli_estimate_bit_costs_for_literals(0, len(data), (1 << 20) - 1,
                                           data + bytes(1 << 20), [0] * 768,
                                           costs)
    assert costs == list(array("f", costs))
    assert all(c >= 0.5 for c in costs)


# --- end-to-end tests vs the C oracle ----------------------------------------

def _text(n, seed):
    rnd = random.Random(seed)
    words = (b"the of and to in is was that for it with as his on be at by "
             b"this had not are but from or have an they which one you were "
             b"her all she there would their we him been has when who will "
             b"Information Development Programming international").split()
    out = bytearray()
    while len(out) < n:
        w = rnd.choice(words)
        r = rnd.random()
        if r < 0.1:
            w = w.upper()
        elif r < 0.25:
            w = w.capitalize()
        out += w + rnd.choice([b" ", b" ", b" ", b", ", b". ", b"\n", b'="'])
    return bytes(out[:n])


def _binary(n, seed):
    rnd = random.Random(seed)
    out = bytearray()
    while len(out) < n:
        rec = bytes([rnd.randrange(4), 0, 0, rnd.randrange(256)]) + \
            rnd.randrange(1 << 16).to_bytes(2, "little") + b"\x00\x00"
        out += rec
    return bytes(out[:n])


def _multilingual(n):
    s = ("Привет, мир! 你好，世界。 Grüße aus Köln — αβγδε. שלום עולם. "
         "Ελληνικά, 日本語のテキスト, हिन्दी पाठ. ").encode("utf-8")
    return (s * (n // len(s) + 1))[:n]


def _repeated(n, seed):
    rnd = random.Random(seed)
    out = bytearray()
    while len(out) < n:
        out += bytes([rnd.randrange(3)]) * rnd.randrange(1, 40)
        out += b"abcabcabcabc"[:rnd.randrange(12)]
    return bytes(out[:n])


SMALL_INPUTS = {
    "empty": b"",
    "one": b"x",
    "two": b"\x00\xff",
    "three": b"abc",
    "quickfox": b"The quick brown fox jumps over the lazy dog",
    "zeros": bytes(1000),
    "random": bytes(random.Random(1).getrandbits(8) for _ in range(1500)),
    "text": _text(4000, 2),
    "binary": _binary(3000, 4),
    "multilingual": _multilingual(3000),
    "repeated": _repeated(4000, 5),
}


def _oracle():
    return pytest.importorskip("brotli")


def _check(data, quality, lgwin, mode):
    brotli = _oracle()
    got = brotli_pure.compress(data, mode=mode, quality=quality, lgwin=lgwin)
    want = brotli.compress(data, mode=mode, quality=quality, lgwin=lgwin)
    assert brotli.decompress(got) == data
    assert got == want, "q%d lgwin=%d mode=%d: %d vs %d bytes" % (
        quality, lgwin, mode, len(got), len(want))


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(SMALL_INPUTS))
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("lgwin", [10, 16, 22, 24])
def test_q11_small_inputs(name, mode, lgwin):
    _check(SMALL_INPUTS[name], 11, lgwin, mode)


@pytest.mark.oracle
@pytest.mark.parametrize("name", sorted(SMALL_INPUTS))
@pytest.mark.parametrize("lgwin", [10, 22])
def test_q10_small_inputs(name, lgwin):
    _check(SMALL_INPUTS[name], 10, lgwin, 0)


@pytest.mark.oracle
@pytest.mark.parametrize("lgwin", [11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23])
def test_q11_all_windows(lgwin):
    _check(_text(2500, lgwin) + _binary(800, lgwin), 11, lgwin, lgwin % 3)


@pytest.mark.oracle
def test_q11_window_wrap_small_lgwin():
    # 3 x 1 KiB window with lgwin 10: matches beyond the window must not be
    # used and the ring buffer wraps.
    data = _text(9000, 9)
    _check(data, 11, 10, 0)


def _testdata(name):
    path = os.path.join(TESTDATA, name)
    if not os.path.exists(path):
        pytest.skip("testdata not found: %s" % path)
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["10x10y", "64x", "quickfox", "monkey",
                                  "cp852-utf8", "cp1251-utf16le",
                                  "random_chunks", "empty"])
def test_q11_testdata_small(name):
    _check(_testdata(name), 11, 22, 0)


@slow
@pytest.mark.slow
@pytest.mark.oracle
@pytest.mark.parametrize("name", ["alice29.txt", "asyoulik.txt",
                                  "backward65536", "compressed_file",
                                  "compressed_repeated", "lcet10.txt",
                                  "mapsdatazrh", "plrabn12.txt",
                                  "quickfox_repeated"])
def test_q11_testdata_large(name):
    _check(_testdata(name), 11, 22, 0)


@slow
@pytest.mark.slow
@pytest.mark.oracle
@pytest.mark.parametrize("lgwin,mode", [(16, 0), (18, 1), (22, 2), (24, 0),
                                        (10, 0)])
def test_q11_multi_block(lgwin, mode):
    # > 256 KiB: crosses input-block / metablock boundaries.
    data = _text(150000, 11) + _binary(60000, 12) + _multilingual(80000) + \
        bytes(random.Random(13).getrandbits(8) for _ in range(20000))
    _check(data, 11, lgwin, mode)


@slow
@pytest.mark.slow
@pytest.mark.oracle
def test_q10_multi_block():
    data = _text(200000, 21) + _repeated(100000, 22)
    _check(data, 10, 18, 0)


@slow
@pytest.mark.slow
@pytest.mark.oracle
def test_q11_bb_binast_prefix():
    _check(_testdata("bb.binast")[:400000], 11, 22, 0)
