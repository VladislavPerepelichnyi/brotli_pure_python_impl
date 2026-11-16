"""Development tests for the quality-5 encoder path (enc-q5 branch).

Byte-exact comparison against the C reference Python binding (``brotli``,
built from google/brotli v1.2.1) plus round trips through its decoder.

Environment variables:
  BROTLI_PURE_SLOW=1   full sweep: every lgwin in 10..24, every mode, all
                       testdata files and multi-megabyte inputs.
"""

import os
import random
import struct

import pytest

import brotli_pure
from brotli_pure.enc.encode import UnsupportedQualityError

brotli = pytest.importorskip('brotli')

SLOW = bool(os.environ.get('BROTLI_PURE_SLOW'))

TESTDATA = os.path.join(os.path.dirname(__file__), '..', '..',
                        'brotli_reference', 'tests', 'testdata')
if not os.path.isdir(TESTDATA):
    TESTDATA = '/home/vlad/thesis/wasm/pure_python_impl/brotli_reference/tests/testdata'

WORDS = ('the of and to in is was that for it with as his on be at by had '
         'this not are but from or have an they which one you were her all '
         'brotli compression window block hasher literal command distance '
         'Huffman context entropy metablock').split()


def _rand_bytes(rng, n):
    return bytes(rng.getrandbits(8) for _ in range(n))


def _text(rng, n):
    out = []
    size = 0
    while size < n:
        w = rng.choice(WORDS)
        if rng.random() < 0.08:
            w = w.capitalize() + rng.choice(['.', ',', ';', '!\n', '\n'])
        out.append(w)
        size += len(w) + 1
    return ' '.join(out).encode()[:n]


def _structured_binary(rng, n):
    recs = []
    t = 0
    while len(recs) * 16 < n:
        t += rng.randrange(1, 50)
        recs.append(struct.pack('<IHHfI', t, rng.randrange(8), 0xBEEF,
                                rng.random() * 100.0, rng.randrange(1000)))
    return b''.join(recs)[:n]


def _utf8_multilingual(rng, n):
    parts = ['Привет, мир! ', 'Καλημέρα κόσμε. ', 'こんにちは世界。', '你好，世界！',
             'مرحبا بالعالم ', 'שלום עולם ', 'Grüße aus Köln; ', 'emoji 😀🎉 ',
             'hello world ', '१२३ नमस्ते ']
    out = []
    size = 0
    while size < n:
        s = rng.choice(parts).encode('utf-8')
        out.append(s)
        size += len(s)
    return b''.join(out)[:n]


def _aaaax(rng, n):
    return b''.join(b'aaaa' + bytes([rng.getrandbits(8)])
                    for _ in range(n // 5 + 1))[:n]


def _make_small_inputs():
    rng = random.Random(1234)
    return [
        ('empty', b''),
        ('one', b'x'),
        ('two', b'ab'),
        ('three', b'abc'),
        ('quickfox', b'The quick brown fox jumps over the lazy dog'),
        ('zeros1k', bytes(1000)),
        ('ff300', b'\xff' * 300),
        ('rand100', _rand_bytes(rng, 100)),
        ('rand5k', _rand_bytes(rng, 5000)),
        ('text3k', _text(rng, 3000)),
        ('text30k', _text(rng, 30000)),
        ('pattern', b'abcdefgh0123' * 700),
        ('utf8', _utf8_multilingual(rng, 20000)),
        ('struct', _structured_binary(rng, 24000)),
        ('aaaax', _aaaax(rng, 12000)),
        ('mixed', _rand_bytes(rng, 2000) + _text(rng, 6000) +
         _rand_bytes(rng, 2000) + bytes(3000) + _text(rng, 4000)),
    ]


def _make_large_inputs():
    rng = random.Random(99)
    return [
        ('text200k', _text(rng, 200000)),            # > one input block
        ('rand70k', _rand_bytes(rng, 70000)),        # uncompressible, 2 blocks
        ('zeros300k', bytes(300000)),
        ('utf8_150k', _utf8_multilingual(rng, 150000)),
        ('struct140k', _structured_binary(rng, 140000)),
        ('block_edge', _text(rng, 65536 * 2)),       # exact multiple of block
        ('block_edge1', _text(rng, 65536 + 1)),
        ('mixed300k', b''.join(
            _rand_bytes(rng, 3000) if i % 4 == 0 else _text(rng, 9000)
            for i in range(40))),
    ]


SMALL = _make_small_inputs()
LARGE = _make_large_inputs()

if SLOW:
    WINDOWS = list(range(10, 25))
else:
    WINDOWS = [10, 14, 16, 17, 18, 20, 22, 24]
MODES = [0, 1, 2]


def _check(data, quality=5, lgwin=22, mode=0):
    expected = brotli.compress(data, quality=quality, lgwin=lgwin, mode=mode)
    actual = brotli_pure.compress(data, quality=quality, lgwin=lgwin, mode=mode)
    assert brotli.decompress(actual) == data
    if actual != expected:
        diff = next((i for i, (a, b) in enumerate(zip(actual, expected))
                     if a != b), min(len(actual), len(expected)))
        pytest.fail('output differs from reference: len %d vs %d, first '
                    'difference at byte %d (q=%d lgwin=%d mode=%d)' % (
                        len(actual), len(expected), diff, quality, lgwin,
                        mode))


@pytest.mark.oracle
@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('lgwin', WINDOWS)
@pytest.mark.parametrize('name,data', SMALL, ids=[n for n, _ in SMALL])
def test_small_inputs_byte_exact(name, data, lgwin, mode):
    _check(data, 5, lgwin, mode)


@pytest.mark.oracle
@pytest.mark.parametrize('lgwin,mode', [(16, 0), (22, 1), (18, 2)] if not SLOW
                         else [(w, m) for w in range(10, 25) for m in MODES])
@pytest.mark.parametrize('name,data', LARGE, ids=[n for n, _ in LARGE])
def test_large_inputs_byte_exact(name, data, lgwin, mode):
    _check(data, 5, lgwin, mode)


def _testdata_files():
    if not os.path.isdir(TESTDATA):
        return []
    names = sorted(f for f in os.listdir(TESTDATA)
                   if '.compressed' not in f and f != 'bb.binast')
    return names


@pytest.mark.oracle
@pytest.mark.parametrize('lgwin,mode', [(22, 0), (16, 1), (11, 2)] if not SLOW
                         else [(w, m) for w in range(10, 25) for m in MODES])
@pytest.mark.parametrize('name', _testdata_files())
def test_testdata_byte_exact(name, lgwin, mode):
    with open(os.path.join(TESTDATA, name), 'rb') as f:
        data = f.read()
    _check(data, 5, lgwin, mode)


def _megabyte_text():
    names = ['lcet10.txt', 'plrabn12.txt', 'alice29.txt', 'asyoulik.txt']
    chunks = []
    for n in names:
        p = os.path.join(TESTDATA, n)
        if os.path.exists(p):
            with open(p, 'rb') as f:
                chunks.append(f.read())
    data = b''.join(chunks)
    if len(data) < (1 << 20):
        data = (data + _text(random.Random(7), 1 << 20))[:(1 << 20) + 12345]
    return data


@pytest.mark.oracle
@pytest.mark.slow
@pytest.mark.parametrize('lgwin', [22] if not SLOW else [16, 18, 19, 24])
def test_megabyte_input_h68(lgwin):
    # size >= 1 MiB and lgwin >= 19 selects the H68 hasher.
    _check(_megabyte_text(), 5, lgwin, 0)


@pytest.mark.oracle
@pytest.mark.slow
@pytest.mark.skipif(not SLOW, reason='set BROTLI_PURE_SLOW=1')
@pytest.mark.parametrize('lgwin', [18, 24])
def test_multi_megabyte_binary(lgwin):
    p = os.path.join(TESTDATA, 'bb.binast')
    if not os.path.exists(p):
        pytest.skip('bb.binast missing')
    with open(p, 'rb') as f:
        data = f.read()[:4000000]
    _check(data, 5, lgwin, 0)


@pytest.mark.oracle
@pytest.mark.slow
@pytest.mark.skipif(not SLOW, reason='set BROTLI_PURE_SLOW=1')
def test_hasher_bucket_counter_wraparound():
    # > 65536 insertions into one H58 bucket (uint16 counter wrap).
    data = _aaaax(random.Random(5), 1500000)
    _check(data, 5, 18, 0)


@pytest.mark.oracle
@pytest.mark.parametrize('name,data', SMALL[4:12], ids=[n for n, _ in SMALL[4:12]])
@pytest.mark.parametrize('lgwin', [12, 16, 22])
def test_quality6_byte_exact(name, data, lgwin):
    _check(data, 6, lgwin, 0)


@pytest.mark.oracle
def test_streaming_compressor_matches_reference():
    from brotli_pure.enc.encode import Compressor
    data = _text(random.Random(3), 150000)
    for chunks in ([data], [data[:1000], data[1000:70000], data[70000:]]):
        ref = brotli.Compressor(quality=5, lgwin=20)
        mine = Compressor(quality=5, lgwin=20)
        a = b''.join(mine.process(c) for c in chunks)
        b = b''.join(ref.process(c) for c in chunks)
        a += mine.flush()
        b += ref.flush()
        a += mine.process(b'tail data' * 10) + mine.finish()
        b += ref.process(b'tail data' * 10) + ref.finish()
        assert a == b
        assert brotli.decompress(a) == data + b'tail data' * 10


def test_bytes_like_inputs():
    data = b'hello hello hello world' * 50
    ref = brotli_pure.compress(data, quality=5)
    assert brotli_pure.compress(bytearray(data), quality=5) == ref
    assert brotli_pure.compress(memoryview(data), quality=5) == ref


@pytest.mark.parametrize('kwargs', [
    dict(quality=12), dict(quality=255),
    dict(lgwin=9), dict(lgwin=25), dict(lgwin=0),
    dict(mode=3), dict(mode=255),
    dict(lgblock=15), dict(lgblock=25), dict(lgblock=1),
])
def test_invalid_parameters_raise_brotli_error(kwargs):
    with pytest.raises(brotli_pure.error):
        brotli_pure.compress(b'data', **dict(dict(quality=5), **kwargs))
    with pytest.raises(brotli.error):
        brotli.compress(b'data', **dict(dict(quality=5), **kwargs))


@pytest.mark.parametrize('kwargs', [dict(quality=-1), dict(lgwin=256),
                                    dict(mode=-1), dict(lgblock=1000)])
def test_out_of_byte_range_parameters(kwargs):
    with pytest.raises(OverflowError):
        brotli_pure.compress(b'data', **dict(dict(quality=5), **kwargs))
    with pytest.raises(OverflowError):
        brotli.compress(b'data', **dict(dict(quality=5), **kwargs))


@pytest.mark.parametrize('quality', [0, 1, 2, 3, 4, 7, 8, 9])
def test_unsupported_quality(quality):
    with pytest.raises(UnsupportedQualityError) as info:
        brotli_pure.compress(b'data', quality=quality)
    assert isinstance(info.value, brotli_pure.error)
    assert isinstance(info.value, NotImplementedError)


@pytest.mark.oracle
@pytest.mark.parametrize('lgblock', [16, 17, 20, 24])
def test_explicit_lgblock(lgblock):
    data = _text(random.Random(11), 300000)
    expected = brotli.compress(data, quality=5, lgwin=22, lgblock=lgblock)
    assert brotli_pure.compress(data, quality=5, lgwin=22,
                                lgblock=lgblock) == expected
