"""Deterministic test corpus for brotli_pure.

Every generator is seeded, so the corpus is byte-for-byte identical on every
run and machine (it only uses ``random.Random(seed)`` and arithmetic).

Tiers
-----
``small()``   samples <= ~20 KiB, used in the big parameter matrices
              (decode x quality x lgwin, byte-exact q5).  Fast.
``tiny()``    subset of ``small()`` (<= ~3 KiB) for quality-11 byte-exact tests
              in the default (fast) run.
``medium()``  64 KiB .. ~600 KiB: input-block boundaries (q5: 2^16, q11: 2^18),
              runs around 2^16 / 2^18, distances beyond small windows.
``large()``   > 1 MiB (only used by ``slow`` tests).
``single_bytes()``  all 256 one-byte inputs.
``testdata()`` (name, original, compressed-or-None) from the reference
              test vectors (``BROTLI_TESTDATA``) plus tests/data/.

Run ``python -m tests.corpus`` for a listing with sizes.
"""

import functools
import math
import os
import random
import struct

from tests import support

# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

ENGLISH_WORDS = (
    "the of and to in is was that for it with as his on be at by had are but from or "
    "have an they which one you were all her she there would their we him been has when "
    "who will no more if out so said what up its about than into them can only other new "
    "some could time these two may first then do any like my now over such our man me even "
    "most made after also did many before must through back years where much your way well "
    "down should because each just those people how too little state good very make world "
    "still own see men work long get here between both life being under never day same "
    "another know while last might us great old year off come since against go came right "
    "used take three alice rabbit queen hatter turtle garden little thought herself "
    "compression window distance literal context dictionary transform stream block"
).split()

# Words / fragments that are known to be dense in the RFC 7932 static
# dictionary, plus suffixes and prefixes that coincide with transform
# prefixes/suffixes (" the ", "ing ", ".com/", "=\"", ...).
DICT_WORDS = (
    "time down life left back code data show only site city open just like free work "
    "text year over home page search information about contact business online world "
    "news services products links service would there their other music video people "
    "first great email which these should system government available policy number "
    "management development university international education community national "
    "following technology different important experience something together application "
    "javascript function document window location http https www com org html body "
    "style class width height content description keywords copyright privacy terms"
).split()

TRANSFORM_AFFIXES = (
    "", " ", ", ", " of the ", " of ", "s ", ".", " and ", " in ", "\"", " to ", "\">",
    "\n", ". ", "]", " for ", " a ", " that ", "'", " with ", " from ", " by ", "(",
    ". The ", " on ", " as ", " is ", "ing ", "\n\t", ":", "ed ", "=\"", " at ", "ly ",
    ",", "='", ".com/", ". This ", " not ", "er ", "al ", "ful ", "ive ", "less ",
    "est ", "ize ", "\xa0", "ous ", " the ", "e ",
)

UTF8_SNIPPETS = (
    "Съешь же ещё этих мягких французских булок, да выпей чаю. ",
    "我能吞下玻璃而不伤身体。 ",
    "私はガラスを食べられます。それは私を傷つけません。 ",
    "나는 유리를 먹을 수 있어요. 그래도 아프지 않아요 ",
    "أنا قادر على أكل الزجاج و هذا لا يؤلمني. ",
    "Τη γλώσσα μου έδωσαν ελληνική ",
    "Příliš žluťoučký kůň úpěl ďábelské ódy. ",
    "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich. ",
    "🙂🚀✨ emoji 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ",
    "हिन्दी देवनागरी लिपि ",
)


def rng(seed):
    return random.Random(seed)


def random_bytes(n, seed):
    return rng(seed).randbytes(n)


def low_entropy(n, seed, alphabet=b"ACGT"):
    r = rng(seed)
    return bytes(r.choice(alphabet) for _ in range(n))


def skewed(n, seed):
    """Geometric-ish byte distribution: mostly small values."""
    r = rng(seed)
    out = bytearray()
    for _ in range(n):
        v = int(r.expovariate(0.15))
        out.append(min(v, 255))
    return bytes(out)


def english(n, seed):
    r = rng(seed)
    out = []
    size = 0
    sentence_start = True
    while size < n:
        w = r.choice(ENGLISH_WORDS)
        if sentence_start:
            w = w.capitalize()
            sentence_start = False
        sep = " "
        x = r.random()
        if x < 0.08:
            sep, sentence_start = ". ", True
        elif x < 0.12:
            sep = ", "
        elif x < 0.13:
            sep, sentence_start = ".\n\n", True
        out.append(w + sep)
        size += len(w) + len(sep)
    return "".join(out).encode()[:n]


def dict_dense(n, seed):
    """Static-dictionary words in all case variants with transform affixes."""
    r = rng(seed)
    out = []
    size = 0
    while size < n:
        w = r.choice(DICT_WORDS)
        c = r.random()
        if c < 0.25:
            w = w.capitalize()
        elif c < 0.4:
            w = w.upper()
        elif c < 0.5 and len(w) > 4:
            w = w[r.randrange(1, 4):]  # "omit first N" shapes
        elif c < 0.6 and len(w) > 4:
            w = w[:-r.randrange(1, 4)]  # "omit last N" shapes
        s = r.choice(TRANSFORM_AFFIXES) + w + r.choice(TRANSFORM_AFFIXES)
        out.append(s)
        size += len(s.encode())
    return "".join(out).encode()[:n]


def utf8_text(n, seed):
    r = rng(seed)
    out = bytearray()
    while len(out) < n:
        out += r.choice(UTF8_SNIPPETS).encode()
    return bytes(out[:n])


def json_like(n, seed):
    r = rng(seed)
    out = ["["]
    size = 1
    i = 0
    while size < n:
        rec = ('{"id": %d, "name": "%s", "email": "%s@example.com", "active": %s, '
               '"score": %.3f, "tags": ["%s", "%s"]},\n'
               % (i, r.choice(ENGLISH_WORDS).title(), r.choice(DICT_WORDS),
                  r.choice(("true", "false")), r.uniform(0, 1000),
                  r.choice(DICT_WORDS), r.choice(ENGLISH_WORDS)))
        out.append(rec)
        size += len(rec)
        i += 1
    return "".join(out).encode()[:n]


def html_like(n, seed):
    r = rng(seed)
    out = ['<!DOCTYPE html>\n<html lang="en">\n<head><meta charset="utf-8">'
           '<title>Test page</title></head>\n<body>\n']
    size = len(out[0])
    while size < n:
        tag = r.choice(("div", "p", "span", "a", "li", "td"))
        cls = r.choice(DICT_WORDS)
        txt = " ".join(r.choice(ENGLISH_WORDS) for _ in range(r.randrange(2, 12)))
        if tag == "a":
            s = '<a href="https://www.%s.com/%s/index.html" class="%s">%s</a>\n' % (
                r.choice(DICT_WORDS), r.choice(DICT_WORDS), cls, txt)
        else:
            s = '<%s class="%s" style="width:%dpx">%s</%s>\n' % (tag, cls, r.randrange(10, 900), txt, tag)
        out.append(s)
        size += len(s)
    return "".join(out).encode()[:n]


def le_ints(n, seed):
    r = rng(seed)
    vals = []
    v = 1000
    while len(vals) * 4 < n:
        v += r.randrange(-50, 120)
        vals.append(v & 0xFFFFFFFF)
    return struct.pack("<%dI" % len(vals), *vals)[:n]


def le_floats(n, seed):
    cnt = (n + 7) // 8
    vals = [math.sin(i * 0.01 + seed) * 1000.0 for i in range(cnt)]
    return struct.pack("<%dd" % cnt, *vals)[:n]


def pcm16(n, seed):
    r = rng(seed)
    cnt = (n + 1) // 2
    vals = [int(12000 * math.sin(i * 0.05) + r.randrange(-300, 300)) for i in range(cnt)]
    return struct.pack("<%dh" % cnt, *vals)[:n]


def table_records(n, seed):
    """Fixed-size binary records: regular strides favour NPOSTFIX > 0."""
    r = rng(seed)
    out = bytearray()
    i = 0
    while len(out) < n:
        out += struct.pack("<IHHf", i, r.randrange(4), 7, i * 0.5)
        i += 1
    return bytes(out[:n])


def long_repeat(block, gap, seed):
    """block random bytes, gap bytes of other random data, then the block again."""
    r = rng(seed)
    a = r.randbytes(block)
    return a + r.randbytes(gap) + a + r.randbytes(16)


def mixed(n, seed):
    r = rng(seed)
    parts = []
    size = 0
    gens = (english, random_bytes, dict_dense, le_ints, utf8_text, low_entropy)
    k = 0
    while size < n:
        g = gens[r.randrange(len(gens))]
        m = r.randrange(200, 3000)
        parts.append(g(m, seed * 1000 + k))
        size += m
        k += 1
    return b"".join(parts)[:n]


def pattern(unit, n):
    return (unit * (n // len(unit) + 1))[:n]


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------

SHORT_STRINGS = {
    "short_a": b"a",
    "short_ab": b"ab",
    "short_abc": b"abc",
    "short_hello": b"hello",
    "short_hello_world": b"Hello, World!\n",
    "short_00ff": b"\x00\xff",
    "short_aaaa": b"aaaa",
    "short_quickfox": b"The quick brown fox jumps over the lazy dog",
    "short_text_the": b" the ",
    "short_dict_words": b"time down life left back code data",
    "short_crlf": b"line1\r\nline2\r\n\r\n",
    "short_utf8": "naïve café 日本".encode(),
}


@functools.lru_cache(maxsize=None)
def small():
    c = {"empty": b""}
    c.update(SHORT_STRINGS)
    for n in (1, 2, 3, 4, 15, 16, 17, 100, 1000, 4097):
        c["zeros_%d" % n] = bytes(n)
    c["ff_run_300"] = b"\xff" * 300
    c["abc_pattern_2000"] = pattern(b"abc", 2000)
    c["period_257_5000"] = pattern(bytes(range(256)) + b"!", 5000)
    for n in (7, 64, 255, 256, 1000, 4096, 10000):
        c["random_%d" % n] = random_bytes(n, 1000 + n)
    c["low_entropy_acgt_5000"] = low_entropy(5000, 7)
    c["low_entropy_2sym_3000"] = low_entropy(3000, 8, b"\x00\x01")
    c["skewed_6000"] = skewed(6000, 9)
    c["english_text_1500"] = english(1500, 10)
    c["english_text_12000"] = english(12000, 11)
    c["dict_dense_text_2000"] = dict_dense(2000, 12)
    c["dict_dense_text_9000"] = dict_dense(9000, 13)
    c["utf8_multilingual_text_4000"] = utf8_text(4000, 14)
    c["json_text_6000"] = json_like(6000, 15)
    c["html_text_8000"] = html_like(8000, 16)
    c["le_int32_4000"] = le_ints(4000, 17)
    c["le_float64_4000"] = le_floats(4000, 18)
    c["pcm16_3000"] = pcm16(3000, 19)
    c["records_5000"] = table_records(5000, 20)
    # Repeats beyond the lgwin=10 window (1008 bytes) and around it.
    c["long_repeat_gap_1000"] = long_repeat(300, 1000, 21)
    c["long_repeat_gap_5000"] = long_repeat(500, 5000, 22)
    c["mixed_15000"] = mixed(15000, 23)
    c["all_bytes_ascending"] = bytes(range(256))
    c["all_bytes_x4_desc"] = bytes(range(255, -1, -1)) * 4
    return c


TINY_LIMIT = 3000


@functools.lru_cache(maxsize=None)
def tiny():
    names = ["empty", "short_a", "short_abc", "short_hello_world", "short_quickfox",
             "short_dict_words", "short_utf8", "zeros_17", "zeros_1000", "abc_pattern_2000",
             "random_255", "random_1000", "english_text_1500", "dict_dense_text_2000",
             "long_repeat_gap_1000", "all_bytes_ascending"]
    s = small()
    return {n: s[n] for n in names if len(s[n]) <= TINY_LIMIT}


@functools.lru_cache(maxsize=None)
def medium():
    c = {}
    for n in (65535, 65536, 65537, 2 * 65536 + 17):
        c["zeros_%d" % n] = bytes(n)
    for n in (262143, 262144, 262145):
        c["zeros_%d" % n] = bytes(n)
    c["abc_run_70000"] = pattern(b"abcdefgh", 70000)
    c["random_70000"] = random_bytes(70000, 31)
    # Text crossing 64 KiB (q5 input block) and 256 KiB (q11 lgblock default).
    c["english_text_65536p1000"] = english(65536 + 1000, 32)
    c["mixed_200000"] = mixed(200000, 33)
    c["english_text_300000"] = english(300000, 34)
    c["dict_dense_text_80000"] = dict_dense(80000, 35)
    # Repeats beyond lgwin 16 (65520) and lgwin 18 windows.
    c["long_repeat_gap_70000"] = long_repeat(2000, 70000, 36)
    c["long_repeat_gap_270000"] = long_repeat(2000, 270000, 37)
    # Boundary-spanning match: identical text just before and just after 2^16.
    t = english(5000, 38)
    c["boundary_match_64k"] = random_bytes(65536 - 2500, 39) + t + random_bytes(3000, 40) + t
    t2 = english(6000, 41)
    c["boundary_match_256k"] = random_bytes(262144 - 3000, 42) + t2 + english(1000, 43) + t2
    return c


@functools.lru_cache(maxsize=None)
def large():
    c = {}
    c["zeros_1048577"] = bytes(1 << 20 | 1)
    c["english_text_1500000"] = english(1_500_000, 51)
    c["mixed_2000000"] = mixed(2_000_000, 52)
    # Distance beyond lgwin=20 / 22 windows.
    c["long_repeat_gap_1100000"] = long_repeat(4000, 1_100_000, 53)
    c["long_repeat_gap_4300000"] = long_repeat(4000, 4_300_000, 54)
    return c


def single_bytes():
    return [bytes([i]) for i in range(256)]


# --------------------------------------------------------------------------
# Reference test vectors
# --------------------------------------------------------------------------

def _dirs():
    out = []
    for d in (support.LOCAL_DATA_DIR, support.TESTDATA_DIR):
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


@functools.lru_cache(maxsize=None)
def testdata_compressed():
    """List of (label, compressed_path, original_path) for every reference
    ``*.compressed*`` vector whose original is present.  Files found both in
    tests/data and BROTLI_TESTDATA are listed once."""
    seen = {}
    for d in _dirs():
        for fn in sorted(os.listdir(d)):
            if ".compressed" not in fn or fn in seen:
                continue
            orig = os.path.join(d, fn.split(".compressed")[0])
            if os.path.isfile(orig):
                seen[fn] = (fn, os.path.join(d, fn), orig)
    return sorted(seen.values())


@functools.lru_cache(maxsize=None)
def testdata_originals():
    """Uncompressed reference files (name -> path)."""
    out = {}
    for d in _dirs():
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if ".compressed" in fn or not os.path.isfile(p) or fn in out:
                continue
            out[fn] = p
    return out


def read(path):
    with open(path, "rb") as f:
        return f.read()


if __name__ == "__main__":
    for tier in (small, tiny, medium, large):
        c = tier()
        print("%s: %d samples, %d bytes" % (tier.__name__, len(c), sum(map(len, c.values()))))
        for k, v in c.items():
            print("   %-32s %9d" % (k, len(v)))
    print("testdata vectors:", len(testdata_compressed()))
