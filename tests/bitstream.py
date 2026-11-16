"""Hand-crafted Brotli streams for decoder tests (RFC 7932).

This is a deliberately tiny, test-only *encoder* plus an independent
reference *model* of what a conforming decoder must output.  It is written
from the RFC / reference sources, not from brotli_pure, so that the
expectations of the crafted-stream tests are independent of the code under
test.  ``tests/test_harness.py`` cross-checks the model against the oracle
(C brotli) so the expectations themselves are trusted.

Usage::

    b = StreamBuilder(wbits=16)
    b.uncompressed(b"hello ")
    b.compressed([Cmd(b"wor", copy=4, dist=6), Cmd(b"!")], last=True)
    stream, expected = b.finish()    # expected is None if the model says invalid

Supported features: any WBITS encoding (incl. the reserved large-window
marker), uncompressed / metadata / compressed metablocks, simple prefix codes
(1..4 symbols, both tree shapes), complex prefix codes (HSKIP 0/2/3, optional
run-length codes 16/17, explicit length overrides for long codes), NPOSTFIX
and NDIRECT, all 16 short distance codes, implicit distance, explicit
distances and static-dictionary references with any transform.  Block
switching and context maps are *not* generated here (the oracle-compressed
corpus covers them).
"""

import hashlib
import heapq
import os

from tests import support
from tests.rfc_transforms import CONTEXT_LUT, TRANSFORMS

# --------------------------------------------------------------------------
# Format constants (RFC 7932)
# --------------------------------------------------------------------------

INS_BASE = [0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090,
            2114, 6210, 22594]
INS_EXTRA = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24]
COPY_BASE = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 22, 30, 38, 54, 70, 102, 134, 198, 326,
             582, 1094, 2118]
COPY_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24]

CODE_LENGTH_ORDER = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# Static prefix code for code-length-code lengths: value -> (bits, nbits),
# bits already in stream (LSB-first) order.
CLCL_CODE = {0: (0, 2), 1: (7, 4), 2: (3, 3), 3: (2, 2), 4: (1, 2), 5: (15, 4)}

NDBITS = [0, 0, 0, 0, 10, 10, 11, 11, 10, 10, 10, 10, 10, 9, 9, 8, 7, 7, 8, 7, 7, 6, 6, 5, 5]
DICT_OFFSETS = {}
_o = 0
for _l in range(4, 25):
    DICT_OFFSETS[_l] = _o
    _o += _l << NDBITS[_l]
DICT_SIZE = _o
del _o, _l
DICTIONARY_SHA256 = "20e42eb1b511c21806d4d227d07e5dd06877d8ce7b3a817f378f313653f35c70"

NUM_TRANSFORMS = len(TRANSFORMS)
WINDOW_GAP = 16
MAX_ALLOWED_DISTANCE = 0x7FFFFFFC

assert NUM_TRANSFORMS == 121


def _load_dictionary():
    cands = [os.path.join(support.REFERENCE_DIR, "c", "common", "dictionary.bin"),
             os.path.join(support.impl_location(), "brotli_pure", "common", "dictionary.bin")]
    for p in cands:
        if os.path.isfile(p):
            with open(p, "rb") as f:
                d = f.read()
            if hashlib.sha256(d).hexdigest() == DICTIONARY_SHA256:
                return d
    return None


DICTIONARY = _load_dictionary()


def dict_word(length, idx):
    off = DICT_OFFSETS[length] + idx * length
    return DICTIONARY[off:off + length]


def num_words(length):
    return 1 << NDBITS[length]


def _to_upper(buf, i):
    """C ToUpperCase(): returns step; may touch buf[i+1]/buf[i+2]."""
    c = buf[i]
    if c < 0xC0:
        if 0x61 <= c <= 0x7A:
            buf[i] ^= 32
        return 1
    if c < 0xE0:
        buf[i + 1] ^= 32
        return 2
    buf[i + 2] ^= 5
    return 3


def transform_word(word, t):
    """RFC 7932 section 8 transform (C semantics, incl. the UTF-8 overrun)."""
    prefix, typ, suffix = TRANSFORMS[t]
    w = bytes(word)
    if typ <= 9:
        w = w[:max(0, len(w) - typ)]
    elif 12 <= typ <= 20:
        w = w[typ - 11:]
    buf = bytearray(w) + b"\0\0\0"  # scratch for writes past the word end
    n = len(w)
    if typ == 10 and n:
        _to_upper(buf, 0)
    elif typ == 11:
        i = 0
        while i < n:
            i += _to_upper(buf, i)
    return prefix + bytes(buf[:n]) + suffix


# --------------------------------------------------------------------------
# Bit writer
# --------------------------------------------------------------------------

class BitWriter:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nacc = 0

    def write(self, value, nbits):
        if nbits == 0:
            return
        if not 0 <= value < (1 << nbits):
            raise ValueError("value %d does not fit in %d bits" % (value, nbits))
        self.acc |= value << self.nacc
        self.nacc += nbits
        while self.nacc >= 8:
            self.buf.append(self.acc & 0xFF)
            self.acc >>= 8
            self.nacc -= 8

    def bits(self, bitstring):
        """Write '0'/'1' characters in stream order."""
        for ch in bitstring:
            self.write(int(ch), 1)

    def align(self, fill=0):
        """Pad to a byte boundary.  ``fill`` gives the pad bits (LSB first)."""
        if self.nacc:
            n = 8 - self.nacc
            self.write(fill & ((1 << n) - 1), n)

    def write_bytes(self, data):
        assert self.nacc == 0, "not byte aligned"
        self.buf += data

    @property
    def bit_length(self):
        return len(self.buf) * 8 + self.nacc

    def getvalue(self):
        out = bytes(self.buf)
        if self.nacc:
            out += bytes([self.acc & 0xFF])
        return out


# --------------------------------------------------------------------------
# Prefix codes
# --------------------------------------------------------------------------

def _rev(code, n):
    r = 0
    for _ in range(n):
        r = (r << 1) | (code & 1)
        code >>= 1
    return r


def canonical_codes(lengths):
    """lengths: {symbol: length>0} -> {symbol: (bits_in_stream_order, length)}."""
    items = sorted((l, s) for s, l in lengths.items() if l > 0)
    codes = {}
    code = 0
    prev = items[0][0]
    for i, (l, s) in enumerate(items):
        if i:
            code = (code + 1) << (l - prev)
        prev = l
        codes[s] = (_rev(code, l), l)
    return codes


def kraft_ok(lengths):
    return sum(1 << (15 - l) for l in lengths.values() if l) == 1 << 15


def flat_lengths(symbols):
    """Complete code over the sorted symbols with lengths k-1 / k."""
    symbols = sorted(symbols)
    n = len(symbols)
    k = (n - 1).bit_length()
    a = (1 << k) - n
    return {s: (k - 1 if i < a else k) for i, s in enumerate(symbols)}


def huffman_lengths(freqs, max_len=15):
    """Huffman code lengths (>=2 symbols); falls back to flat if too long."""
    if len(freqs) < 2:
        raise ValueError("need >= 2 symbols")
    heap = [(f, i, (s,)) for i, (s, f) in enumerate(sorted(freqs.items()))]
    heapq.heapify(heap)
    depth = {s: 0 for s in freqs}
    cnt = len(heap)
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap)
        f2, _, b = heapq.heappop(heap)
        for s in a + b:
            depth[s] += 1
        heapq.heappush(heap, (f1 + f2, cnt, a + b))
        cnt += 1
    if max(depth.values()) > max_len:
        return flat_lengths(freqs)
    return depth


def staircase_lengths(symbols, max_len=15):
    """Lengths 1,2,3,...,m then a flat tail: produces codes up to ``max_len``
    bits (exercises second-level tables of table-driven decoders)."""
    symbols = list(symbols)
    n = len(symbols)
    if n < 2:
        raise ValueError("need >= 2 symbols")
    for m in range(min(n - 2, max_len - 1), -1, -1):
        r = n - m
        k = (r - 1).bit_length()
        if m + k <= max_len:
            break
    else:
        raise ValueError("too many symbols")
    lengths = {s: i + 1 for i, s in enumerate(symbols[:m])}
    tail = flat_lengths(range(n - m))
    for j, s in enumerate(symbols[m:]):
        lengths[s] = m + tail[j]
    assert kraft_ok(lengths)
    return lengths


def _rle_tokens(seq, use_repeats):
    """Code-length sequence -> [(clsym, extra_value, extra_bits)].  Mirrors the
    algorithm of the reference encoder (enc/entropy_encode.c)."""
    toks = []
    if not use_repeats:
        return [(v, 0, 0) for v in seq]
    prev = 8
    i = 0
    n = len(seq)
    while i < n:
        v = seq[i]
        r = 1
        while i + r < n and seq[i + r] == v:
            r += 1
        i += r
        if v == 0:
            if r == 11:
                toks.append((0, 0, 0))
                r -= 1
            if r < 3:
                toks += [(0, 0, 0)] * r
            else:
                r -= 3
                part = []
                while True:
                    part.append((17, r & 7, 3))
                    r >>= 3
                    if r == 0:
                        break
                    r -= 1
                toks += reversed(part)
        else:
            if prev != v:
                toks.append((v, 0, 0))
                r -= 1
            if r == 7:
                toks.append((v, 0, 0))
                r -= 1
            if r < 3:
                toks += [(v, 0, 0)] * r
            else:
                r -= 3
                part = []
                while True:
                    part.append((16, r & 3, 2))
                    r >>= 2
                    if r == 0:
                        break
                    r -= 1
                toks += reversed(part)
            prev = v
    return toks


class PrefixCode:
    """A prefix code plus the instructions to serialise it."""

    def __init__(self, alphabet_size, symbols_by_freq, *, force_complex=False, lengths=None,
                 hskip=0, use_repeats=False, tree_select=None, simple_order=None):
        self.alphabet_size = alphabet_size
        self.max_bits = (alphabet_size - 1).bit_length()
        syms = [s for s, _ in sorted(symbols_by_freq.items(), key=lambda kv: (-kv[1], kv[0]))]
        if not syms:
            syms = [0]
            symbols_by_freq = {0: 1}
        self.hskip = hskip
        self.use_repeats = use_repeats
        if lengths is None and (len(syms) <= 4 and not force_complex):
            self.simple = True
            order = list(simple_order) if simple_order else syms
            n = len(order)
            ts = 0 if tree_select is None else tree_select
            if n == 1:
                self.lengths = {order[0]: 0}
                self.codes = {order[0]: (0, 0)}
            else:
                if n == 2:
                    ln = [1, 1]
                elif n == 3:
                    ln = [1, 2, 2]
                elif ts:
                    ln = [1, 2, 3, 3]
                else:
                    ln = [2, 2, 2, 2]
                self.lengths = dict(zip(order, ln))
                self.codes = canonical_codes(self.lengths)
            self.order = order
            self.tree_select = ts
        else:
            self.simple = False
            if lengths is None:
                if len(syms) == 1:  # complex codes need >= 2 symbols: add a dummy
                    symbols_by_freq = dict(symbols_by_freq)
                    symbols_by_freq[(syms[0] + 1) % alphabet_size] = 1
                lengths = huffman_lengths(symbols_by_freq)
            assert kraft_ok(lengths), "incomplete code lengths"
            self.lengths = dict(lengths)
            self.codes = canonical_codes(self.lengths)
            max_sym = max(s for s, l in self.lengths.items() if l)
            seq = [self.lengths.get(s, 0) for s in range(max_sym + 1)]
            self.toks = _rle_tokens(seq, self.use_repeats)
            freq = {}
            for t in self.toks:
                freq[t[0]] = freq.get(t[0], 0) + 1
            if len(freq) == 1:
                self.cl_lengths = {next(iter(freq)): 1}  # single symbol -> 0-bit code
                self.cl_codes = {next(iter(freq)): (0, 0)}
            else:
                self.cl_lengths = huffman_lengths(freq, max_len=5)
                self.cl_codes = canonical_codes(self.cl_lengths)
            for s in CODE_LENGTH_ORDER[:self.hskip]:
                if self.cl_lengths.get(s):
                    raise ValueError("HSKIP=%d skips used code length symbol %d" % (self.hskip, s))

    def write(self, w):
        if self.simple:
            w.write(1, 2)
            w.write(len(self.order) - 1, 2)
            for s in self.order:
                w.write(s, self.max_bits)
            if len(self.order) == 4:
                w.write(self.tree_select, 1)
            return
        cl_lengths, cl_codes = self.cl_lengths, self.cl_codes
        w.write(self.hskip, 2)
        space = 32
        for s in CODE_LENGTH_ORDER[self.hskip:]:
            l = cl_lengths.get(s, 0)
            w.write(*CLCL_CODE[l])
            if l:
                space -= 32 >> l
                if space <= 0:
                    break
        for sym, extra, nextra in self.toks:
            w.write(*cl_codes[sym])
            w.write(extra, nextra)

    def emit(self, w, sym):
        bits, n = self.codes[sym]
        w.write(bits, n)


# --------------------------------------------------------------------------
# Lengths / distances
# --------------------------------------------------------------------------

def _code_for(n, base, extra):
    for c in range(23, -1, -1):
        if base[c] <= n:
            if n - base[c] >= (1 << extra[c]):
                raise ValueError("length %d not encodable" % n)
            return c, n - base[c], extra[c]
    raise ValueError("length %d not encodable" % n)


def insert_code(n):
    return _code_for(n, INS_BASE, INS_EXTRA)


def copy_code(n):
    return _code_for(n, COPY_BASE, COPY_EXTRA)


_CELLS = {(0, 0): 128, (0, 1): 192, (0, 2): 384, (1, 0): 256, (1, 1): 320, (1, 2): 512,
          (2, 0): 448, (2, 1): 576, (2, 2): 640}


def command_symbol(ic, cc, implicit):
    if implicit:
        if ic >= 8 or cc >= 16:
            raise ValueError("implicit distance needs insert code < 8 and copy code < 16")
        return (0 if cc < 8 else 64) + ((ic & 7) << 3) + (cc & 7)
    return _CELLS[(ic >> 3, cc >> 3)] + ((ic & 7) << 3) + (cc & 7)


def distance_alphabet_size(npostfix, ndirect):
    return 16 + ndirect + (48 << npostfix)


def encode_distance(d, npostfix=0, ndirect=0):
    """distance -> (symbol, extra_value, extra_bits) (RFC 7932 section 4)."""
    if d < 1:
        raise ValueError("distance must be >= 1")
    if d <= ndirect:
        return 15 + d, 0, 0
    x = d - ndirect - 1
    lcode = x & ((1 << npostfix) - 1)
    y = x >> npostfix
    for hcode in range(48):
        nbits = 1 + (hcode >> 1)
        offset = ((2 + (hcode & 1)) << nbits) - 4
        if offset <= y < offset + (1 << nbits):
            return 16 + ndirect + (hcode << npostfix) + lcode, y - offset, nbits
    raise ValueError("distance %d not encodable" % d)


def decode_distance_symbol(sym, extra, npostfix, ndirect):
    """Inverse of encode_distance for symbols >= 16 (used in self-tests)."""
    if sym < 16 + ndirect:
        return sym - 15
    x = sym - ndirect - 16
    nbits = 1 + (x >> (npostfix + 1))
    hcode = x >> npostfix
    lcode = x & ((1 << npostfix) - 1)
    offset = ((2 + (hcode & 1)) << nbits) - 4
    return ((offset + extra) << npostfix) + lcode + ndirect + 1


def wbits_header(wbits):
    """(value, nbits) for the stream header; ``wbits='large'`` gives the
    large-window marker (invalid for a standard decoder)."""
    if wbits == "large":
        return 0x11, 7
    if wbits == 16:
        return 0, 1
    if wbits == 17:
        return 1, 7
    if 18 <= wbits <= 24:
        return 1 | ((wbits - 17) << 1), 4
    if 10 <= wbits <= 15:
        return 1 | ((wbits - 8) << 4), 7
    raise ValueError(wbits)


# --------------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------------

class ModelError(Exception):
    """The model predicts that a conforming decoder must reject the stream."""


class Model:
    def __init__(self, wbits):
        self.wbits = wbits if isinstance(wbits, int) else 24
        self.out = bytearray()
        self.rb = [16, 15, 11, 4]  # rb[-1] is the last distance

    @property
    def max_distance(self):
        return min(len(self.out), (1 << self.wbits) - WINDOW_GAP)

    def short_code_distance(self, code):
        last, second = self.rb[-1], self.rb[-2]
        if code < 4:
            return self.rb[-1 - code]
        base, k = (last, code - 4) if code < 10 else (second, code - 10)
        d = base + (-1, 1, -2, 2, -3, 3)[k]
        return d if d > 0 else 0x7FFFFFFF

    def copy(self, length, d, push):
        """Back-reference or dictionary word; returns bytes produced."""
        if d > self.max_distance:
            if d > MAX_ALLOWED_DISTANCE:
                raise ModelError("distance too large")
            if not 4 <= length <= 24:
                raise ModelError("distance %d beyond window with copy length %d" % (d, length))
            if DICTIONARY is None:
                raise RuntimeError("static dictionary not available to the test model")
            address = d - self.max_distance - 1
            idx = address & ((1 << NDBITS[length]) - 1)
            t = address >> NDBITS[length]
            if t >= NUM_TRANSFORMS:
                raise ModelError("transform index %d out of range" % t)
            word = transform_word(dict_word(length, idx), t)
            self.out += word
            return len(word)
        if push:
            self.rb = self.rb[1:] + [d]
        start = len(self.out) - d
        for i in range(length):
            self.out.append(self.out[start + i])
        return length


# --------------------------------------------------------------------------
# Stream builder
# --------------------------------------------------------------------------

class Cmd:
    """One insert-and-copy command.

    Exactly one distance form when copy > 0:
      dist=D        explicit distance (codes >= 16, or direct codes)
      code=k        explicit short distance code 0..15
      implicit=True insert&copy symbol < 128 (reuses last distance)
      dict_ref=(word_index, transform)  static dictionary, copy = word length
    copy=0 means "insert only" (only valid as the last command of a metablock).
    """

    def __init__(self, insert=b"", copy=0, dist=None, code=None, implicit=False, dict_ref=None):
        self.insert = bytes(insert)
        self.copy = copy
        self.dist = dist
        self.code = code
        self.implicit = implicit
        self.dict_ref = dict_ref
        forms = sum(x is not None and x is not False for x in (dist, code, dict_ref)) + bool(implicit)
        if copy and forms != 1:
            raise ValueError("copy needs exactly one distance form")

    def __repr__(self):
        return "Cmd(%r, copy=%d, dist=%r, code=%r, implicit=%r, dict_ref=%r)" % (
            self.insert[:16], self.copy, self.dist, self.code, self.implicit, self.dict_ref)


class StreamBuilder:
    def __init__(self, wbits=16):
        self.w = BitWriter()
        self.model = Model(wbits)
        self.invalid = None  # reason, once the model predicts a decode error
        self.closed = False
        v, n = wbits_header(wbits)
        self.w.write(v, n)
        if wbits == "large":
            self.invalid = "large-window marker"

    # -- metablock headers ---------------------------------------------------
    def _mlen(self, mlen, nibbles):
        m1 = mlen - 1
        need = max(4, (max(m1, 1).bit_length() + 3) // 4)
        nib = need if nibbles is None else nibbles
        self.w.write(nib - 4, 2)
        self.w.write(m1, 4 * nib)

    def uncompressed(self, data, nibbles=None, pad=0):
        assert data, "uncompressed metablock needs MLEN >= 1"
        self.w.write(0, 1)  # ISLAST
        self._mlen(len(data), nibbles)
        self.w.write(1, 1)  # ISUNCOMPRESSED
        self.w.align(pad)
        self.w.write_bytes(data)
        self.model.out += data
        if pad:
            self._fail("non-zero padding after uncompressed header")
        return self

    def metadata(self, payload, last=False, skip_bytes=None, reserved=0, pad=0):
        self.w.write(1 if last else 0, 1)
        if last:
            self.w.write(0, 1)  # ISLASTEMPTY
        self.w.write(3, 2)  # MNIBBLES == 0
        self.w.write(reserved, 1)
        n = len(payload)
        if skip_bytes is None:
            skip_bytes = 0 if n == 0 else max(1, ((n - 1).bit_length() + 7) // 8)
        self.w.write(skip_bytes, 2)
        if skip_bytes:
            self.w.write(n - 1, 8 * skip_bytes)
        self.w.align(pad)
        self.w.write_bytes(payload)
        if reserved:
            self._fail("reserved metadata bit set")
        if pad:
            self._fail("non-zero padding after metadata header")
        if last:
            self.closed = True
        return self

    def _fail(self, why):
        if self.invalid is None:
            self.invalid = why

    # -- compressed metablock ------------------------------------------------
    def compressed(self, cmds, last=False, npostfix=0, ndirect=0, context_mode=0,
                   force_complex=(), hskip=0, use_repeats=False, tree_select=None,
                   lengths=None, nibbles=None, mlen_delta=0, blocks=None,
                   context_modes=None, lit_cmap=None, dist_cmap=None, cmap_rle=None,
                   cmap_imtf=None, rng=None):
        """Append a compressed metablock.

        force_complex: names among {'lit', 'cmd', 'dist'} that must use a
        complex prefix code; lengths: {'lit'|'cmd'|'dist': {sym: len}}
        overrides (tree 0 only); mlen_delta: corrupt MLEN on purpose.

        Block switching / context modelling (all optional):
          blocks: {'lit'|'cmd'|'dist': [(type, count), ...] or
                   callable(nsymbols) -> such a list}; first type must be 0.
          context_modes: literal context mode per literal block type.
          lit_cmap / dist_cmap: (values, ntrees) or callable(size) -> that;
                   sizes are NBLTYPESL*64 and NBLTYPESD*4.
          cmap_rle: {'lit'|'dist': RLEMAX 0..16}; cmap_imtf: {'lit'|'dist': bool}
          rng: random.Random used to pick among equivalent block type codes.
        """
        if ndirect % (1 << npostfix) or ndirect > (15 << npostfix):
            raise ValueError("bad NDIRECT for NPOSTFIX")
        snap = (bytearray(self.model.out), list(self.model.rb), self.invalid)
        try:
            return self._compressed(cmds, last, npostfix, ndirect, context_mode, force_complex,
                                    hskip, use_repeats, tree_select, lengths, nibbles, mlen_delta,
                                    blocks or {}, context_modes, lit_cmap, dist_cmap,
                                    cmap_rle or {}, cmap_imtf or {}, rng)
        except ValueError:
            self.model.out, self.model.rb, self.invalid = snap
            raise

    def _compressed(self, cmds, last, npostfix, ndirect, context_mode, force_complex,
                    hskip, use_repeats, tree_select, lengths, nibbles, mlen_delta,
                    blocks, context_modes, lit_cmap, dist_cmap, cmap_rle, cmap_imtf, rng):
        start = len(self.model.out)
        enc = []
        produced = []
        for i, c in enumerate(cmds):
            ic, iextra, ibits = insert_code(len(c.insert))
            lit_pos = len(self.model.out)
            self.model.out += c.insert
            if not c.copy:
                if i != len(cmds) - 1:
                    raise ValueError("insert-only command must be last")
                sym = command_symbol(ic, 0, ic < 8)
                enc.append((sym, (iextra, ibits), (0, 0), c.insert, None, lit_pos, 0))
                produced.append(0)
                continue
            cc, cextra, cbits = copy_code(c.copy)
            if c.implicit:
                sym = command_symbol(ic, cc, True)
                dsym = None
                d, push = self.model.rb[-1], False
            else:
                sym = command_symbol(ic, cc, False)
                if c.code is not None:
                    d = self.model.short_code_distance(c.code)
                    dsym = (c.code, 0, 0)
                    push = c.code != 0
                else:
                    if c.dict_ref is not None:
                        widx, t = c.dict_ref
                        d = self.model.max_distance + 1 + ((t << NDBITS[c.copy]) | widx)
                    else:
                        d = c.dist
                    dsym = encode_distance(d, npostfix, ndirect)
                    push = True
            if self.invalid is None:
                try:
                    self.model.copy(c.copy, d, push)
                except ModelError as e:
                    self._fail(str(e))
                    # keep MLEN covering the command so the decoder reaches it
                    self.model.out += bytes(c.copy)
            else:
                self.model.out += bytes(c.copy)
            enc.append((sym, (iextra, ibits), (cextra, cbits), c.insert, dsym, lit_pos, c.copy))
            produced.append(len(self.model.out) - lit_pos - len(c.insert))
        # A decoder ends the metablock as soon as MLEN bytes exist, so it never
        # reads the copy/distance of a command whose copy adds nothing at the
        # end (zero-length dictionary transforms).  Mirror that: trailing
        # zero-output commands are dropped, and a trailing command whose copy
        # is empty becomes insert-only.
        while enc and self.invalid is None:
            k = len(enc) - 1
            sym, ins, cpy, lits, dsym, lit_pos, copy_len = enc[k]
            if copy_len and produced[k] == 0:
                if lits:
                    ic = insert_code(len(lits))[0]
                    enc[k] = (command_symbol(ic, 0, ic < 8), ins, (0, 0), lits, None, lit_pos, 0)
                    break
                enc.pop()
                produced.pop()
                continue
            break
        if not enc:
            raise ValueError("metablock produces no output")
        mlen = len(self.model.out) - start
        mlen += mlen_delta
        if mlen_delta:
            self._fail("MLEN does not match the commands")
        if mlen < 1:
            raise ValueError("compressed metablock must produce >= 1 byte")

        # ---- block structure ---------------------------------------------
        nsyms = {"lit": sum(len(e[3]) for e in enc), "cmd": len(enc),
                 "dist": sum(1 for e in enc if e[4] is not None)}
        blist, ntypes = {}, {}
        for cat in CATS:
            spec = blocks.get(cat)
            if callable(spec):
                spec = spec(nsyms[cat])
            spec = list(spec) if spec else [(0, 1 << 24)]
            if spec[0][0] != 0:
                raise ValueError("first block type must be 0")
            ntypes[cat] = max(t for t, _ in spec) + 1
            # drop blocks that are never reached
            used, total = [], 0
            for t, n in spec:
                if n < 1:
                    raise ValueError("block count must be >= 1")
                if used and total >= nsyms[cat]:
                    break
                used.append((t, n))
                total += n
            if total < nsyms[cat]:
                raise ValueError("blocks for %s cover %d of %d symbols" % (cat, total, nsyms[cat]))
            blist[cat] = used
        nL, nD = ntypes["lit"], ntypes["dist"]
        modes = list(context_modes) if context_modes is not None else [context_mode] * nL
        if len(modes) < nL:
            modes += [context_mode] * (nL - len(modes))
        cmaps = {}
        for cat, spec, size in (("lit", lit_cmap, nL * 64), ("dist", dist_cmap, nD * 4)):
            if callable(spec):
                spec = spec(size)
            if spec is None:
                spec = ([0] * size, 1)
            values, ntrees = list(spec[0]), spec[1]
            if len(values) != size or (values and max(values) >= ntrees) or not 1 <= ntrees <= 256:
                raise ValueError("bad %s context map" % cat)
            cmaps[cat] = (values, ntrees)

        # ---- pass 1: event list ----------------------------------------
        out = self.model.out
        events = []
        state = {cat: _BlockState(blist[cat], ntypes[cat], rng) for cat in CATS}

        def symbol(cat):
            st = state[cat]
            if st.n >= 2 and st.remaining == 0:
                tcode, (csym, cextra, cbits) = st.switch()
                events.append(("btype", cat, tcode))
                events.append(("bcount", cat, csym))
                events.append(("raw", cextra, cbits))
            st.remaining -= 1
            return st.type

        lit_values, lit_ntrees = cmaps["lit"]
        dist_values, dist_ntrees = cmaps["dist"]
        for sym, (iv, ib), (cv, cb), lits, dsym, lit_pos, copy_len in enc:
            t = symbol("cmd")
            events.append(("cmd", t, sym))
            events.append(("raw", iv, ib))
            events.append(("raw", cv, cb))
            for k, byte in enumerate(lits):
                t = symbol("lit")
                pos = lit_pos + k
                p1 = out[pos - 1] if pos >= 1 else 0
                p2 = out[pos - 2] if pos >= 2 else 0
                lut = CONTEXT_LUT[modes[t]]
                ctx = lut[0][p1] | lut[1][p2]
                events.append(("lit", lit_values[t * 64 + ctx], byte))
            if dsym is not None:
                t = symbol("dist")
                dctx = min(copy_len, 5) - 2
                events.append(("dist", dist_values[t * 4 + dctx], dsym[0]))
                events.append(("raw", dsym[1], dsym[2]))

        # ---- prefix codes ------------------------------------------------
        freqs = {}
        for ev in events:
            if ev[0] != "raw":
                key = (ev[0], ev[1])
                f = freqs.setdefault(key, {})
                f[ev[2]] = f.get(ev[2], 0) + 1
        lengths = lengths or {}
        common = dict(hskip=hskip, use_repeats=use_repeats, tree_select=tree_select)
        groups = {"lit": (256, lit_ntrees), "cmd": (704, ntypes["cmd"]),
                  "dist": (distance_alphabet_size(npostfix, ndirect), dist_ntrees)}
        codes = {}
        for g, (asize, ntrees) in groups.items():
            for k in range(ntrees):
                codes[(g, k)] = PrefixCode(asize, freqs.get((g, k), {}),
                                           force_complex=g in force_complex,
                                           lengths=lengths.get(g) if k == 0 else None, **common)
        for cat in CATS:
            if ntypes[cat] >= 2:
                codes[("btype", cat)] = PrefixCode(ntypes[cat] + 2, freqs.get(("btype", cat), {}))
                f = dict(freqs.get(("bcount", cat), {}))
                first = _block_count_symbol(blist[cat][0][1])
                f[first[0]] = f.get(first[0], 0) + 1
                codes[("bcount", cat)] = PrefixCode(26, f)
        cmap_writers = {cat: _ContextMapWriter(cmaps[cat][0], cmaps[cat][1], cmap_rle.get(cat, 0),
                                               cmap_imtf.get(cat, False))
                        for cat in ("lit", "dist")}

        # ---- pass 2: write ---------------------------------------------
        w = self.w
        w.write(1 if last else 0, 1)
        if last:
            w.write(0, 1)  # ISLASTEMPTY
        self._mlen(mlen, nibbles)
        if not last:
            w.write(0, 1)  # ISUNCOMPRESSED
        for cat in CATS:
            write_varlen_uint8(w, ntypes[cat] - 1)
            if ntypes[cat] >= 2:
                codes[("btype", cat)].write(w)
                codes[("bcount", cat)].write(w)
                csym, cextra, cbits = _block_count_symbol(blist[cat][0][1])
                codes[("bcount", cat)].emit(w, csym)
                w.write(cextra, cbits)
        w.write(npostfix, 2)
        w.write(ndirect >> npostfix, 4)
        for m in modes[:nL]:
            w.write(m, 2)
        cmap_writers["lit"].write(w)
        cmap_writers["dist"].write(w)
        for g, (asize, ntrees) in groups.items():
            for k in range(ntrees):
                codes[(g, k)].write(w)
        for ev in events:
            if ev[0] == "raw":
                w.write(ev[1], ev[2])
            else:
                codes[(ev[0], ev[1])].emit(w, ev[2])
        if last:
            self.closed = True
        return self

    def finish(self, pad=0, trailing=b""):
        """Close the stream (adds an empty last metablock unless one was
        written) -> (stream_bytes, expected_output_or_None)."""
        if not self.closed:
            self.w.write(1, 1)  # ISLAST
            self.w.write(1, 1)  # ISLASTEMPTY
            self.closed = True
        if pad and self.w.nacc:
            self._fail("non-zero padding at end of stream")
        self.w.align(pad)
        data = self.w.getvalue() + trailing
        if trailing:
            self._fail("trailing bytes after the last metablock")
        return data, (None if self.invalid else bytes(self.model.out))


CATS = ("lit", "cmd", "dist")
BLOCK_LEN_OFFSET = [1, 5, 9, 13, 17, 25, 33, 41, 49, 65, 81, 97, 113, 145, 177, 209, 241, 305,
                    369, 497, 753, 1265, 2289, 4337, 8433, 16625]
BLOCK_LEN_NBITS = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13,
                   24]


def _block_count_symbol(n):
    for k in range(25, -1, -1):
        if BLOCK_LEN_OFFSET[k] <= n:
            if n - BLOCK_LEN_OFFSET[k] >= (1 << BLOCK_LEN_NBITS[k]):
                raise ValueError("block count %d too large" % n)
            return k, n - BLOCK_LEN_OFFSET[k], BLOCK_LEN_NBITS[k]
    raise ValueError("block count must be >= 1")


def write_varlen_uint8(w, v):
    """RFC 7932 VarLenUint8 (0..255)."""
    if v == 0:
        w.write(0, 1)
        return
    n = v.bit_length() - 1
    w.write(1, 1)
    w.write(n, 3)
    w.write(v - (1 << n), n)


class _BlockState:
    def __init__(self, blocks, n, rng):
        self.blocks = blocks
        self.n = n
        self.idx = 0
        self.type = 0
        self.remaining = blocks[0][1]
        self.rb = [1, 0]  # [second-to-last, last] block types, as in the decoder
        self.rng = rng

    def switch(self):
        self.idx += 1
        t, count = self.blocks[self.idx]
        options = [t + 2]
        if t == self.rb[0]:
            options.append(0)
        if t == (self.rb[1] + 1) % self.n:
            options.append(1)
        code = self.rng.choice(options) if self.rng else min(options)
        self.rb = [self.rb[1], t]
        self.type = t
        self.remaining = count
        return code, _block_count_symbol(count)


def mtf_encode(values):
    lst = list(range(256))
    out = []
    for v in values:
        i = lst.index(v)
        out.append(i)
        lst.pop(i)
        lst.insert(0, v)
    return out


class _ContextMapWriter:
    """Context map encoding (RFC 7932 section 7.3): RLE of zeros + IMTF."""

    def __init__(self, values, ntrees, rlemax, imtf):
        self.ntrees, self.rlemax, self.imtf = ntrees, rlemax, imtf
        if ntrees < 2:
            return
        vals = mtf_encode(values) if imtf else list(values)
        toks = []
        i = 0
        while i < len(vals):
            if vals[i] == 0 and rlemax > 0:
                r = 1
                while i + r < len(vals) and vals[i + r] == 0:
                    r += 1
                i += r
                while r > 0:
                    if r == 1:
                        toks.append((0, 0, 0))
                        r -= 1
                        continue
                    k = min(r.bit_length() - 1, rlemax)
                    take = min(r, (2 << k) - 1)
                    toks.append((k, take - (1 << k), k))
                    r -= take
            else:
                toks.append((vals[i] + rlemax if vals[i] else 0, 0, 0))
                i += 1
        self.toks = toks
        freq = {}
        for t in toks:
            freq[t[0]] = freq.get(t[0], 0) + 1
        self.code = PrefixCode(ntrees + rlemax, freq)

    def write(self, w):
        write_varlen_uint8(w, self.ntrees - 1)
        if self.ntrees < 2:
            return
        w.write(1 if self.rlemax else 0, 1)
        if self.rlemax:
            w.write(self.rlemax - 1, 4)
        self.code.write(w)
        for sym, extra, nbits in self.toks:
            self.code.emit(w, sym)
            w.write(extra, nbits)
        w.write(1 if self.imtf else 0, 1)


def random_blocks(r, max_types=4):
    """Callable block spec producing a random split with up to max_types types."""
    def make(nsym):
        n = r.choice([1, 2, 2, 3, max_types])
        if n == 1 or nsym <= 1:
            return [(0, max(nsym, 1))]
        out = [(0, r.randint(1, max(1, nsym // 2)))]
        total = out[0][1]
        while total < nsym:
            cnt = r.choice([1, 1, 2, 3, r.randint(1, max(1, nsym))])
            out.append((r.randrange(n), cnt))
            total += cnt
        # make sure the declared number of types is n (types may be unused)
        if max(t for t, _ in out) < n - 1:
            out.append((n - 1, 1))
        return out
    return make


def random_cmap(r, max_trees=4):
    def make(size):
        ntrees = r.choice([1, 2, 3, max_trees])
        if ntrees == 1:
            return [0] * size, 1
        style = r.random()
        if style < 0.3:
            vals = [r.randrange(ntrees) for _ in range(size)]
        elif style < 0.6:
            vals = [0] * size
            for _ in range(r.randint(1, 4)):
                vals[r.randrange(size)] = r.randrange(ntrees)
        else:
            vals = [(i // max(1, size // ntrees)) % ntrees for i in range(size)]
        vals[r.randrange(size)] = ntrees - 1  # ensure the top tree index occurs
        return vals, ntrees
    return make


def simple_stream(content, wbits=16):
    """``content`` as a single uncompressed metablock (or the empty stream)."""
    b = StreamBuilder(wbits)
    if content:
        b.uncompressed(content)
    return b.finish()[0]


# --------------------------------------------------------------------------
# Random (mostly valid) crafted streams, for differential fuzzing
# --------------------------------------------------------------------------

def random_stream(r, max_out=4000):
    """Build a random stream with ``random.Random`` instance ``r``.

    Returns (stream, expected_or_None, description)."""
    wbits = r.choice([10, 11, 12, 15, 16, 17, 18, 20, 22, 24])
    b = StreamBuilder(wbits)
    desc = ["wbits=%d" % wbits]
    alphabet = r.choice([b"ab", b"ACGT", bytes(range(32, 127)), bytes(range(256)), b"x"])
    nblocks = r.randint(1, 4)
    for k in range(nblocks):
        last = k == nblocks - 1 and r.random() < 0.6
        kind = r.random()
        if kind < 0.15 and not last:
            data = bytes(r.choice(alphabet) for _ in range(r.randint(1, 300)))
            b.uncompressed(data)
            desc.append("U%d" % len(data))
            continue
        if kind < 0.25 and not last:
            payload = r.randbytes(r.choice([0, 1, 5, 300]))
            b.metadata(payload)
            desc.append("M%d" % len(payload))
            continue
        npostfix = r.choice([0, 0, 1, 2, 3])
        ndirect = r.randint(0, 15) << npostfix if r.random() < 0.5 else 0
        cmds = []
        produced = len(b.model.out)
        target = produced + r.randint(1, max_out // nblocks)
        mdl = Model(wbits)
        mdl.out = bytearray(b.model.out)
        mdl.rb = list(b.model.rb)
        while len(mdl.out) < target:
            ins = bytes(r.choice(alphabet) for _ in range(r.choice([0, 0, 1, 2, 5, 17, 40])))
            mdl.out += ins
            if not mdl.out:
                ins = bytes([r.choice(alphabet)])
                mdl.out += ins
            mode = r.random()
            maxd = mdl.max_distance
            L = r.choice([2, 3, 4, 5, 8, 9, 10, 13, 17, 30, 70, 135, 400])
            if mode < 0.15 and len(ins) <= 9 and L <= 69:
                c = Cmd(ins, copy=L, implicit=True)
                d, push = mdl.rb[-1], False
            elif mode < 0.35:
                code = r.randrange(16)
                c = Cmd(ins, copy=L, code=code)
                d, push = mdl.short_code_distance(code), code != 0
            elif mode < 0.45 and DICTIONARY is not None:
                L = r.randint(4, 24)
                ref = (r.randrange(num_words(L)), r.randrange(NUM_TRANSFORMS))
                c = Cmd(ins, copy=L, dict_ref=ref)
                d, push = maxd + 1 + ((ref[1] << NDBITS[L]) | ref[0]), True
            else:
                d = r.randint(1, maxd) if maxd >= 1 else 1
                c = Cmd(ins, copy=L, dist=d)
                push = True
            try:
                mdl.copy(L, d, push)
            except ModelError:
                # keep the stream valid: fall back to an insert-only tail
                mdl.out = mdl.out  # noqa
                cmds.append(Cmd(ins or bytes([alphabet[0]])))
                if not ins:
                    mdl.out.append(alphabet[0])
                break
            cmds.append(c)
        else:
            if r.random() < 0.3:
                cmds.append(Cmd(bytes([r.choice(alphabet)])))
        if not cmds:
            cmds = [Cmd(b"z")]
        fc = tuple(x for x in ("lit", "cmd", "dist") if r.random() < 0.3)
        hskip = r.choice([0, 0, 2, 3])
        extra = {}
        if r.random() < 0.5:
            extra = dict(blocks={c: random_blocks(r) for c in CATS if r.random() < 0.7},
                         context_modes=[r.randrange(4) for _ in range(4)],
                         lit_cmap=random_cmap(r) if r.random() < 0.7 else None,
                         dist_cmap=random_cmap(r) if r.random() < 0.5 else None,
                         cmap_rle={"lit": r.choice([0, 1, 3, 16]), "dist": r.choice([0, 2])},
                         cmap_imtf={"lit": r.random() < 0.5, "dist": r.random() < 0.5},
                         rng=r)
            desc.append("B")
        try:
            b.compressed(cmds, last=last, npostfix=npostfix, ndirect=ndirect,
                         context_mode=r.randrange(4), force_complex=fc, hskip=hskip,
                         use_repeats=r.random() < 0.6, tree_select=r.randrange(2), **extra)
        except ValueError:
            b.compressed(cmds, last=last, npostfix=npostfix, ndirect=ndirect, **extra)
        desc.append("C%d%s" % (len(cmds), "L" if last else ""))
        if last:
            break
    stream, expected = b.finish()
    return stream, expected, " ".join(desc)
