"""Catalogue of named hand-crafted streams.

Each case is ``Case(name, build, slow)`` where ``build()`` returns
``(stream, expected)``; ``expected`` is the decoded output predicted by the
independent model in bitstream.py, or ``None`` if a conforming decoder must
reject the stream.  test_harness.py checks every case against the oracle;
test_decompress_crafted.py checks brotli_pure against the expectation.
"""

import functools
import random

from tests.bitstream import (CLCL_CODE, COPY_BASE, COPY_EXTRA, DICTIONARY, INS_BASE, INS_EXTRA,
                             NUM_TRANSFORMS, TRANSFORMS, BitWriter, Cmd, StreamBuilder,
                             command_symbol, copy_code, dict_word, flat_lengths, encode_distance, insert_code,
                             num_words, random_stream, staircase_lengths)


class Case:
    def __init__(self, name, build, slow=False, needs_dictionary=False):
        self.name = name
        self._build = build
        self.slow = slow
        self.needs_dictionary = needs_dictionary

    @functools.lru_cache(maxsize=None)
    def build(self):
        return self._build()

    def __repr__(self):
        return "Case(%s)" % self.name


CASES = []


def case(name, slow=False, needs_dictionary=False):
    def deco(fn):
        CASES.append(Case(name, fn, slow, needs_dictionary))
        return fn
    return deco


def add(name, fn, slow=False, needs_dictionary=False):
    CASES.append(Case(name, fn, slow, needs_dictionary))


def _r(seed):
    return random.Random(seed)


# --------------------------------------------------------------------------
# Stream header / empty streams
# --------------------------------------------------------------------------

for _wb in list(range(10, 25)):
    add("empty_wbits%d" % _wb, (lambda wb: lambda: StreamBuilder(wb).finish())(_wb))
    add("uncompressed_hello_wbits%d" % _wb,
        (lambda wb: lambda: StreamBuilder(wb).uncompressed(b"hello").finish())(_wb))

add("large_window_marker", lambda: StreamBuilder("large").finish())
add("large_window_marker_with_data",
    lambda: StreamBuilder("large").uncompressed(b"abc").finish())


@case("canonical_empty_0x06")
def _():
    return b"\x06", b""


@case("empty_input")
def _():
    return b"", None


@case("islast_empty_nonzero_padding")
def _():
    return StreamBuilder(16).finish(pad=0x1F)


@case("trailing_zero_byte")
def _():
    return StreamBuilder(16).uncompressed(b"x").finish(trailing=b"\x00")


@case("trailing_second_stream")
def _():
    s, _ = StreamBuilder(16).uncompressed(b"x").finish()
    return StreamBuilder(16).uncompressed(b"x").finish(trailing=s)


@case("trailing_garbage_after_empty")
def _():
    return StreamBuilder(16).finish(trailing=b"garbage")


# --------------------------------------------------------------------------
# Uncompressed metablocks / MLEN encodings
# --------------------------------------------------------------------------

@case("uncompressed_sizes_mix")
def _():
    b = StreamBuilder(16)
    r = _r(1)
    for n in (1, 2, 255, 256, 1000, 65536):
        b.uncompressed(r.randbytes(n))
    return b.finish()


@case("uncompressed_65537_five_nibbles")
def _():
    return StreamBuilder(18).uncompressed(_r(2).randbytes(65537)).finish()


@case("uncompressed_1M_plus1_six_nibbles", slow=True)
def _():
    return StreamBuilder(22).uncompressed(_r(3).randbytes((1 << 20) + 1)).finish()


@case("uncompressed_padding_nonzero")
def _():
    return StreamBuilder(16).uncompressed(b"abc", pad=1).finish()


@case("mlen_exuberant_nibble_5")
def _():
    b = StreamBuilder(16).uncompressed(b"a", nibbles=5)
    b._fail("exuberant nibble")
    return b.finish()


@case("mlen_exuberant_nibble_6")
def _():
    b = StreamBuilder(18).uncompressed(_r(4).randbytes(0x10001), nibbles=6)
    b._fail("exuberant nibble")
    return b.finish()


@case("mlen_nonminimal_4_nibbles_ok")
def _():
    # MLEN-1 = 0 written with 4 nibbles is the *normal* encoding; 4 nibbles of
    # zeros are allowed (only >4 nibbles with a zero top nibble are rejected).
    return StreamBuilder(16).uncompressed(b"z", nibbles=4).finish()


@case("uncompressed_then_compressed_backref")
def _():
    b = StreamBuilder(16)
    b.uncompressed(b"0123456789abcdef" * 4)
    b.compressed([Cmd(b"XY", copy=20, dist=66), Cmd(b"", copy=5, code=0)], last=True)
    return b.finish()


# --------------------------------------------------------------------------
# Metadata metablocks
# --------------------------------------------------------------------------

@case("metadata_sizes")
def _():
    b = StreamBuilder(16)
    b.uncompressed(b"before ")
    for n in (0, 1, 255, 256, 257, 65536, 65537):
        b.metadata(_r(n).randbytes(n))
    b.uncompressed(b"after")
    return b.finish()


@case("metadata_only_stream")
def _():
    return StreamBuilder(16).metadata(b"meta").finish()


@case("metadata_is_last")
def _():
    return StreamBuilder(16).uncompressed(b"data").metadata(b"tail", last=True).finish()


@case("metadata_reserved_bit")
def _():
    return StreamBuilder(16).metadata(b"x", reserved=1).finish()


@case("metadata_exuberant_skip_bytes")
def _():
    b = StreamBuilder(16).metadata(b"x", skip_bytes=2)  # MSKIPLEN-1 = 0 in 2 bytes
    b._fail("exuberant metadata length")
    return b.finish()


@case("metadata_nonzero_padding")
def _():
    return StreamBuilder(16).metadata(b"x", pad=1).finish()


@case("metadata_between_compressed")
def _():
    b = StreamBuilder(16)
    b.compressed([Cmd(b"abcabc", copy=6, dist=3)])
    b.metadata(b"\x00" * 10)
    b.compressed([Cmd(b"", copy=9, dist=12)], last=True)
    return b.finish()


# --------------------------------------------------------------------------
# Prefix codes
# --------------------------------------------------------------------------

@case("literals_simple_1sym")
def _():
    return StreamBuilder(16).compressed([Cmd(b"aaaa")], last=True).finish()


@case("literals_simple_2sym")
def _():
    return StreamBuilder(16).compressed([Cmd(b"abba" * 3)], last=True).finish()


@case("literals_simple_3sym")
def _():
    return StreamBuilder(16).compressed([Cmd(b"abcaab")], last=True).finish()


for _ts in (0, 1):
    add("literals_simple_4sym_tree%d" % _ts,
        (lambda ts: lambda: StreamBuilder(16).compressed(
            [Cmd(b"abcdaabbaaaad")], last=True, tree_select=ts).finish())(_ts))


@case("literals_complex_all_256")
def _():
    data = bytes(range(256)) + _r(5).randbytes(500)
    return StreamBuilder(16).compressed([Cmd(data)], last=True).finish()


_HSKIP_TEXT = b"the quick brown fox jumps over the lazy dog" * 3 + bytes([200, 201, 250])

for _hs in (0, 2, 3):
    for _rep in (False, True):
        # flat lengths 4/5 over 30 symbols: code-length symbols 1..3 unused,
        # so HSKIP 2 and 3 are legal.
        add("literals_complex_hskip%d_repeat%d" % (_hs, _rep),
            (lambda hs, rep: lambda: StreamBuilder(16).compressed(
                [Cmd(_HSKIP_TEXT)], last=True, hskip=hs, use_repeats=rep,
                lengths={"lit": flat_lengths(set(_HSKIP_TEXT))}).finish())(_hs, _rep))


@case("literals_15bit_codes")
def _():
    syms = list(range(256))
    data = bytes(syms) * 2
    lengths = staircase_lengths(syms, 15)
    assert max(lengths.values()) == 15
    return StreamBuilder(16).compressed([Cmd(data)], last=True, lengths={"lit": lengths},
                                        use_repeats=True).finish()


@case("commands_and_distances_huffman")
def _():
    r = _r(6)
    cmds = []
    out = 0
    for i in range(300):
        ins = bytes(r.choice(b"xyz") for _ in range(r.randrange(0, 30)))
        out += len(ins)
        if out == 0:
            ins, out = b"x", 1
        d = r.randint(1, out)
        L = r.choice(range(2, 300))
        cmds.append(Cmd(ins, copy=L, dist=d))
        out += L
    return StreamBuilder(18).compressed(cmds, last=True, force_complex=("lit", "cmd", "dist"),
                                        use_repeats=True).finish()


@case("commands_and_distances_15bit_codes")
def _():
    r = _r(11)
    cmds = []
    out = 0
    for i in range(400):
        ins = bytes([r.randrange(256)]) * r.choice((0, 1, 2, 3, 7, 12, 30, 100))
        out += len(ins)
        if out == 0:
            ins, out = b"x", 1
        d = r.choice((1, 2, 3, out, r.randint(1, out)))
        L = r.choice((2, 3, 4, 9, 10, 15, 40, 100, 1000, r.randint(2, 3000)))
        cmds.append(Cmd(ins, copy=L, dist=d))
        out += L
    syms_cmd, syms_dist = set(), set()
    for c in cmds:
        ic = insert_code(len(c.insert))[0]
        cc = copy_code(c.copy)[0]
        syms_cmd.add(command_symbol(ic, cc, False))
        syms_dist.add(encode_distance(c.dist)[0])
    lengths = {"cmd": staircase_lengths(sorted(syms_cmd)),
               "dist": staircase_lengths(sorted(syms_dist))}
    return StreamBuilder(20).compressed(cmds, last=True, lengths=lengths,
                                        use_repeats=True).finish()


@case("single_symbol_complex_dummy")
def _():
    return StreamBuilder(16).compressed([Cmd(b"q" * 40)], last=True,
                                        force_complex=("lit", "cmd", "dist")).finish()


def _raw_header(w, mlen=1, npostfix=0, ndirect_field=0):
    """wbits=16, ISLAST, !ISLASTEMPTY, 4 nibbles, block types/trees trivial."""
    w.write(0, 1)
    w.write(1, 1)
    w.write(0, 1)
    w.write(0, 2)
    w.write(mlen - 1, 16)
    w.write(0, 3)  # NBLTYPES L/I/D = 1
    w.write(npostfix, 2)
    w.write(ndirect_field, 4)
    w.write(0, 2)  # context mode
    w.write(0, 2)  # NTREESL, NTREESD = 1


@case("simple_code_duplicate_symbols")
def _():
    w = BitWriter()
    _raw_header(w)
    w.write(1, 2)
    w.write(1, 2)  # 2 symbols
    w.write(65, 8)
    w.write(65, 8)
    w.write(0, 32)
    return w.getvalue(), None


@case("simple_code_symbol_out_of_alphabet")
def _():
    w = BitWriter()
    _raw_header(w, npostfix=0, ndirect_field=1)  # distance alphabet 65 -> 7-bit symbols
    w.write(1, 2), w.write(0, 2), w.write(65, 8)  # literal: 1 symbol 'A'
    w.write(1, 2), w.write(0, 2), w.write(0, 10)  # command: symbol 0 (insert 0, copy 2)
    w.write(1, 2), w.write(0, 2), w.write(100, 7)  # distance: 100 >= 65 -> invalid
    w.write(0, 32)
    return w.getvalue(), None


@case("simple_code_command_symbol_704_out_of_range")
def _():
    w = BitWriter()
    _raw_header(w)
    w.write(1, 2), w.write(0, 2), w.write(65, 8)
    w.write(1, 2), w.write(0, 2), w.write(1000, 10)  # >= 704
    w.write(0, 40)
    return w.getvalue(), None


@case("complex_code_incomplete")
def _():
    w = BitWriter()
    _raw_header(w)
    # literal code: complex, HSKIP=0; code-length code with symbols 1 and 2
    # (lengths 1 each -> complete), then symbol lengths: only symbol 0 with
    # length 1 -> Kraft sum 1/2 -> incomplete.  Then zero padding.
    w.write(0, 2)
    for s in [1, 2]:  # CODE_LENGTH_ORDER starts with 1, 2
        w.write(*CLCL_CODE[1])
    w.write(0, 1)  # symbol 0 -> clc symbol 1 (code '0')
    w.write(1, 1)  # symbol 1 -> clc symbol 2 (code '1')  (now space 1/2+1/4)
    w.write(0, 64)
    return w.getvalue(), None


@case("complex_code_length_code_oversubscribed")
def _():
    w = BitWriter()
    _raw_header(w)
    w.write(0, 2)
    for v in (2, 1, 1):  # 1/4 + 1/2 + 1/2 > 1: oversubscribed code length code
        w.write(*CLCL_CODE[v])
    w.write(0, 64)
    return w.getvalue(), None


# --------------------------------------------------------------------------
# Insert / copy lengths
# --------------------------------------------------------------------------

def _lengths_case(kind):
    def build():
        b = StreamBuilder(24)
        cmds = []
        r = _r(7)
        total = 0
        for code in range(24):
            if kind == "insert":
                base, extra = INS_BASE[code], INS_EXTRA[code]
                for n in sorted({base, base + (1 << extra) - 1 if extra < 15 else base + 1000}):
                    cmds.append(Cmd(bytes(r.choice(b"ab") for _ in range(n)), copy=4, dist=1))
                    total += n + 4
            else:
                base, extra = COPY_BASE[code], COPY_EXTRA[code]
                for n in sorted({base, base + (1 << extra) - 1 if extra < 15 else base + 5000}):
                    cmds.append(Cmd(b"ab", copy=n, dist=2))
                    total += n + 2
        return b.compressed(cmds, last=True, force_complex=("lit", "cmd", "dist")).finish()
    return build


add("all_insert_length_codes", _lengths_case("insert"))
add("all_copy_length_codes", _lengths_case("copy"))


@case("insert_only_long_tail_codes")
def _():
    # insert codes >= 8 have no implicit-distance variant; ends insert-only.
    return StreamBuilder(16).compressed([Cmd(b"k" * 22594)], last=True).finish()


@case("overlapping_copy_distance1")
def _():
    return StreamBuilder(16).compressed([Cmd(b"Z", copy=5000, dist=1)], last=True).finish()


@case("overlapping_copy_distance3")
def _():
    return StreamBuilder(16).compressed([Cmd(b"abc", copy=1000, dist=3), Cmd(b"!")],
                                        last=True).finish()


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------

@case("short_codes_initial_ring_buffer")
def _():
    # Before any explicit distance the ring buffer is 4, 11, 15, 16 (last first).
    b = StreamBuilder(16)
    cmds = [Cmd(bytes(range(65, 65 + 20)), copy=4, code=0),  # dist 4, no push
            Cmd(b"", copy=4, code=1),   # 11
            Cmd(b"", copy=4, code=2),   # 15
            Cmd(b"", copy=4, code=3)]   # 16
    return b.compressed(cmds, last=True).finish()


@case("short_codes_all_16")
def _():
    b = StreamBuilder(16)
    cmds = [Cmd(_r(8).randbytes(300), copy=3, dist=100), Cmd(b"x", copy=3, dist=50)]
    for code in range(16):
        cmds.append(Cmd(b"y", copy=3 + code, code=code))
    for code in range(15, -1, -1):
        cmds.append(Cmd(b"", copy=2, code=code))
    return b.compressed(cmds, last=True, force_complex=("dist",)).finish()


@case("implicit_distance_commands")
def _():
    b = StreamBuilder(16)
    cmds = [Cmd(b"abcdefgh" * 4, copy=8, dist=8), Cmd(b"x", copy=9, implicit=True),
            Cmd(b"yy", copy=17, implicit=True), Cmd(b"", copy=2, implicit=True)]
    return b.compressed(cmds, last=True).finish()


@case("short_code_negative_distance")
def _():
    # last distance 1, code 6 => 1 - 2 = -1 -> invalid (C maps to a huge distance)
    b = StreamBuilder(16)
    return b.compressed([Cmd(b"abc", copy=3, dist=1), Cmd(b"", copy=4, code=6)],
                        last=True).finish()


@case("short_code_to_zero_distance")
def _():
    b = StreamBuilder(16)
    return b.compressed([Cmd(b"abc", copy=3, dist=1), Cmd(b"", copy=5, code=4)],
                        last=True).finish()


for _np in range(4):
    for _nd_field in (0, 1, 7, 15):
        _nd = _nd_field << _np

        def _mk(np_, nd):
            def build():
                r = _r(100 + np_ * 16 + nd)
                b = StreamBuilder(20)
                b.uncompressed(r.randbytes(5000))
                dists = set(range(1, 140)) | {nd, nd + 1, nd + 2, 255, 256, 257, 1000, 4096, 4999}
                cmds = [Cmd(bytes([r.randrange(256)]), copy=r.randint(2, 9), dist=d)
                        for d in sorted(x for x in dists if x >= 1)]
                return b.compressed(cmds, last=True, npostfix=np_, ndirect=nd,
                                    force_complex=("dist",)).finish()
            return build
        add("npostfix%d_ndirect%d" % (_np, _nd), _mk(_np, _nd))


@case("distance_beyond_output_len3_invalid")
def _():
    return StreamBuilder(16).compressed([Cmd(b"abc", copy=3, dist=10)], last=True).finish()


@case("distance_beyond_output_len25_invalid")
def _():
    return StreamBuilder(16).compressed([Cmd(b"abc", copy=25, dist=10)], last=True).finish()


@case("mlen_too_small_for_copy")
def _():
    return StreamBuilder(16).compressed([Cmd(b"abc", copy=10, dist=3)], last=True,
                                        mlen_delta=-1).finish()


@case("mlen_too_small_for_insert")
def _():
    return StreamBuilder(16).compressed([Cmd(b"abcdef")], last=True, mlen_delta=-1).finish()


def _window_case(wbits, delta, length):
    """Fill more than the window with uncompressed data, then copy from
    distance (window - 16) + delta.  delta <= 0: valid back-reference;
    delta > 0: becomes a dictionary reference (valid only for len 4..24)."""
    def build():
        r = _r(wbits)
        b = StreamBuilder(wbits)
        size = (1 << wbits) + 100
        chunk = 1 << 16
        data = r.randbytes(size)
        for i in range(0, size, chunk):
            b.uncompressed(data[i:i + chunk])
        maxd = (1 << wbits) - 16
        b.compressed([Cmd(b"", copy=length, dist=maxd + delta)], last=True)
        return b.finish()
    return build


for _wb in range(10, 25):
    _slow = _wb >= 19
    add("window%d_max_distance" % _wb, _window_case(_wb, 0, 7), slow=_slow)
    add("window%d_max_distance_plus1_dictionary" % _wb, _window_case(_wb, 1, 7), slow=_slow,
        needs_dictionary=True)
    add("window%d_max_distance_plus1_len3_invalid" % _wb, _window_case(_wb, 1, 3), slow=_slow)


# --------------------------------------------------------------------------
# Static dictionary
# --------------------------------------------------------------------------

def _dict_sample_indices(L):
    n = num_words(L)
    return sorted({0, 1, n // 3, n // 2, n - 2, n - 1})


def _dict_transform_case(t, exhaustive=False):
    def build():
        b = StreamBuilder(16)
        cmds = []
        for L in range(4, 25):
            idxs = range(num_words(L)) if exhaustive else _dict_sample_indices(L)
            for i in idxs:
                cmds.append(Cmd(b"", copy=L, dict_ref=(i, t)))
        # A dictionary reference must not update the distance ring buffer:
        # the trailing code-0 copy uses the explicit distance 1 set first.
        cmds.insert(0, Cmd(b"#", copy=2, dist=1))
        cmds.append(Cmd(b"+", copy=3, code=0))
        b.compressed(cmds, last=True, force_complex=("cmd", "dist"))
        return b.finish()
    return build


for _t in range(NUM_TRANSFORMS):
    add("dict_transform%03d" % _t, _dict_transform_case(_t), needs_dictionary=True)
for _t in (0, 9, 44, 3, 12):  # identity, uppercase-first, uppercase-all, omit-first-1, omit-last-1
    add("dict_transform%03d_all_words" % _t, _dict_transform_case(_t, True), slow=True,
        needs_dictionary=True)


def _uppercase_overrun_words():
    """Words whose UTF-8 'uppercasing' steps past the word end (C writes 1-2
    bytes beyond the word; a port must not crash or leak them)."""
    out = []
    if DICTIONARY is None:
        return out
    for L in range(4, 25):
        for i in range(num_words(L)):
            w = dict_word(L, i)
            j, n = 0, L
            while n > 0:
                c = w[j]
                step = 1 if c < 0xC0 else (2 if c < 0xE0 else 3)
                if step > n:
                    out.append((L, i))
                    break
                j += step
                n -= step
    return out


@case("dict_uppercase_utf8_overrun", needs_dictionary=True)
def _():
    words = _uppercase_overrun_words()
    assert words, "expected some overrun words in the RFC dictionary"
    upper = [t for t, (_, typ, _) in enumerate(TRANSFORMS) if typ in (10, 11)]
    b = StreamBuilder(16)
    cmds = [Cmd(b"<", copy=L, dict_ref=(i, t)) for (L, i) in words for t in upper]
    b.compressed(cmds + [Cmd(b">")], last=True, force_complex=("cmd", "dist"))
    return b.finish()


@case("dict_uppercase_utf8_overrun_last_in_stream", needs_dictionary=True)
def _():
    # Overrun word with UPPERCASE_ALL and no suffix as the very last output.
    words = _uppercase_overrun_words()
    t = [t for t, (p, typ, s) in enumerate(TRANSFORMS) if typ == 11 and not s][0]
    L, i = words[0]
    return StreamBuilder(16).compressed([Cmd(b"", copy=L, dict_ref=(i, t))],
                                        last=True).finish()


@case("dict_zero_length_results", needs_dictionary=True)
def _():
    # omit-first-N / omit-last-N with N >= word length and no affixes: 0 bytes.
    zero_ts = [t for t, (p, typ, s) in enumerate(TRANSFORMS)
               if not p and not s and (1 <= typ <= 9 or 12 <= typ <= 20)]
    cmds = [Cmd(b"ab", copy=2, dist=2)]
    for t in zero_ts:
        typ = TRANSFORMS[t][1]
        n = typ if typ <= 9 else typ - 11
        for L in range(4, min(n, 24) + 1):
            cmds.append(Cmd(b"", copy=L, dict_ref=(0, t)))
    cmds.append(Cmd(b"z"))
    return StreamBuilder(16).compressed(cmds, last=True, force_complex=("cmd", "dist")).finish()


@case("dict_transform_index_121_invalid", needs_dictionary=True)
def _():
    b = StreamBuilder(16)
    return b.compressed([Cmd(b"", copy=4, dict_ref=(0, 121))], last=True).finish()


@case("dict_word_len_24_last_transform", needs_dictionary=True)
def _():
    b = StreamBuilder(16)
    return b.compressed([Cmd(b"", copy=24, dict_ref=(31, 120))], last=True).finish()


@case("dict_after_window_full", needs_dictionary=True)
def _():
    # max_distance capped at window-16: dictionary addresses start there.
    r = _r(9)
    b = StreamBuilder(10)
    b.uncompressed(r.randbytes(3000))
    cmds = [Cmd(b"", copy=L, dict_ref=(5, t)) for L in (4, 10, 24) for t in (0, 10, 44, 120)]
    return b.compressed(cmds, last=True).finish()


@case("dict_ref_between_backrefs_ring_buffer", needs_dictionary=True)
def _():
    b = StreamBuilder(16)
    cmds = [Cmd(b"0123456789", copy=4, dist=7),
            Cmd(b"", copy=8, dict_ref=(17, 5)),
            Cmd(b"", copy=4, code=0),          # last distance is still 7
            Cmd(b"", copy=6, dict_ref=(3, 77)),
            Cmd(b"q", copy=5, code=1),        # second-last (initial 4) is still there
            Cmd(b"", copy=7, implicit=True)]
    return b.compressed(cmds, last=True).finish()


# --------------------------------------------------------------------------
# Block switching and context modelling
# --------------------------------------------------------------------------

@case("blocks_256_literal_types_256_trees")
def _():
    r = _r(12)
    data = bytes(r.randrange(256) for _ in range(700))
    # every literal switches block type; 256 types; 256 literal trees
    types = [0] + [r.randrange(256) for _ in range(len(data) - 1)]
    types[1:3] = [255, 254]
    blocks = {"lit": [(t, 1) for t in types]}
    cmap = ([(i * 7) % 256 for i in range(256 * 64)], 256)
    return StreamBuilder(16).compressed([Cmd(data)], last=True, blocks=blocks,
                                        context_modes=[i % 4 for i in range(256)],
                                        lit_cmap=cmap, cmap_rle={"lit": 16},
                                        cmap_imtf={"lit": True}).finish()


@case("blocks_type_codes_prev_and_next")
def _():
    # rng=None makes the builder prefer type codes 0 (second-to-last type)
    # and 1 (last type + 1, wrapping) whenever they apply.
    data = bytes(range(97, 123)) * 4
    seq = [0, 1, 0, 1, 2, 0, 2, 2, 1, 2, 0, 1, 2]
    blocks = {"lit": [(t, 8) for t in seq]}
    return StreamBuilder(16).compressed([Cmd(data)], last=True, blocks=blocks).finish()


@case("blocks_large_counts")
def _():
    r = _r(13)
    data = bytes(r.choice(b"abcd") for _ in range(40000))
    blocks = {"lit": [(0, 16625 + 1000), (1, 3), (0, 1 << 24)],
              "cmd": [(0, 1), (1, 1 << 20)], "dist": [(0, 2), (1, 9000)]}
    cmds = [Cmd(data[:20000], copy=100, dist=17), Cmd(data[20000:], copy=50, dist=3),
            Cmd(b"x", copy=4, dist=1), Cmd(b"yz")]
    return StreamBuilder(18).compressed(cmds, last=True, blocks=blocks).finish()


for _m in range(4):
    def _mk_mode(mode):
        def build():
            r = _r(200 + mode)
            text = ("Grüße, мир! 日本語 hello WORLD 123 -45 +6 \x00\x7f\xff " * 20).encode("utf-8", "surrogatepass")
            data = text + bytes(r.randrange(256) for _ in range(500))
            cmap = ([c % 5 for c in range(64)], 5)  # tree depends on context id
            return StreamBuilder(16).compressed([Cmd(data)], last=True, context_modes=[mode],
                                                lit_cmap=cmap, cmap_rle={"lit": 2}).finish()
        return build
    add("context_mode_%d_context_dependent_trees" % _m, _mk_mode(_m))


@case("distance_context_map_by_copy_length")
def _():
    r = _r(14)
    cmds = [Cmd(r.randbytes(3), copy=L, dist=r.randint(1, 3)) for L in (2, 3, 4, 5, 9, 2, 3, 4, 40) * 20]
    dist_cmap = ([0, 1, 2, 3, 3, 2, 1, 0], 4)
    blocks = {"dist": [(0, 30), (1, 30), (0, 1000)]}
    return StreamBuilder(16).compressed(cmds, last=True, blocks=blocks, dist_cmap=dist_cmap,
                                        cmap_imtf={"dist": True}).finish()


@case("command_block_types_own_trees")
def _():
    r = _r(15)
    cmds = []
    for i in range(200):
        cmds.append(Cmd(bytes([97 + i % 26]) * (i % 7), copy=2 + i % 30, dist=1 + i % 3 if i else None,
                        code=None if i else 0))
    cmds[0] = Cmd(b"start", copy=3, dist=2)
    blocks = {"cmd": [(i % 3, 1 + (i % 5)) for i in range(120)]}
    return StreamBuilder(16).compressed(cmds, last=True, blocks=blocks, rng=r).finish()


for _seed in range(30):
    def _mk_blk(seed):
        def build():
            from tests.bitstream import random_blocks, random_cmap, CATS
            r = random.Random(30_000 + seed)
            data = bytes(r.choice(b"abcdefgh\x00\xff") for _ in range(r.randint(50, 2000)))
            cmds = []
            pos = 0
            while pos < len(data) - 1:
                n = r.randint(1, 40)
                ins = data[pos:pos + n]
                pos += len(ins)
                cmds.append(Cmd(ins, copy=r.randint(2, 30), dist=r.randint(1, pos)))
            cmds.append(Cmd(b"!"))
            return StreamBuilder(16).compressed(
                cmds, last=True, blocks={c: random_blocks(r, 8) for c in CATS},
                context_modes=[r.randrange(4) for _ in range(8)],
                lit_cmap=random_cmap(r, 8), dist_cmap=random_cmap(r, 6),
                cmap_rle={"lit": r.choice([0, 1, 4, 16]), "dist": r.choice([0, 1, 3])},
                cmap_imtf={"lit": r.random() < 0.5, "dist": r.random() < 0.5}, rng=r).finish()
        return build
    add("blocks_random_%02d" % _seed, _mk_blk(_seed))


# --------------------------------------------------------------------------
# Multi-metablock / mixed
# --------------------------------------------------------------------------

@case("many_small_metablocks")
def _():
    b = StreamBuilder(16)
    r = _r(10)
    for i in range(60):
        k = i % 3
        if k == 0:
            b.uncompressed(r.randbytes(r.randint(1, 50)))
        elif k == 1:
            b.compressed([Cmd(b"m%d" % i, copy=3, dist=2)])
        else:
            b.metadata(r.randbytes(r.randint(0, 20)))
    return b.finish()


@case("compressed_last_with_context_modes")
def _():
    b = StreamBuilder(16)
    for cm in range(4):
        b.compressed([Cmd(b"context mode %d " % cm, copy=8, dist=5)], context_mode=cm)
    b.compressed([Cmd(b"end")], last=True)
    return b.finish()


for _seed in range(60):
    add("random_crafted_%02d" % _seed,
        (lambda s: lambda: random_stream(random.Random(10_000 + s))[:2])(_seed),
        needs_dictionary=True)


def by_name():
    return {c.name: c for c in CASES}
