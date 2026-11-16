"""brotli_pure.compress(quality=11): validity (oracle + own decoder) and
byte-exact equality with oracle.compress at the same parameters.

Pure-Python quality 11 is slow, so the default run uses the tiny corpus
(<= 3 KiB samples) at lgwin 10 and 22, one sample at every lgwin 10..24, all
modes on dictionary-dense text, three lgblock values and all single bytes.
``slow`` adds the full small corpus, all lgwin x modes, the medium corpus
(inputs crossing the 256 KiB q11 block) and the large corpus.
"""

import pytest

from tests import corpus, support
from tests.compress_common import params, run_check, cycle_modes

Q = 11

TINY = list(cycle_modes(sorted(corpus.tiny()), (10, 22)))
ALL_LGWIN = [("small", "english_text_1500", w, 1, 0) for w in range(10, 25)]
MODES = [("small", "dict_dense_text_2000", 22, m, 0) for m in range(3)]
LGBLOCK = [("small", "english_text_1500", 22, 1, b) for b in (16, 20, 24)]

SMALL_SLOW = [c for c in cycle_modes(sorted(corpus.small()), (10, 16, 22, 24))
              if c[1] not in corpus.tiny()]
ALL_LGWIN_MODES_SLOW = [("small", n, w, m, 0) for n in ("english_text_12000", "mixed_15000")
                        for w in range(10, 25) for m in range(3)]
MEDIUM_SLOW = [("medium", n, w, support.default_mode_for(n), 0)
               for n in sorted(corpus.medium()) for w in (16, 22, 24)]
LGBLOCK_SLOW = [("medium", "mixed_200000", 24, 0, b) for b in (16, 18, 20, 24)]
LARGE_SLOW = [("large", n, 24, 0, 0) for n in sorted(corpus.large())]


@pytest.mark.parametrize("check,tier,name,lgwin,mode,lgblock",
                         params(TINY) + params(ALL_LGWIN) + params(MODES) + params(LGBLOCK)
                         + params(SMALL_SLOW, slow=True) + params(ALL_LGWIN_MODES_SLOW, slow=True)
                         + params(MEDIUM_SLOW, slow=True) + params(LGBLOCK_SLOW, slow=True)
                         + params(LARGE_SLOW, slow=True))
def test_q11(check, tier, name, lgwin, mode, lgblock):
    run_check(check, tier, name, Q, lgwin, mode, lgblock)


@pytest.mark.oracle
def test_q11_all_single_bytes_byte_exact():
    support.need_encoder(Q)
    o = support.need_exact_oracle()
    import brotli_pure
    bad = []
    for b in corpus.single_bytes():
        out = brotli_pure.compress(b, quality=Q)
        if out != o.compress(b, quality=Q) or o.decompress(out) != b:
            bad.append(b)
    assert not bad, "single-byte inputs not byte-exact: %r" % bad[:10]


COMBOS = [("small", "mixed_15000" if Q == 5 else "json_text_2500", w, 0, b)
          for w, b in ((10, 24), (10, 16), (24, 16), (16, 24), (18, 18))]


@pytest.mark.parametrize("check,tier,name,lgwin,mode,lgblock", params(COMBOS))
def test_q11_lgwin_lgblock_combinations(check, tier, name, lgwin, mode, lgblock):
    run_check(check, tier, name, Q, lgwin, mode, lgblock)


@pytest.mark.oracle
@pytest.mark.parametrize("lgwin", (10, 24))
def test_q11_short_lengths_byte_exact(lgwin):
    """Every prefix length 0..160 of a text sample (header/last-block edge cases)."""
    support.need_encoder(Q)
    o = support.need_exact_oracle()
    import brotli_pure
    text = corpus.small()["english_text_1500"]
    bad = []
    for n in range(160 + 1):
        d = text[:n]
        out = brotli_pure.compress(d, quality=Q, lgwin=lgwin)
        if out != o.compress(d, quality=Q, lgwin=lgwin) or o.decompress(out) != d:
            bad.append(n)
    assert not bad, "lengths not byte-exact: %r" % bad[:20]
