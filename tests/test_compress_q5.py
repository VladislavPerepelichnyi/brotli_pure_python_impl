"""brotli_pure.compress(quality=5): validity (oracle + own decoder) and
byte-exact equality with oracle.compress at the same parameters.

Default run: the whole small corpus at lgwin 10/16/22/24 (all modes
appear), lgwin 10..24 for two samples, lgblock variants, a few inputs that
cross the 64 KiB q5 input block, and all 256 single bytes.  ``slow`` adds
every lgwin x mode, the rest of the medium corpus and the large corpus.
"""

import pytest

from tests import corpus, support
from tests.compress_common import params, run_check, cycle_modes

Q = 5

SMALL = list(cycle_modes(sorted(corpus.small()), (10, 16, 22, 24)))
ALL_LGWIN = [("small", n, w, w % 3, 0) for n in ("english_text_12000", "mixed_15000")
             for w in range(10, 25)]
ALL_LGWIN_MODES_SLOW = [("small", n, w, m, 0)
                        for n in ("english_text_12000", "mixed_15000", "records_5000", "html_text_8000")
                        for w in range(10, 25) for m in range(3)]
LGBLOCK = ([("small", n, 22, 0, b) for n in ("english_text_12000", "mixed_15000")
            for b in (16, 17, 18, 20, 24)] + [("small", "mixed_15000", 10, 1, 16)])
MEDIUM_DEFAULT = [("medium", "english_text_65536p1000", 22, 1, 0),
                  ("medium", "boundary_match_64k", 22, 0, 0),
                  ("medium", "zeros_65537", 16, 0, 0),
                  ("medium", "random_70000", 16, 0, 0)]
MEDIUM_SLOW = [("medium", n, w, support.default_mode_for(n), 0)
               for n in sorted(corpus.medium()) for w in (16, 22, 24)
               if ("medium", n, w, support.default_mode_for(n), 0) not in MEDIUM_DEFAULT]
LARGE_SLOW = [("large", n, w, 0, 0) for n in sorted(corpus.large()) for w in (18, 24)]


@pytest.mark.parametrize("check,tier,name,lgwin,mode,lgblock",
                         params(SMALL) + params(ALL_LGWIN) + params(LGBLOCK) + params(MEDIUM_DEFAULT)
                         + params(ALL_LGWIN_MODES_SLOW, slow=True) + params(MEDIUM_SLOW, slow=True)
                         + params(LARGE_SLOW, slow=True))
def test_q5(check, tier, name, lgwin, mode, lgblock):
    run_check(check, tier, name, Q, lgwin, mode, lgblock)


@pytest.mark.oracle
def test_q5_all_single_bytes_byte_exact():
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
def test_q5_lgwin_lgblock_combinations(check, tier, name, lgwin, mode, lgblock):
    run_check(check, tier, name, Q, lgwin, mode, lgblock)


@pytest.mark.oracle
@pytest.mark.parametrize("lgwin", (10, 24))
def test_q5_short_lengths_byte_exact(lgwin):
    """Every prefix length 0..400 of a text sample (header/last-block edge cases)."""
    support.need_encoder(Q)
    o = support.need_exact_oracle()
    import brotli_pure
    text = corpus.small()["english_text_1500"]
    bad = []
    for n in range(400 + 1):
        d = text[:n]
        out = brotli_pure.compress(d, quality=Q, lgwin=lgwin)
        if out != o.compress(d, quality=Q, lgwin=lgwin) or o.decompress(out) != d:
            bad.append(n)
    assert not bad, "lengths not byte-exact: %r" % bad[:20]
