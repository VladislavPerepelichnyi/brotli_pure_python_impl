"""Shared machinery for test_compress_q5.py / test_compress_q11.py.

Each (sample, quality, lgwin, mode, lgblock) combination is compressed once
by brotli_pure (cached, including a raised exception) and then checked three
ways, reported as separate test items:

  oracle_roundtrip  oracle.decompress(ours) == data          (stream validity)
  own_roundtrip     brotli_pure.decompress(ours) == data     (needs decoder)
  byte_exact        ours == oracle.compress(data, same params)
"""

import pytest

import brotli_pure
from tests import corpus, support

CHECKS = ("oracle_roundtrip", "own_roundtrip", "byte_exact")

_ours = {}
_theirs = {}


def compress_budget(n, quality):
    if quality >= 10:
        return 120.0 + n / 300.0
    return 60.0 + n / 3000.0


def get_data(tier, name):
    if tier == "bytes":
        return bytes([int(name)])
    return getattr(corpus, tier)()[name]


def ours(tier, name, quality, lgwin, mode, lgblock):
    key = (tier, name, quality, lgwin, mode, lgblock)
    if key not in _ours:
        data = get_data(tier, name)
        try:
            with support.deadline(compress_budget(len(data), quality), "brotli_pure.compress"):
                _ours[key] = (True, brotli_pure.compress(data, mode=mode, quality=quality,
                                                         lgwin=lgwin, lgblock=lgblock))
        except support.Hang:
            raise
        except Exception as e:  # noqa: BLE001 - reported by every check of this combo
            _ours[key] = (False, e)
    ok, val = _ours[key]
    if not ok:
        raise AssertionError("brotli_pure.compress(q=%d, lgwin=%d, mode=%d, lgblock=%d) raised %s: %s"
                             % (quality, lgwin, mode, lgblock, type(val).__name__, val)) from val
    return val


def theirs(tier, name, quality, lgwin, mode, lgblock):
    key = (tier, name, quality, lgwin, mode, lgblock)
    if key not in _theirs:
        _theirs[key] = support.oracle.compress(get_data(tier, name), mode=mode, quality=quality,
                                               lgwin=lgwin, lgblock=lgblock)
    return _theirs[key]


def run_check(check, tier, name, quality, lgwin, mode, lgblock=0):
    support.need_encoder(quality)
    data = get_data(tier, name)
    if check == "own_roundtrip":
        support.need_decoder()
    else:
        support.need_oracle()
    out = ours(tier, name, quality, lgwin, mode, lgblock)
    assert type(out) is bytes, "compress returned %r" % type(out)
    label = "%s q=%d lgwin=%d mode=%d lgblock=%d" % (name, quality, lgwin, mode, lgblock)
    if check == "oracle_roundtrip":
        try:
            back = support.oracle.decompress(out)
        except support.oracle.error:
            raise AssertionError("oracle rejects our stream (%s): %s" % (label, support.describe(out)))
        support.assert_same_bytes(back, data, "oracle-decoded output (%s)" % label)
    elif check == "own_roundtrip":
        back = support.ours_decompress(out, len(data))
        support.assert_same_bytes(back, data, "own-decoded output (%s)" % label)
    elif check == "byte_exact":
        support.need_exact_oracle()
        ref = theirs(tier, name, quality, lgwin, mode, lgblock)
        if out != ref:
            raise AssertionError("not byte-identical to oracle (%s): ours %d bytes, oracle %d bytes; %s"
                                 % (label, len(out), len(ref), support.first_diff(out, ref)))
    else:
        raise ValueError(check)


def params(combos, slow=False):
    """combos: iterable of (tier, name, lgwin, mode, lgblock) -> pytest params."""
    out = []
    for tier, name, lgwin, mode, lgblock in combos:
        for check in CHECKS:
            marks = [pytest.mark.slow] if slow else []
            if check != "own_roundtrip":
                marks.append(pytest.mark.oracle)
            pid = "%s-w%d-m%d%s-%s" % (name, lgwin, mode, "-b%d" % lgblock if lgblock else "", check)
            out.append(pytest.param(check, tier, name, lgwin, mode, lgblock, id=pid, marks=marks))
    return out


def cycle_modes(names, lgwins, tier="small"):
    """(tier, name, lgwin, mode, 0) with modes cycling so all three appear."""
    for name in names:
        base = support.default_mode_for(name)
        for i, w in enumerate(lgwins):
            yield tier, name, w, (base + i) % 3, 0
