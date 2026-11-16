"""Shared helpers for the brotli_pure functional test-suite.

* Locating the code under test (``BROTLI_PURE_IMPL`` may point at another
  checkout / worktree; see conftest.py).
* Oracle access (the C-backed ``brotli`` module built from google/brotli
  v1.2.1).  Every oracle-dependent test calls :func:`need_oracle`, which skips
  cleanly when the module is missing.
* "Is this part implemented yet?" probes.  A missing module or a
  ``NotImplementedError`` turns into a *skip* with an explicit reason (never a
  pass).  ``BROTLI_PURE_STRICT=1`` turns those skips into failures, which is
  what you want once all branches are merged.
* A SIGALRM based deadline so that a decoder that loops forever on a corrupt
  stream fails the test instead of hanging the run.
"""

import contextlib
import functools
import importlib
import os
import signal
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_REFERENCE = os.path.normpath(os.path.join(REPO, "..", "..", "brotli_reference"))


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() not in ("", "0", "no", "false", "off")


SLOW = _env_flag("BROTLI_PURE_SLOW")
STRICT = _env_flag("BROTLI_PURE_STRICT")
TIMEOUT_SCALE = float(os.environ.get("BROTLI_PURE_TIMEOUT_SCALE", "1") or 1)
REFERENCE_DIR = os.environ.get("BROTLI_REFERENCE", DEFAULT_REFERENCE)
TESTDATA_DIR = os.environ.get("BROTLI_TESTDATA", os.path.join(REFERENCE_DIR, "tests", "testdata"))
LOCAL_DATA_DIR = os.path.join(HERE, "data")

# Qualities the project must support (others may raise).
REQUIRED_QUALITIES = (5, 11)


# --------------------------------------------------------------------------
# Code under test
# --------------------------------------------------------------------------

import brotli_pure  # noqa: E402  (sys.path is arranged by conftest.py)
from brotli_pure.common.errors import BrotliError  # noqa: E402


def impl_location():
    return os.path.dirname(os.path.dirname(os.path.abspath(brotli_pure.__file__)))


def _not_implemented(reason):
    if STRICT:
        pytest.fail("BROTLI_PURE_STRICT: " + reason, pytrace=False)
    pytest.skip(reason)


@functools.lru_cache(maxsize=None)
def _module_status(modname):
    """Return None if importable, else a human readable reason."""
    try:
        importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if e.name and (modname == e.name or modname.startswith(e.name + ".")):
            return "%s not implemented yet (module missing)" % modname
        raise  # a genuine import bug inside the implementation -> error out
    return None


@functools.lru_cache(maxsize=None)
def _decoder_status():
    reason = _module_status("brotli_pure.dec.decode")
    if reason:
        return reason
    try:
        brotli_pure.decompress(b"\x06")  # canonical empty stream (WBITS=16)
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("brotli_pure."):
            return "decoder not implemented yet (%s missing)" % e.name
        raise
    except NotImplementedError as e:
        return "decoder raises NotImplementedError: %s" % e
    except Exception:  # noqa: BLE001 - a broken decoder is *implemented*, tests will fail
        pass
    return None


@functools.lru_cache(maxsize=None)
def _encoder_status(quality):
    reason = _module_status("brotli_pure.enc.encode")
    if reason:
        return reason
    try:
        brotli_pure.compress(b"probe", quality=quality)
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("brotli_pure."):
            return "quality %d not implemented yet (%s missing)" % (quality, e.name)
        raise
    except NotImplementedError as e:
        return "quality %d not implemented (NotImplementedError: %s)" % (quality, e)
    except BrotliError as e:
        return "quality %d not supported (brotli_pure.error: %s)" % (quality, e)
    except Exception:  # noqa: BLE001
        pass
    return None


def need_decoder():
    reason = _decoder_status()
    if reason:
        _not_implemented(reason)


def need_encoder(quality=5):
    """Skip unless ``compress(quality=quality)`` is implemented.

    For qualities outside REQUIRED_QUALITIES the skip is never escalated by
    BROTLI_PURE_STRICT (they are optional)."""
    reason = _encoder_status(quality)
    if reason:
        if quality in REQUIRED_QUALITIES:
            _not_implemented(reason)
        pytest.skip(reason)


def need_encoder_module():
    reason = _module_status("brotli_pure.enc.encode")
    if reason:
        _not_implemented(reason)


def decoder_available():
    return _decoder_status() is None


def encoder_available(quality):
    return _encoder_status(quality) is None


# --------------------------------------------------------------------------
# Oracle
# --------------------------------------------------------------------------

def _load_oracle():
    try:
        import brotli as mod  # the C binding, google/brotli v1.2.1
    except ImportError:
        return None
    # Must be the C binding (it exposes Compressor/Decompressor/error), not
    # something that merely happens to be called "brotli".
    if not all(hasattr(mod, a) for a in ("decompress", "compress", "error", "Compressor")):
        return None
    return mod


oracle = _load_oracle()
ORACLE_VERSION = getattr(oracle, "__version__", None) if oracle else None
EXPECTED_ORACLE_VERSION = "1.2.1"


def need_oracle():
    if oracle is None:
        pytest.skip("oracle module 'brotli' (C binding) not importable")
    return oracle


def need_exact_oracle():
    """Byte-exact comparisons are only meaningful against the same version."""
    o = need_oracle()
    if ORACLE_VERSION != EXPECTED_ORACLE_VERSION:
        pytest.skip("byte-exact tests need brotli %s, oracle is %s"
                    % (EXPECTED_ORACLE_VERSION, ORACLE_VERSION))
    return o


def oracle_decompress(data):
    """Return (ok, output_or_None)."""
    try:
        return True, oracle.decompress(bytes(data))
    except oracle.error:
        return False, None


# --------------------------------------------------------------------------
# Deadlines (hang detection)
# --------------------------------------------------------------------------

class Hang(BaseException):
    """Raised by :func:`deadline`.  BaseException so that an implementation's
    ``except Exception`` cannot swallow it."""


@contextlib.contextmanager
def deadline(seconds, what="call"):
    seconds = seconds * TIMEOUT_SCALE
    usable = (hasattr(signal, "setitimer") and seconds > 0
              and signal.getsignal(signal.SIGALRM) in (signal.SIG_DFL, signal.SIG_IGN, None))
    try:
        import threading
        usable = usable and threading.current_thread() is threading.main_thread()
    except Exception:  # noqa: BLE001
        pass
    if not usable:
        yield
        return

    def _on_alarm(signum, frame):
        raise Hang("%s exceeded %.1fs (possible infinite loop)" % (what, seconds))

    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def decode_budget(out_len, in_len=0):
    """Generous wall-clock budget for decoding ``out_len`` bytes."""
    return 20.0 + (out_len + 8 * in_len) / 50_000.0


def ours_decompress(data, expected_len=0):
    with deadline(decode_budget(expected_len, len(data)), "brotli_pure.decompress"):
        return brotli_pure.decompress(data)


# --------------------------------------------------------------------------
# Differential decode check
# --------------------------------------------------------------------------

def describe(data, limit=48):
    data = bytes(data)
    h = data[:limit].hex()
    return "%d bytes: %s%s" % (len(data), h, "..." if len(data) > limit else "")


def check_decode_matches_oracle(stream, max_output=None):
    """Core differential property of the decoder.

    * oracle accepts -> ours returns identical bytes (type ``bytes``)
    * oracle rejects -> ours raises ``brotli_pure.error`` (and nothing else)
    Returns the oracle verdict (ok, out).  ``max_output`` lets fuzzers skip
    streams that legitimately expand to something huge."""
    stream = bytes(stream)
    ok, expected = oracle_decompress(stream)
    if ok and max_output is not None and len(expected) > max_output:
        return ok, expected
    if ok:
        try:
            got = ours_decompress(stream, len(expected))
        except BrotliError as e:
            raise AssertionError("brotli_pure rejected a stream the oracle accepts (%s): %s"
                                 % (e, describe(stream)))
        assert type(got) is bytes, "decompress must return bytes, got %r" % type(got)
        if got != expected:
            raise AssertionError("output mismatch for %s: %s" % (describe(stream), first_diff(got, expected)))
    else:
        try:
            got = ours_decompress(stream)
        except BrotliError:
            return ok, expected
        except Hang:
            raise
        except Exception as e:  # noqa: BLE001
            raise AssertionError("invalid stream raised %s instead of brotli_pure.error: %r  stream=%s"
                                 % (type(e).__name__, e, describe(stream)))
        raise AssertionError("oracle rejects but brotli_pure returned %d bytes for %s"
                             % (len(got), describe(stream)))
    return ok, expected


def first_diff(got, expected):
    n = min(len(got), len(expected))
    for i in range(n):
        if got[i] != expected[i]:
            break
    else:
        i = n
    return ("len got=%d expected=%d, first difference at offset %d (got %r, expected %r)"
            % (len(got), len(expected), i, bytes(got[i:i + 8]), bytes(expected[i:i + 8])))


def assert_same_bytes(got, expected, what="output"):
    if got != expected:
        raise AssertionError("%s differs: %s" % (what, first_diff(got, expected)))


def assert_raises_brotli_error(fn, *args, what="call", budget=30.0, **kw):
    try:
        with deadline(budget, what):
            res = fn(*args, **kw)
    except BrotliError:
        return
    except Hang:
        raise
    except Exception as e:  # noqa: BLE001
        raise AssertionError("%s raised %s (%r) instead of brotli_pure.error" % (what, type(e).__name__, e))
    raise AssertionError("%s did not raise; returned %r" % (what, res if not isinstance(res, bytes) else describe(res)))


def default_mode_for(name):
    """Pick a mode deterministically from a corpus sample name."""
    if any(t in name for t in ("text", "english", "dict", "utf8", "json", "html", ".txt")):
        return 1
    if "font" in name:
        return 2
    return 0


def python_info():
    return "%s %s" % (sys.implementation.name, sys.version.split()[0])
