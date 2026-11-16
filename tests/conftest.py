"""pytest configuration for the brotli_pure functional test-suite.

Environment knobs (see tests/README.md):

  BROTLI_PURE_IMPL=<dir>      test the brotli_pure package found in <dir>
                              (e.g. another worktree) instead of this checkout
  BROTLI_PURE_SLOW=1          also run tests marked ``slow`` (same as --run-slow)
  BROTLI_PURE_STRICT=1        "not implemented yet" becomes a failure, not a skip
  BROTLI_TESTDATA=<dir>       reference test vectors (default:
                              ../../brotli_reference/tests/testdata)
  BROTLI_REFERENCE=<dir>      reference checkout (dictionary.bin lookup)
  BROTLI_PURE_HYPOTHESIS_PROFILE=fast|default|thorough
  BROTLI_PURE_TIMEOUT_SCALE=<float>  multiply hang-detection deadlines
"""

import importlib
import os
import sys

_IMPL = os.environ.get("BROTLI_PURE_IMPL")
if _IMPL:
    _IMPL = os.path.abspath(_IMPL)
    if "brotli_pure" in sys.modules:
        raise RuntimeError("brotli_pure imported before conftest could redirect it")
    sys.path.insert(0, _IMPL)
    try:
        importlib.import_module("brotli_pure")
    finally:
        # Only brotli_pure should come from there (its submodules resolve via
        # the package __path__); keep that tree's own ``tests`` package out.
        sys.path.remove(_IMPL)

import pytest  # noqa: E402

from tests import support  # noqa: E402

try:
    from hypothesis import HealthCheck, settings

    _common = dict(deadline=None, print_blob=True,
                   suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large,
                                          HealthCheck.filter_too_much])
    settings.register_profile("fast", max_examples=15, derandomize=True, **_common)
    settings.register_profile("default", max_examples=40, derandomize=True, **_common)
    settings.register_profile("thorough", max_examples=400, **_common)
    _profile = os.environ.get("BROTLI_PURE_HYPOTHESIS_PROFILE") or ("thorough" if support.SLOW else "default")
    settings.load_profile(_profile)
except ImportError:  # hypothesis tests skip themselves
    _profile = None


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="run tests marked slow (or set BROTLI_PURE_SLOW=1)")


def pytest_configure(config):
    # Markers are declared in pyproject.toml; repeat them so the suite also
    # works when run with a different rootdir/ini file.
    config.addinivalue_line("markers", "slow: long-running tests (large inputs / quality 11)")
    config.addinivalue_line("markers", "oracle: tests that need the C reference brotli module")
    if config.getoption("--run-slow"):
        support.SLOW = True


def pytest_report_header(config):
    o = support.ORACLE_VERSION or "NOT AVAILABLE"
    lines = [
        "brotli_pure under test: %s" % support.impl_location(),
        "oracle brotli: %s   testdata: %s%s" % (
            o, support.TESTDATA_DIR, "" if os.path.isdir(support.TESTDATA_DIR) else " (missing)"),
        "slow=%s strict=%s hypothesis-profile=%s" % (support.SLOW, support.STRICT, _profile),
    ]
    return lines


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="slow test: set BROTLI_PURE_SLOW=1 or pass --run-slow")
    skip_oracle = pytest.mark.skip(reason="oracle module 'brotli' not importable")
    for item in items:
        if "slow" in item.keywords and not support.SLOW:
            item.add_marker(skip_slow)
        if "oracle" in item.keywords and support.oracle is None:
            item.add_marker(skip_oracle)
