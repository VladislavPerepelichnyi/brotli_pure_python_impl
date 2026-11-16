#!/bin/sh
# Run the test-suite of this worktree against the *committed* state of another
# branch (exported to a temporary snapshot, so in-progress edits in that
# branch's worktree do not interfere).
#
#   tools/run_branch_tests.sh <branch> [pytest args...]
#   PY=/path/to/python tools/run_branch_tests.sh enc-q5 -k q5
set -e
BRANCH=${1:?usage: $0 <branch> [pytest args]}
shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HERE/../../brotli_pure_python/.venv/bin/python}
SNAP=$(mktemp -d "${TMPDIR:-/tmp}/brotli_pure_snap.XXXXXX")
trap 'rm -rf "$SNAP"' EXIT
git -C "$HERE" archive "$BRANCH" brotli_pure | tar -x -C "$SNAP"
echo "testing $BRANCH ($(git -C "$HERE" rev-parse --short "$BRANCH")) from $SNAP"
cd "$HERE"
BROTLI_PURE_IMPL="$SNAP" "$PY" -m pytest tests -p no:cacheprovider "$@"
