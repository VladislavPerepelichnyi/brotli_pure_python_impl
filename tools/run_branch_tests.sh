#!/bin/sh
# Run the test-suite of this worktree against the *committed* state of another
# branch, or of several branches merged together ("decoder+enc-q11"), exported
# to a temporary snapshot so in-progress edits in other worktrees do not
# interfere.  Merging uses `git merge-tree --write-tree` (git >= 2.38): no
# branch, ref or worktree is modified.
#
#   tools/run_branch_tests.sh <branch>[+<branch>...] [pytest args...]
#   PY=/path/to/python tools/run_branch_tests.sh decoder+enc-q11 -q
set -e
SPEC=${1:?usage: $0 <branch>[+<branch>...] [pytest args]}
shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$HERE/../../brotli_pure_python/.venv/bin/python}
SNAP=$(mktemp -d "${TMPDIR:-/tmp}/brotli_pure_snap.XXXXXX")
trap 'rm -rf "$SNAP"' EXIT

FIRST=${SPEC%%+*}
TREE=$(git -C "$HERE" rev-parse "$FIRST^{tree}")
COMMIT=$(git -C "$HERE" rev-parse "$FIRST")
REST=${SPEC#"$FIRST"}
DESC="$FIRST@$(git -C "$HERE" rev-parse --short "$FIRST")"
while [ -n "$REST" ]; do
  REST=${REST#+}
  B=${REST%%+*}
  REST=${REST#"$B"}
  if ! TREE=$(git -C "$HERE" merge-tree --write-tree "$COMMIT" "$B"); then
    echo "merge conflict merging $B" >&2
    exit 2
  fi
  # temporary (unreferenced) commit object so that further merges have a base
  COMMIT=$(git -C "$HERE" commit-tree "$TREE" -p "$COMMIT" -p "$B" -m "tmp test merge")
  DESC="$DESC + $B@$(git -C "$HERE" rev-parse --short "$B")"
done
git -C "$HERE" archive "$TREE" brotli_pure | tar -x -C "$SNAP"
echo "testing $DESC (snapshot $SNAP)"
cd "$HERE"
BROTLI_PURE_IMPL="$SNAP" "$PY" -m pytest tests -p no:cacheprovider "$@"
