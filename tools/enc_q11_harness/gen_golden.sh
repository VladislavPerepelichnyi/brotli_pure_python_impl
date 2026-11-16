#!/bin/sh
# Builds small C harnesses that call internal functions of the reference
# encoder (google/brotli v1.2.1) directly and regenerates
# tests/data/enc_q11_golden.bin used by tests/test_enc_q11_dev.py.
#
#   sd_harness  FILE MINLEN               BrotliFindAllStaticDictionaryMatches per position
#   h10_harness FILE LGWIN Q              FindAllMatchesH10 (+StoreRangeH10) like the HQ loop
#   zop_harness FILE LGWIN Q [FONT]       BrotliCreate{Hq,}ZopfliBackwardReferences commands
#   mb_harness  FILE LGWIN Q FONT CMODE   + BrotliBuildMetaBlock (splits, context maps, histograms)
#
# Usage: tools/enc_q11_harness/gen_golden.sh [path/to/brotli_reference]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REF=${1:-$HERE/../../../../brotli_reference}
R=$REF/c
OUT=${TMPDIR:-/tmp}/enc_q11_harness.$$
mkdir -p "$OUT/obj"
for f in "$R"/common/*.c "$R"/enc/*.c; do
  gcc -O2 -c -I"$R/include" "$f" -o "$OUT/obj/$(basename "$f" .c).o"
done
ar rcs "$OUT/libbref.a" "$OUT"/obj/*.o
for h in sd h10 zop mb; do
  gcc -O2 -I"$R" -I"$R/include" "$HERE/${h}_harness.c" "$OUT/libbref.a" -lm -o "$OUT/${h}_harness"
done
python3 "$HERE/gen_golden.py" "$OUT" "$REF/tests/testdata/alice29.txt" "$HERE/../../tests/data/enc_q11_golden.bin"
