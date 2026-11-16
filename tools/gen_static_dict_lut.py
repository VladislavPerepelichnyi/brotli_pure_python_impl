#!/usr/bin/env python3
"""Generate brotli_pure/enc/static_dict_lut.bin from the reference C sources.

Parses ``c/enc/static_dict_lut_inc.h`` (google/brotli v1.2.1), which holds
``kStaticDictionaryBuckets`` (uint16_t[32768]) and ``kStaticDictionaryWords``
(DictWord[31705], DictWord = {uint8 len, uint8 transform, uint16 idx}).

Output layout (little endian):
    32768 * uint16  buckets
    31705 * 4 bytes words, each: len(u8) transform(u8) idx(u16)

Usage:
    python3 tools/gen_static_dict_lut.py path/to/brotli/c/enc/static_dict_lut_inc.h
"""

import os
import re
import struct
import sys

NUM_BUCKETS = 32768
NUM_ITEMS = 31705


def main(argv):
    src = argv[1] if len(argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "brotli_reference", "c", "enc",
        "static_dict_lut_inc.h")
    with open(src, "r") as f:
        text = f.read()
    b_start = text.index("kStaticDictionaryBuckets")
    w_start = text.index("kStaticDictionaryWords")
    b_body = text[text.index("{", b_start) + 1:text.index("}", b_start)]
    buckets = [int(x) for x in re.findall(r"\d+", b_body.replace("\n", ""))]
    w_body = text[text.index("= {", w_start) + 3:text.rindex("};")]
    words = [tuple(int(v) for v in m)
             for m in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}",
                                 w_body.replace("\n", ""))]
    assert len(buckets) == NUM_BUCKETS, len(buckets)
    assert len(words) == NUM_ITEMS, len(words)
    out = bytearray(struct.pack("<%dH" % NUM_BUCKETS, *buckets))
    for (ln, tr, idx) in words:
        out += struct.pack("<BBH", ln, tr, idx)
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "brotli_pure", "enc", "static_dict_lut.bin")
    with open(dst, "wb") as f:
        f.write(out)
    print("wrote %s (%d bytes)" % (dst, len(out)))


if __name__ == "__main__":
    main(sys.argv)
