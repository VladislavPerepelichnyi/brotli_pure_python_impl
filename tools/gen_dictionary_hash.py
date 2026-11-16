"""Generates brotli_pure/enc/dictionary_hash.bin from the reference C header
c/enc/dictionary_hash_inc.h (kStaticDictionaryHashWords /
kStaticDictionaryHashLengths, 32768 entries each).

File layout: 32768 little-endian uint16 words, then 32768 uint8 lengths.

Usage: python tools/gen_dictionary_hash.py /path/to/brotli_reference
"""

import os
import re
import struct
import sys

N = 32768


def parse_array(text, name):
    m = re.search(name + r'\[[^\]]*\]\s*=\s*\{([^}]*)\}', text)
    if not m:
        raise SystemExit('array %s not found' % name)
    vals = [int(v) for v in re.findall(r'\d+', m.group(1))]
    if len(vals) != N:
        raise SystemExit('array %s has %d entries' % (name, len(vals)))
    return vals


def main():
    ref = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', '..', 'brotli_reference')
    with open(os.path.join(ref, 'c', 'enc', 'dictionary_hash_inc.h')) as f:
        text = f.read()
    words = parse_array(text, 'kStaticDictionaryHashWords')
    lengths = parse_array(text, 'kStaticDictionaryHashLengths')
    out = os.path.join(os.path.dirname(__file__), '..', 'brotli_pure', 'enc',
                       'dictionary_hash.bin')
    with open(out, 'wb') as f:
        f.write(struct.pack('<%dH' % N, *words))
        f.write(bytes(lengths))
    print('wrote', out)


if __name__ == '__main__':
    main()
