"""Generates tests/data/enc_q11_golden.bin (zlib-compressed JSON) from the C
harness binaries; see gen_golden.sh."""
import base64
import json
import os
import random
import subprocess
import sys
import zlib

bindir, alice, dst = sys.argv[1:4]
g1 = (b'Once upon a time, THE Quick brown fox said: "Hello World". The end of '
      b'the story is not the end of the world, it is the beginning. <a href='
      b'"http://www.example.com/">Example</a> e lifes Information, '
      b'Programming and the Development of new things.')
random.seed(7)
w = open(alice, "rb").read()[:1500]
g2 = w + bytes(random.getrandbits(8) for _ in range(300)) + w[200:900] + \
    b"z" * 400
p1 = os.path.join(bindir, "g1.bin")
p2 = os.path.join(bindir, "g2.bin")
open(p1, "wb").write(g1)
open(p2, "wb").write(g2)


def run(name, *args):
    exe = os.path.join(bindir, name)
    return subprocess.run((exe,) + args, capture_output=True,
                          check=True).stdout.decode()


gold = {
    "g1": base64.b64encode(g1).decode(),
    "g2": base64.b64encode(g2).decode(),
    "static_dict_g1_4": run("sd_harness", p1, "4"),
    "static_dict_g1_9": run("sd_harness", p1, "9"),
    "h10_g2_w16_q11": run("h10_harness", p2, "16", "11"),
    "zopfli_g2_w16_q11": run("zop_harness", p2, "16", "11"),
    "zopfli_g2_w12_q10": run("zop_harness", p2, "12", "10"),
    "zopfli_g2_w18_q11_font": run("zop_harness", p2, "18", "11", "1"),
    "metablock_g2_w16_q11_m0_c2": run("mb_harness", p2, "16", "11", "0", "2"),
    "metablock_g2_w18_q11_m1_c3": run("mb_harness", p2, "18", "11", "1", "3"),
}
with open(dst, "wb") as f:
    f.write(zlib.compress(json.dumps(gold, sort_keys=True).encode(), 9))
print("wrote", dst)
