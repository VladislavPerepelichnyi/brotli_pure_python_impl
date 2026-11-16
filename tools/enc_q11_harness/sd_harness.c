#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "enc/encoder_dict.h"
#include "enc/static_dict.h"
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); static uint8_t buf[1<<22]; size_t n = fread(buf, 1, sizeof(buf)-64, f); fclose(f);
  size_t minlen = argc > 2 ? atoi(argv[2]) : 4;
  SharedEncoderDictionary sd; BrotliInitSharedEncoderDictionary(&sd);
  const BrotliEncoderDictionary* d = &sd.contextual.instance_;
  for (size_t p = 0; p + 4 <= n; ++p) {
    uint32_t m[38]; for (int i = 0; i < 38; ++i) m[i] = kInvalidMatch;
    int r = BrotliFindAllStaticDictionaryMatches(d, buf + p, minlen, n - p, m);
    printf("%zu %d", p, r);
    for (int i = 0; i < 38; ++i) if (m[i] != kInvalidMatch) printf(" %d:%u", i, m[i]);
    printf("\n");
  }
  return 0;
}
