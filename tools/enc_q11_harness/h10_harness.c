#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "enc/hash.h"
#include "enc/metablock.h"
#include "enc/quality.h"
#include "enc/memory.h"
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); size_t cap = 1 << 23; uint8_t* buf = calloc(cap + 1024, 1);
  size_t n = fread(buf, 1, cap, f); fclose(f);
  int lgwin = atoi(argv[2]); int quality = atoi(argv[3]);
  size_t mask = cap - 1;
  MemoryManager m; BrotliInitMemoryManager(&m, 0, 0, 0);
  BrotliEncoderParams params; memset(&params, 0, sizeof(params));
  params.quality = quality; params.lgwin = lgwin; params.mode = BROTLI_MODE_GENERIC;
  BrotliInitDistanceParams(&params.dist, 0, 0, 0);
  BrotliInitSharedEncoderDictionary(&params.dictionary);
  Hasher h; HasherInit(&h);
  HasherSetup(&m, &h, &params, buf, 0, n, 1);
  size_t max_backward_limit = BROTLI_MAX_BACKWARD_LIMIT(lgwin);
  size_t store_end = n >= 128 ? n - 128 + 1 : 0;
  BackwardMatch matches[256];
  for (size_t i = 0; i + 3 < n; ++i) {
    size_t max_distance = i < max_backward_limit ? i : max_backward_limit;
    size_t num = FindAllMatchesH10(&h.privat._H10, params.dictionary.contextual.dict[0], buf, mask, i, n - i,
        max_distance, max_distance, &params, matches);
    printf("%zu:", i);
    for (size_t j = 0; j < num; ++j) printf(" %u/%u", matches[j].distance, matches[j].length_and_code);
    printf("\n");
    if (num > 0) {
      size_t ml = BackwardMatchLength(&matches[num - 1]);
      if (ml > 325) {
        StoreRangeH10(&h.privat._H10, buf, mask, i + 1, (i + ml < store_end ? i + ml : store_end));
        i += ml - 1;
      }
    }
  }
  return 0;
}
