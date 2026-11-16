#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "enc/literal_cost.h"
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); size_t cap = 1 << 22; uint8_t* buf = calloc(cap + 64, 1);
  size_t n = fread(buf, 1, cap, f); fclose(f);
  size_t pos = argc > 2 ? atoi(argv[2]) : 0; size_t mask = argc > 3 ? (size_t)atoi(argv[3]) : cap - 1;
  /* rotate data into a ring buffer of size mask+1 starting at pos */
  uint8_t* rb = calloc(mask + 1 + 64, 1); for (size_t i = 0; i < n && i <= mask; ++i) rb[(pos + i) & mask] = buf[i];
  size_t len = n <= mask + 1 ? n : mask + 1;
  size_t hist[3 * 256]; float* cost = malloc(sizeof(float) * (len + 1));
  BrotliEstimateBitCostsForLiterals(pos, len, mask, rb, hist, cost);
  for (size_t i = 0; i < len; ++i) { uint32_t u; memcpy(&u, &cost[i], 4); printf("%08x\n", u); }
  return 0;
}
