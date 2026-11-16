#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "enc/hash.h"
#include "enc/metablock.h"
#include "enc/quality.h"
#include "enc/memory.h"
#include "enc/backward_references_hq.h"
#include "common/context.h"
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); size_t cap = 1 << 24; uint8_t* buf = calloc(cap + 1024, 1);
  size_t n = fread(buf, 1, cap, f); fclose(f);
  int lgwin = atoi(argv[2]); int quality = atoi(argv[3]); int font = argc > 4 ? atoi(argv[4]) : 0;
  size_t mask = cap - 1;
  MemoryManager m; BrotliInitMemoryManager(&m, 0, 0, 0);
  BrotliEncoderParams params; memset(&params, 0, sizeof(params));
  params.quality = quality; params.lgwin = lgwin; params.mode = font ? BROTLI_MODE_FONT : BROTLI_MODE_GENERIC;
  if (font) BrotliInitDistanceParams(&params.dist, 1, 12, 0); else BrotliInitDistanceParams(&params.dist, 0, 0, 0);
  BrotliInitSharedEncoderDictionary(&params.dictionary);
  Hasher h; HasherInit(&h);
  HasherSetup(&m, &h, &params, buf, 0, n, 1);
  int dist_cache[16] = {4, 11, 15, 16};
  size_t last_insert_len = 0, num_commands = 0, num_literals = 0;
  Command* cmds = malloc(sizeof(Command) * (n / 2 + 16));
  if (quality == 11)
    BrotliCreateHqZopfliBackwardReferences(&m, n, 0, buf, mask, BROTLI_CONTEXT_LUT(CONTEXT_UTF8), &params, &h, dist_cache, &last_insert_len, cmds, &num_commands, &num_literals);
  else
    BrotliCreateZopfliBackwardReferences(&m, n, 0, buf, mask, BROTLI_CONTEXT_LUT(CONTEXT_UTF8), &params, &h, dist_cache, &last_insert_len, cmds, &num_commands, &num_literals);
  for (size_t i = 0; i < num_commands; ++i)
    printf("%u %u %u %u %u\n", cmds[i].insert_len_, cmds[i].copy_len_, cmds[i].dist_extra_, cmds[i].cmd_prefix_, cmds[i].dist_prefix_);
  printf("end %zu %zu %zu %d %d %d %d\n", num_commands, last_insert_len, num_literals, dist_cache[0], dist_cache[1], dist_cache[2], dist_cache[3]);
  return 0;
}
