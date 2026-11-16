#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "enc/hash.h"
#include "enc/metablock.h"
#include "enc/quality.h"
#include "enc/memory.h"
#include "enc/backward_references_hq.h"
#include "common/context.h"
static void dump_split(const char* name, const BlockSplit* s) {
  printf("%s %zu %zu:", name, s->num_types, s->num_blocks);
  for (size_t i = 0; i < s->num_blocks; ++i) printf(" %u/%u", s->types[i], s->lengths[i]);
  printf("\n");
}
#define DUMPH(name, arr, n, sz) do { printf("%s %zu\n", name, (size_t)(n)); for (size_t i = 0; i < (n); ++i) { printf("%zu:", arr[i].total_count_); for (size_t k = 0; k < sz; ++k) if (arr[i].data_[k]) printf(" %zu=%u", k, arr[i].data_[k]); printf("\n"); } } while (0)
int main(int argc, char** argv) {
  FILE* f = fopen(argv[1], "rb"); size_t cap = 1 << 24; uint8_t* buf = calloc(cap + 1024, 1);
  size_t n = fread(buf, 1, cap, f); fclose(f);
  int lgwin = atoi(argv[2]); int quality = atoi(argv[3]); int font = atoi(argv[4]); int cmode = atoi(argv[5]);
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
    BrotliCreateHqZopfliBackwardReferences(&m, n, 0, buf, mask, BROTLI_CONTEXT_LUT(cmode), &params, &h, dist_cache, &last_insert_len, cmds, &num_commands, &num_literals);
  else
    BrotliCreateZopfliBackwardReferences(&m, n, 0, buf, mask, BROTLI_CONTEXT_LUT(cmode), &params, &h, dist_cache, &last_insert_len, cmds, &num_commands, &num_literals);
  if (last_insert_len > 0) { InitInsertCommand(&cmds[num_commands++], last_insert_len); }
  MetaBlockSplit mb; InitMetaBlockSplit(&mb);
  BrotliBuildMetaBlock(&m, buf, 0, mask, &params, 0, 0, cmds, num_commands, (ContextType)cmode, &mb);
  printf("dist %u %u %u %zu\n", params.dist.distance_postfix_bits, params.dist.num_direct_distance_codes, params.dist.alphabet_size_limit, params.dist.max_distance);
  for (size_t i = 0; i < num_commands; ++i) printf("%u %u %u %u %u\n", cmds[i].insert_len_, cmds[i].copy_len_, cmds[i].dist_extra_, cmds[i].cmd_prefix_, cmds[i].dist_prefix_);
  dump_split("lit", &mb.literal_split); dump_split("cmd", &mb.command_split); dump_split("dst", &mb.distance_split);
  printf("lcm"); for (size_t i = 0; i < mb.literal_context_map_size; ++i) printf(" %u", mb.literal_context_map[i]); printf("\n");
  printf("dcm"); for (size_t i = 0; i < mb.distance_context_map_size; ++i) printf(" %u", mb.distance_context_map[i]); printf("\n");
  DUMPH("lh", mb.literal_histograms, mb.literal_histograms_size, 256);
  DUMPH("ch", mb.command_histograms, mb.command_histograms_size, 704);
  DUMPH("dh", mb.distance_histograms, mb.distance_histograms_size, 544);
  return 0;
}
