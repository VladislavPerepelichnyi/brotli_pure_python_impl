/* Prints kBrotliLog2Table from the reference C sources as hex floats.
   Build:  gcc -I$BROTLI/c/include -I$BROTLI/c/enc tools/check_log2_table.c
   Compare against brotli_pure.enc.fast_log.kBrotliLog2Table
   (each entry == float32(log2(i)) widened to double). */
#include <stdio.h>
#include "fast_log.c"
int main(void) {
  int i;
  for (i = 0; i < 256; i++) printf("%a\n", kBrotliLog2Table[i]);
  return 0;
}
