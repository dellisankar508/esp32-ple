// Kernel bit-exactness check on the host:  kern_test model.bin kvec.bin
// Same check as the embedded targets run (core/kern_check.c).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kern_check.h"

static uint8_t *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n);
    if (fread(b, 1, *n, f) != *n) { perror(path); exit(1); }
    fclose(f);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin kvec.bin\n", argv[0]); return 1; }
    size_t ms, ks;
    uint8_t *m = slurp(argv[1], &ms), *k = slurp(argv[2], &ks);
    uint32_t vocab;
    memcpy(&vocab, m + 8, 4);
    float *y = malloc(2 * sizeof(float) * (vocab > 4096 ? vocab : 4096));
    int r = kern_check(m, ms, k, ks, y);
    if (r < 0) fprintf(stderr, "bad kvec file\n");
    return r ? 2 : 0;
}
