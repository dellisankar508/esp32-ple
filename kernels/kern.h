#pragma once
#include <stdint.h>
// Kernel backend interface: the one seam between the model code and the math.
//
// Weights are row-packed int4 with fp16 group-32 scales (PLE1 format):
//   row = in/2 bytes of nibbles (low nibble = even column), then in/16 bytes of
//   fp16 scales, padded to 4 bytes.
//
// Calling contract, per matvec:
//   kern_prepare(x, in);                       once, on the calling core
//   kern_matvec(w, in, x, y, r0, r1);          any number of row ranges, any core
// kern_matvec may read state written by kern_prepare (e.g. int8-quantized x), so
// prepare must finish before any core runs a range, and x must not change between.

#define KERN_ALIGN4(n) (((n) + 3) & ~3u)
static inline uint32_t kern_row_bytes(uint32_t in) { return KERN_ALIGN4(in / 2 + in / 16); }

const char *kern_name(void);
void kern_prepare(const float *x, uint32_t in);
void kern_matvec(const uint8_t *w, uint32_t in, const float *x, float *y, int r0, int r1);
