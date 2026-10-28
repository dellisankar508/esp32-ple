// i8ref: the int8-activation reference kernel, plain C for any target.
// It defines the answer the vector kernels must reproduce bit for bit (I8 v1, see
// train/export_ple.py matvec_i8 and kernels/q8.h):
//   kern_prepare  x -> int8 per group of 32, one float scale per group
//   kern_matvec   per row and group: exact int32 dot of int4 weights x int8 inputs,
//                 then acc += (float)dot * (weight scale * input scale), groups in order
// Weights are read exactly as stored (PLE1 rows), so nothing changes in model.bin.
#include "kern.h"
#include "q8.h"
#include "f16.h"
#include "ple_port.h"

static int8_t qx[KERN_MAX_IN] __attribute__((aligned(16)));
static float  sx[KERN_MAX_IN / 32];

const char *kern_name(void) { return "i8ref"; }

void kern_prepare(const float *x, uint32_t in) {
    for (uint32_t g = 0; g < in / 32; g++) kern_q8_group(x + 32 * g, qx + 32 * g, &sx[g]);
}

// 32 int4 weights (16 bytes) dotted with 32 int8 inputs, exact.
static inline int32_t dot32(const uint8_t *w, const int8_t *q) {
    int32_t d = 0;
    for (int j = 0; j < 16; j++) {
        uint8_t b = w[j];
        d += ((int8_t)(b << 4) >> 4) * q[2 * j]          // low nibble = even column
           + ((int8_t)b >> 4) * q[2 * j + 1];            // high nibble = odd column
    }
    return d;
}

PLE_HOT void kern_matvec(const uint8_t *w, uint32_t in, const float *x, float *y, int r0, int r1) {
    (void)x;                                             // kern_prepare already took it
    const uint32_t rb = kern_row_bytes(in), ng = in / 32;
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = w + (uint32_t)r * rb;
        const uint16_t *sc = (const uint16_t *)(row + in / 2);
        float acc = 0.f;
        for (uint32_t g = 0; g < ng; g++) {
            float c = f16_to_f32(sc[g]) * sx[g];
            acc += (float)dot32(row + 16 * g, qx + 32 * g) * c;
        }
        y[r] = acc;
    }
}
