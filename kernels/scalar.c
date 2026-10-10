// Scalar backend: the ESP32 Phase 3 kernel, unchanged. Float activations,
// 32-bit weight loads, nibbles sign-extended with shifts.
#include "kern.h"
#include "f16.h"
#include "ple_port.h"

const char *kern_name(void) { return "scalar"; }

void kern_prepare(const float *x, uint32_t in) { (void)x; (void)in; }

// 32 int4 weights (16 bytes, 4 x u32) dotted with 32 floats.
static inline float dot32(const uint8_t *q, const float *x) {
    const uint32_t *w = (const uint32_t *)q;
    float s = 0.f;
    for (int j = 0; j < 4; j++) {
        uint32_t b = w[j];
        const float *xx = x + 8 * j;
        s += ((int32_t)(b << 28) >> 28) * xx[0] + ((int32_t)(b << 24) >> 28) * xx[1]
           + ((int32_t)(b << 20) >> 28) * xx[2] + ((int32_t)(b << 16) >> 28) * xx[3]
           + ((int32_t)(b << 12) >> 28) * xx[4] + ((int32_t)(b <<  8) >> 28) * xx[5]
           + ((int32_t)(b <<  4) >> 28) * xx[6] + ((int32_t) b        >> 28) * xx[7];
    }
    return s;
}

PLE_HOT void kern_matvec(const uint8_t *w, uint32_t in, const float *x, float *y, int r0, int r1) {
    const uint32_t rb = kern_row_bytes(in), ng = in / 32;
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = w + (uint32_t)r * rb;
        const uint16_t *sc = (const uint16_t *)(row + in / 2);
        float acc = 0.f;
        for (uint32_t g = 0; g < ng; g++)
            acc += f16_to_f32(sc[g]) * dot32(row + g * 16, x + g * 32);
        y[r] = acc;
    }
}
