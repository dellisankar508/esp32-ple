// dsp: Cortex-M33 / M4 / M7 kernel using the DSP extension (SIMD32): two 16-bit
// multiply-adds per SMLAD. Same I8 v1 contract as i8ref.c, bit for bit.
//
// A 32-bit weight word holds 8 columns, byte k = column 2k (low nibble) and 2k+1 (high):
//   L = (w & 0x0F0F0F0F) << 4   bytes = even weights * 16 (exact, no carry between bytes)
//   H =  w & 0xF0F0F0F0         bytes = odd weights * 16
//   SXTB16(L)        -> [c0, c4] * 16      SXTB16(L, ROR 8) -> [c2, c6] * 16
//   SXTB16(H)        -> [c1, c5] * 16      SXTB16(H, ROR 8) -> [c3, c7] * 16
// kern_prepare packs the int8 inputs as 16-bit pairs in that same order, so each
// SMLAD multiplies matching columns. The int32 sum is 16 * the exact dot; the 16 is
// divided out of the input scale instead (sx / 16, exact), so no shift is needed.
// 8 weights cost 1 weight load, 2 LDRD of inputs, 2 ANDs (the mask stays in a register,
// and L uses the shifted-operand form), 4 SXTB16 (the ROR folded in) and 4 SMLAD.
#include "kern.h"
#include "q8.h"
#include "f16.h"
#include "ple_port.h"

#if !defined(__ARM_FEATURE_DSP)
#error "kernels/dsp.c needs the Arm DSP extension (Cortex-M33/M4/M7): build with -mcpu=cortex-m33"
#endif
#include <arm_acle.h>

// Per group: 16 words, 4 per 8 columns: (q0,q4) (q2,q6) (q1,q5) (q3,q7), low half first.
static uint32_t qp[KERN_MAX_IN / 2] __attribute__((aligned(16)));
static float    sx[KERN_MAX_IN / 32];
static float    sx16[KERN_MAX_IN / 32];                // sx / 16

const char *kern_name(void) { return "dsp"; }

// SXTB16 with its rotate operand: GCC does not fold a separate ROR into it by itself.
static inline uint32_t sxtb16_ror8(uint32_t v) {
    uint32_t r;
    __asm__("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(v));
    return r;
}

static inline uint32_t pair(int8_t a, int8_t b) { return (uint16_t)(int16_t)a | ((uint32_t)(uint16_t)(int16_t)b << 16); }

void kern_prepare(const float *x, uint32_t in) {
    int8_t q[32];
    for (uint32_t g = 0; g < in / 32; g++) {
        kern_q8_group(x + 32 * g, q, &sx[g]);
        sx16[g] = sx[g] * 0.0625f;                     // exact: a power of two
        uint32_t *p = qp + 16 * g;
        for (int k = 0; k < 4; k++) {                 // 4 words of 8 columns each
            const int8_t *c = q + 8 * k;
            p[4 * k + 0] = pair(c[0], c[4]);
            p[4 * k + 1] = pair(c[2], c[6]);
            p[4 * k + 2] = pair(c[1], c[5]);
            p[4 * k + 3] = pair(c[3], c[7]);
        }
    }
}

// 16 * the exact dot of one group of 32.
static inline int32_t dot16_32(const uint32_t *w, const uint32_t *p, uint32_t m) {
    int32_t s = 0;
#pragma GCC unroll 4
    for (int k = 0; k < 4; k++, p += 4) {
        uint32_t v = w[k];
        uint32_t lo = (v << 4) & m, hi = v & m;       // m = 0xF0F0F0F0, kept in a register
        s = __smlad(__sxtb16(lo), p[0], s);
        s = __smlad(sxtb16_ror8(lo), p[1], s);
        s = __smlad(__sxtb16(hi), p[2], s);
        s = __smlad(sxtb16_ror8(hi), p[3], s);
    }
    return s;
}

// fp16 -> fp32 is exact either way; with -mfp16-format=ieee the FPU converts it.
static inline float scale_f32(const uint16_t *p) {
#if defined(__ARM_FP16_FORMAT_IEEE)
    return (float)*(const __fp16 *)p;
#else
    return f16_to_f32(*p);
#endif
}

PLE_HOT void kern_matvec(const uint8_t *w, uint32_t in, const float *x, float *y, int r0, int r1) {
    (void)x;
    const uint32_t rb = kern_row_bytes(in), ng = in / 32;
    uint32_t m = 0xF0F0F0F0u;
    __asm__("" : "+r"(m));                            // keep the mask in a register
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = w + (uint32_t)r * rb;
        const uint16_t *sc = (const uint16_t *)(row + in / 2);
        float acc = 0.f;
        for (uint32_t g = 0; g < ng; g++) {
            // (16 d) * (s * sx/16) rounds exactly like d * (s * sx)
            float c = scale_f32(sc + g) * sx16[g];
            acc += (float)dot16_32((const uint32_t *)(row + 16 * g), qp + 16 * g, m) * c;
        }
        y[r] = acc;
    }
}
