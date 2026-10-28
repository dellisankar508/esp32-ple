// mve: Helium (M-Profile Vector Extension) kernel for Cortex-M55 / M85.
// Same I8 v1 contract as i8ref.c, bit for bit (checked by core/kern_check.c).
//
// Integer part, per group of 32 int4 weights (16 bytes, one vector load):
//   each byte holds an even column (low nibble) and an odd column (high nibble)
//   lo16 = b << 4       every lane = even weight * 16    (exact: -128..112 fits int8)
//   hi16 = b & 0xF0     every lane = odd weight * 16
//   d16  = VMLADAV(lo16, q_even) + VMLADAV(hi16, q_odd)   = 16 * exact dot, in int32
// Keeping the weights scaled by 16 saves the two sign-extending shifts per group, and the
// 16 is divided out of the input scale instead (sx / 16, exact), so no shift is needed:
// (16 d) * (s * sx/16) rounds exactly like d * (s * sx) because powers of two scale exactly.
// kern_prepare stores each group's int8 inputs de-interleaved (16 even, then 16 odd).
//
// Float part: once the dot products were vectorised, the scalar float bookkeeping per
// group (scale load, fp16 convert, two multiplies, an add) cost three times the vector
// work. So rows go four at a time: four dots fill one vector, and the four rows' scales
// are converted with the scalar fp16 instruction straight into the vector's lanes. (The
// vector VCVTB.F32.F16 would be one instruction, but GNU as encodes it wrongly in some
// binutils versions and QEMU faults on it; binutils has a fix for exactly this.)
// Each lane still computes acc + d * (s * sx) in the
// same order as i8ref.c, and the I8 v1 cut-off keeps every value a normal float, so the
// result is identical bit for bit. Leftover rows (count not a multiple of 4) take the
// one-row path.
#include "kern.h"
#include "q8.h"
#include "f16.h"
#include "ple_port.h"

#if !defined(__ARM_FEATURE_MVE) || !(__ARM_FEATURE_MVE & 2)
#error "kernels/mve.c needs Helium with floating point: build with -mcpu=cortex-m55 (or cortex-m85)"
#endif
#include <arm_mve.h>

static int8_t qx[KERN_MAX_IN] __attribute__((aligned(16)));    // per group: 16 even, 16 odd
static float  sx[KERN_MAX_IN / 32];
static float  sx16[KERN_MAX_IN / 32];                           // sx / 16, see kern_matvec

const char *kern_name(void) { return "mve"; }

void kern_prepare(const float *x, uint32_t in) {
    int8_t q[32];
    for (uint32_t g = 0; g < in / 32; g++) {
        kern_q8_group(x + 32 * g, q, &sx[g]);
        sx16[g] = sx[g] * 0.0625f;                               // exact: a power of two
        for (int j = 0; j < 16; j++) {
            qx[32 * g + j] = q[2 * j];
            qx[32 * g + 16 + j] = q[2 * j + 1];
        }
    }
}

// 16 * the exact dot of one group.
static inline int32_t dot16_group(const uint8_t *w, int8x16_t qe, int8x16_t qo, int8x16_t hi_mask) {
    int8x16_t b = vldrbq_s8((const int8_t *)w);
    int32_t d16 = vmladavq_s8(vshlq_n_s8(b, 4), qe);
    return vmladavaq_s8(d16, vandq_s8(b, hi_mask), qo);
}

PLE_HOT void kern_matvec(const uint8_t *w, uint32_t in, const float *x, float *y, int r0, int r1) {
    (void)x;
    const uint32_t rb = kern_row_bytes(in), ng = in / 32;
    const int8x16_t hi_mask = vdupq_n_s8((int8_t)0xF0);
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const uint8_t *row = w + (uint32_t)r * rb;
        float32x4_t acc = vdupq_n_f32(0.f);
        for (uint32_t g = 0; g < ng; g++) {
            int8x16_t qe = vldrbq_s8(qx + 32 * g), qo = vldrbq_s8(qx + 32 * g + 16);
            int32x4_t d = vdupq_n_s32(dot16_group(row + 16 * g, qe, qo, hi_mask));
            d = vsetq_lane_s32(dot16_group(row + rb + 16 * g, qe, qo, hi_mask), d, 1);
            d = vsetq_lane_s32(dot16_group(row + 2 * rb + 16 * g, qe, qo, hi_mask), d, 2);
            d = vsetq_lane_s32(dot16_group(row + 3 * rb + 16 * g, qe, qo, hi_mask), d, 3);
            // the four rows' fp16 scales for group g, widened to fp32 (exact)
            const uint8_t *sc = row + in / 2 + 2 * g;
            float32x4_t s = vdupq_n_f32((float)*(const __fp16 *)sc);
            s = vsetq_lane_f32((float)*(const __fp16 *)(sc + rb), s, 1);
            s = vsetq_lane_f32((float)*(const __fp16 *)(sc + 2 * rb), s, 2);
            s = vsetq_lane_f32((float)*(const __fp16 *)(sc + 3 * rb), s, 3);
            // (16 d) * (s * sx/16) rounds exactly like d * (s * sx): scaling by 16 is exact
            float32x4_t c = vmulq_n_f32(s, sx16[g]);
            acc = vaddq_f32(acc, vmulq_f32(vcvtq_f32_s32(d), c));
        }
        vst1q_f32(y + r, acc);
    }
    for (; r < r1; r++) {                                       // leftover rows, one at a time
        const uint8_t *row = w + (uint32_t)r * rb;
        const __fp16 *sc = (const __fp16 *)(row + in / 2);
        float acc = 0.f;
        for (uint32_t g = 0; g < ng; g++) {
            int32_t d16 = dot16_group(row + 16 * g, vldrbq_s8(qx + 32 * g), vldrbq_s8(qx + 32 * g + 16), hi_mask);
            float c = (float)sc[g] * sx16[g];
            acc += (float)d16 * c;
        }
        y[r] = acc;
    }
}
