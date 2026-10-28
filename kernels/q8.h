#pragma once
#include <stdint.h>
#include <math.h>
#include "kern.h"
// Shared prepare step for every int8-activation kernel (i8ref.c, dsp.c, mve.c): the
// I8 v1 contract, step 1 (see train/export_ple.py, matvec_i8). Each group of 32 inputs
// gets its own scale sx = max|x|/127 and int8 values q = rint(x * 127/max|x|).
//
// Bit-exactness rules for anything that includes this header:
//  - compile with -ffp-contract=off: a fused multiply-add rounds once where the
//    reference rounds twice
//  - rounding is half to even (default rounding mode), like numpy's rint

#ifndef KERN_MAX_IN
#define KERN_MAX_IN 1024               // largest matvec input length any model uses
#endif

static inline void kern_q8_group(const float *x, int8_t *q, float *sx) {
    float m = 0.f;
    for (int i = 0; i < 32; i++) { float a = fabsf(x[i]); if (a > m) m = a; }
    if (m < 0x1p-60f) {                 // zero or vanishing group: keeps every product normal
        for (int i = 0; i < 32; i++) q[i] = 0;
        *sx = 0.f;
        return;
    }
    float inv = 127.0f / m;
    *sx = m / 127.0f;
    for (int i = 0; i < 32; i++) {
        // Round half to even without a library call: |x * inv| <= 127.00002, and adding
        // 1.5 * 2^23 makes the float's last bit worth 1, so the add itself rounds to an
        // integer, ties to even, exactly as rintf / numpy's rint would.
        int v = (int)((x[i] * inv + 0x1.8p23f) - 0x1.8p23f);
        q[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
    }
}
