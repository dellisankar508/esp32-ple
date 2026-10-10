#pragma once
#include <stdint.h>
#include <string.h>
// IEEE half <-> float, no hardware needed (Xtensa LX6 has none).
static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) {
        if (m == 0) f = s;
        else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; f = s | ((e + 112) << 23) | (m << 13); }
    } else if (e == 31) f = s | 0x7F800000 | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}
static inline uint16_t f32_to_f16(float x) {
    uint32_t f; memcpy(&f, &x, 4);
    uint32_t s = (f >> 16) & 0x8000; int32_t e = ((f >> 23) & 0xFF) - 112; uint32_t m = f & 0x7FFFFF;
    if (e <= 0) { if (e < -10) return s; m |= 0x800000; uint32_t sh = 14 - e;
        uint32_t r = m >> sh; if ((m >> (sh - 1)) & 1) r++; return s | r; }
    if (e >= 31) return s | 0x7C00;
    uint32_t r = s | (e << 10) | (m >> 13); if (m & 0x1000) r++; return r;
}
