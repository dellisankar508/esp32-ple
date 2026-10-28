#include "kern_check.h"
#include "kern.h"
#include <stdio.h>
#include <string.h>

typedef struct { char name[24]; uint32_t off, rows, in; } kv_head_t;

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int kern_check(const uint8_t *model, size_t model_size, const uint8_t *kv, size_t kv_size, float *y) {
    if (kv_size < 8 || memcmp(kv, "KVC1", 4) != 0) return -1;
    uint32_t n;
    memcpy(&n, kv + 4, 4);
    const uint8_t *p = kv + 8, *end = kv + kv_size;
    int bad = 0, split_bad = 0;
    unsigned long outputs = 0;
    for (uint32_t c = 0; c < n; c++) {
        kv_head_t h;
        if (p + sizeof h > end) return -1;
        memcpy(&h, p, sizeof h);
        p += sizeof h;
        const float *x = (const float *)p;            // 4-byte aligned: every field is
        const float *want = x + h.in;                 // a multiple of 4 bytes
        p += 4 * ((size_t)h.in + h.rows);
        if (p > end || h.off + (size_t)h.rows * kern_row_bytes(h.in) > model_size) return -1;
        const uint8_t *w = model + h.off;
        float *y2 = y + h.rows;

        kern_prepare(x, h.in);
        kern_matvec(w, h.in, x, y, 0, (int)h.rows);
        int mid = (int)h.rows / 2 + 1;                 // an uneven split, like core 0 / core 1
        kern_matvec(w, h.in, x, y2, mid, (int)h.rows);
        kern_matvec(w, h.in, x, y2, 0, mid);

        uint32_t diff = 0, first = 0;
        for (uint32_t r = 0; r < h.rows; r++)
            if (bits(y[r]) != bits(want[r])) { if (!diff) first = r; diff++; }
        if (memcmp(y, y2, 4 * (size_t)h.rows) != 0) split_bad++;
        if (diff) {
            bad++;
            printf("KCHECK | %.24s | %u x %u | %u of %u rows differ, first row %u: got %.9g (0x%08x) want %.9g (0x%08x)\n",
                   h.name, (unsigned)h.rows, (unsigned)h.in, (unsigned)diff, (unsigned)h.rows, (unsigned)first,
                   (double)y[first], (unsigned)bits(y[first]), (double)want[first], (unsigned)bits(want[first]));
        }
        outputs += h.rows;
    }
    printf("KERNEL | %s | %s %u/%u cases | %lu outputs | split rows: %s\n", kern_name(),
           bad ? "MISMATCH" : "BITEXACT", (unsigned)(n - bad), (unsigned)n, outputs,
           split_bad ? "DIFFERENT" : "same");
    return bad + split_bad;
}
