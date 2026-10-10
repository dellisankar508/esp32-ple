#include "tokenizer.h"
#include <string.h>

int tokenizer_load(tokenizer_t *t, const uint8_t *b, size_t size, void *(*alloc)(size_t)) {
    if (size < 4) return -1;
    t->base = b; t->n = *(const uint32_t *)b;
    t->tok = alloc(t->n * sizeof(const uint8_t *));
    if (!t->tok) return -2;
    const uint8_t *p = b + 4;
    for (uint32_t i = 0; i < t->n; i++) { t->tok[i] = p; p += 1 + p[0]; if (p > b + size) return -3; }
    return 0;
}

const uint8_t *tokenizer_bytes(const tokenizer_t *t, int id, int *len) {
    if (id < 0 || (uint32_t)id >= t->n) { *len = 0; return (const uint8_t *)""; }
    *len = t->tok[id][0]; return t->tok[id] + 1;
}

int tokenizer_encode(const tokenizer_t *t, const char *text, int *out, int max) {
    int n = 0; size_t pos = 0, len = strlen(text);
    while (pos < len && n < max) {
        int best = -1, bl = 0;
        for (uint32_t i = 0; i < t->n; i++) {
            int l = t->tok[i][0];
            if (l > bl && pos + l <= len && memcmp(t->tok[i] + 1, text + pos, l) == 0) { best = i; bl = l; }
        }
        if (best < 0) { pos++; continue; }          // skip unknown byte
        out[n++] = best; pos += bl;
    }
    return n;
}
