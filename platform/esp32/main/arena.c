#include "arena.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <stdlib.h>

static const char *TAG = "arena";
#define CAPS (MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL)

typedef struct { uint8_t *base; size_t cap, used; } bank_t;
static bank_t banks[ARENA_BANKS];

void arena_init(size_t margin) {
    for (int b = 0; b < ARENA_BANKS; b++) {
        size_t largest = heap_caps_get_largest_free_block(CAPS);
        if (largest <= margin + 4096) { banks[b].cap = 0; continue; }
        size_t want = largest - margin;
        uint8_t *p = heap_caps_malloc(want, CAPS);
        if (!p) { banks[b].cap = 0; continue; }
        banks[b].base = p; banks[b].cap = want; banks[b].used = 0;
        ESP_LOGI(TAG, "bank %d: %u B at %p", b, (unsigned)want, p);
    }
    if (banks[0].cap == 0) { ESP_LOGE(TAG, "no DRAM for arena"); abort(); }
}

void *arena_alloc_bank(int b, size_t bytes) {
    bank_t *k = &banks[b];
    size_t off = (k->used + 15) & ~(size_t)15;
    if (off + bytes > k->cap) return NULL;
    k->used = off + bytes;
    return k->base + off;
}

void *arena_alloc(size_t bytes) {
    for (int b = 0; b < ARENA_BANKS; b++) {
        void *p = arena_alloc_bank(b, bytes);
        if (p) return p;
    }
    ESP_LOGE(TAG, "arena exhausted: need %u B, largest free %u B",
             (unsigned)bytes, (unsigned)arena_largest_free());
    abort();
}

size_t arena_capacity(int b)  { return banks[b].cap; }
size_t arena_used(int b)      { return banks[b].used; }
size_t arena_remaining(int b) { return banks[b].cap - banks[b].used; }
size_t arena_largest_free(void) {
    size_t m = 0;
    for (int b = 0; b < ARENA_BANKS; b++)
        if (arena_remaining(b) > m) m = arena_remaining(b);
    return m;
}
void arena_reset(void) { for (int b = 0; b < ARENA_BANKS; b++) banks[b].used = 0; }
void arena_report(void) {
    size_t total = 0;
    for (int b = 0; b < ARENA_BANKS; b++) {
        ESP_LOGI(TAG, "bank %d: %u / %u B used", b,
                 (unsigned)banks[b].used, (unsigned)banks[b].cap);
        total += banks[b].cap;
    }
    ESP_LOGI(TAG, "total arena %u B", (unsigned)total);
}
