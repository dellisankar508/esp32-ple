#pragma once
#include <stdint.h>
#include <stddef.h>
// tokenizer.bin: u32 count, then per token u8 len + raw bytes (decode-only).
typedef struct { const uint8_t *base; uint32_t n; const uint8_t **tok; } tokenizer_t;
int  tokenizer_load(tokenizer_t *t, const uint8_t *base, size_t size, void *(*alloc)(size_t));
// returns pointer to the token's bytes and sets *len
const uint8_t *tokenizer_bytes(const tokenizer_t *t, int id, int *len);
// greedy longest-match encode (good enough for short prompts); returns count
int  tokenizer_encode(const tokenizer_t *t, const char *text, int *out, int max);
