#pragma once
#include <stddef.h>
#include <stdint.h>
// PLE1 int4 runtime, portable core.
//  - weights stay where `base` points (mmapped/XIP flash on a chip, a buffer on the host);
//    RAM holds only the KV cache, activations and logits
//  - int8 KV cache with one fp32 scale per (layer, position, head)
//  - every weight matvec goes through kernels/kern.h, so backends are swappable
//  - optional parallel hook splits the output head across cores
//  - optional head hook lets a platform move the head weights itself (DMA streaming)
#define PLE_MAX_LAYERS 8

typedef struct {
    uint32_t V, D, L, H, F, P, C, G;
    uint32_t size;                      // bytes in the PLE1 image (header total)
    const uint8_t *base, *tok, *ple;
    const float *norm;
    struct {
        const uint8_t *ple_proj, *qkv, *o, *w1, *w3, *w2;
        const float *n1, *n2;
    } layer[PLE_MAX_LAYERS];
} ple_model_t;

// Run job(arg, r0, r1) over rows [0, n), split however the platform likes.
// Must return only when every row is done.
typedef void (*ple_job_fn)(void *arg, int r0, int r1);
typedef void (*ple_parallel_fn)(ple_job_fn job, void *arg, int n);

// Optional replacement for the output head, y[r] = row r of w . x for r in [0, rows).
// For platforms that move the weights themselves, e.g. DMA from flash into SRAM
// while the cores compute (core/head_stream.c). `par` is s->parallel (NULL = one core).
// kern_prepare(x, in) has already been called.
typedef void (*ple_head_fn)(ple_parallel_fn par, const uint8_t *w, uint32_t rows, uint32_t in,
                            const float *x, float *y);

typedef struct {
    int ctx;
    float *h, *x, *qkv, *att, *f1, *f3, *tmp, *ple_row, *logits;
    int8_t *kc, *vc;               // [L][ctx][D] int8
    float  *ks, *vs;               // [L][ctx][H] scales
    float  *rope_cos, *rope_sin;   // [ctx][D/H/2]
    ple_parallel_fn parallel;      // NULL = single core
    ple_head_fn head;              // NULL = kern_matvec straight from w
    uint32_t t_embed, t_attn, t_ffn, t_head;   // microseconds, last forward
} ple_state_t;

typedef void *(*ple_alloc_fn)(size_t);
typedef uint32_t (*ple_time_fn)(void);         // microsecond clock, may be NULL

int    ple_load(ple_model_t *m, const uint8_t *base, size_t size);
size_t ple_state_bytes(const ple_model_t *m, int ctx);
size_t ple_bytes_per_token(const ple_model_t *m);   // weight bytes read per token
// Weight bytes each timed stage reads per token: [0] embed (PLE slices + ple_proj),
// [1] attention (n1, qkv, o), [2] FFN (n2, w1, w3, w2), [3] head (+ final norm).
// Sum + one embedding row = ple_bytes_per_token.
void   ple_stage_bytes(const ple_model_t *m, size_t out[4]);
int    ple_state_init(ple_state_t *s, const ple_model_t *m, int ctx, ple_alloc_fn alloc);
void   ple_forward(const ple_model_t *m, ple_state_t *s, int token, int pos, ple_time_fn now);
int    ple_argmax(const float *logits, int n);
int    ple_sample(const float *logits, int n, float temperature, uint32_t *rng);
