#include "bench.h"
#include "kern.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCH_MAX_REPS 7
#define LONG_FROM 100              // long-context window: positions [LONG_FROM, LONG_TO)
#define LONG_TO   120

static uint32_t t_now(ple_time_fn f) { return f ? f() : 0; }

uint32_t bench_fnv1a(const uint8_t *p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static int parse_line(const char **p, int *out, int max) {
    int n = 0;
    while (**p && **p != '\n' && **p != '\r' && n < max) {
        char *end;
        long v = strtol(*p, &end, 10);
        if (end == *p) break;
        out[n++] = (int)v;
        *p = end;
        while (**p == ' ' || **p == '\t') (*p)++;
    }
    while (**p == '\r' || **p == '\n') (*p)++;
    return n;
}

int bench_parse_golden(const char *txt, bench_golden_t *g) {
    const char *p = txt;
    g->np = parse_line(&p, g->prompt, BENCH_MAX_PROMPT);
    g->nw = parse_line(&p, g->want, BENCH_MAX_GOLDEN);
    return (g->np > 0 && g->nw > 0) ? 0 : -1;
}

// ------------------------------------------------------------------ flash
bench_flash_t bench_flash(const uint8_t *base, size_t size, ple_time_fn now) {
    bench_flash_t f = {0, 0};
    if (!now || size < 512 * 1024) return f;
    volatile uint32_t sink = 0;
    // Sequential: 256 KB, well past any MCU cache, same as the ESP32 Phase 0 test.
    const size_t seq = 256 * 1024;
    const uint32_t *w = (const uint32_t *)(base + 128 * 1024);   // skip header region
    uint32_t acc = 0;                    // local accumulator: a volatile store per word
    uint32_t t0 = now();                 // would add loop cost and under-report bandwidth
    for (size_t i = 0; i < seq / 4; i++) acc += w[i];
    uint32_t dt = now() - t0;
    sink += acc;
    f.seq_mbps = dt ? (float)seq / dt : 0;                        // bytes/us = MB/s
    // Random 216-byte rows (one token's PLE row), 600 of them.
    const size_t row = 216, nrows = (size - row) / row;
    uint32_t seed = 12345;
    t0 = now();
    for (int r = 0; r < 600; r++) {
        seed = seed * 1103515245u + 12345u;
        const uint8_t *p = base + ((seed >> 8) % nrows) * row;
        for (size_t i = 0; i < row; i++) acc += p[i];
    }
    dt = now() - t0;
    sink += acc;
    f.row_us = dt / 600.0f;
    (void)sink;
    return f;
}

// Two 128 KB segments, read sequentially: by one core back to back, or one per core
// at the same time. Equal bandwidth means the cores share one serial flash path.
#define F2_SEG (128 * 1024)
typedef struct { const uint8_t *base; volatile uint32_t sink[2]; } f2_t;

static void f2_job(void *a, int r0, int r1) {
    f2_t *j = a;
    uint32_t acc = 0;
    for (int r = r0; r < r1; r++) {
        const uint32_t *w = (const uint32_t *)(j->base + (size_t)r * F2_SEG);
        for (size_t i = 0; i < F2_SEG / 4; i++) acc += w[i];
    }
    j->sink[r0 & 1] += acc;
}

static void flash2(const bench_platform_t *p, const ple_model_t *m, float *one, float *two) {
    *one = *two = 0;
    if (!p->now || !p->parallel || m->size < 1536 * 1024) return;
    f2_t j;
    memset(&j, 0, sizeof j);
    j.base = m->base + 512 * 1024;                 // fresh regions, not cached
    uint32_t t0 = p->now();
    f2_job(&j, 0, 2);
    uint32_t dt = p->now() - t0;
    *one = dt ? 2.0f * F2_SEG / dt : 0;
    j.base = m->base + 1024 * 1024;                // different fresh regions
    t0 = p->now();
    p->parallel(f2_job, &j, 2);
    dt = p->now() - t0;
    *two = dt ? 2.0f * F2_SEG / dt : 0;
}

// ------------------------------------------------------------------ golden + timing
bench_run_t bench_golden(const ple_model_t *m, ple_state_t *s, const tokenizer_t *tk,
                         const bench_golden_t *g, ple_time_fn now) {
    bench_run_t r;
    memset(&r, 0, sizeof r);
    uint64_t te = 0, ta = 0, tf = 0, th = 0;
    int pos = 0, tok = g->prompt[0];
    if (tk) { int len; const uint8_t *b = tokenizer_bytes(tk, tok, &len); fwrite(b, 1, len, stdout); }
    uint32_t t0 = t_now(now);
    for (int i = 0; i < g->np + g->nw - 1 && pos < s->ctx - 1; i++) {
        ple_forward(m, s, tok, pos, now);
        te += s->t_embed; ta += s->t_attn; tf += s->t_ffn; th += s->t_head;
        int nxt = (i + 1 < g->np) ? g->prompt[i + 1] : ple_argmax(s->logits, m->V);
        if (i + 1 >= g->np) {
            int k = i + 1 - g->np, w = g->want[k];
            if (nxt != w) {
                float gap = s->logits[nxt] - s->logits[w];
                if (gap > BENCH_TIE_GAP) {
                    printf("\n[mismatch at gen %d: got %d want %d gap %.4f]", k, nxt, w, gap);
                    r.mismatches++;
                } else r.ties++;
                nxt = w;                         // stay on the golden path
            }
        }
        if (tk) { int len; const uint8_t *b = tokenizer_bytes(tk, nxt, &len); fwrite(b, 1, len, stdout); fflush(stdout); }
        tok = nxt; pos++;
    }
    uint32_t dt = t_now(now) - t0;
    if (tk) printf("\n");
    r.tokens = pos;
    r.reps = 1;
    if (pos && now) {
        r.ms_per_token = r.ms_min = r.ms_max = dt / 1000.0f / pos;
        r.tok_s = pos * 1e6f / dt;
        r.embed_ms = te / 1000.0f / pos; r.attn_ms = ta / 1000.0f / pos;
        r.ffn_ms = tf / 1000.0f / pos;   r.head_ms = th / 1000.0f / pos;
    }
    return r;
}

// reps silent runs; returns the median run, with min/max and every run's mismatches.
static bench_run_t repeat(const ple_model_t *m, ple_state_t *s, const bench_golden_t *g,
                          ple_time_fn now, int reps) {
    bench_run_t runs[BENCH_MAX_REPS];
    if (reps < 1) reps = 1;
    if (reps > BENCH_MAX_REPS) reps = BENCH_MAX_REPS;
    int mism = 0, ties = 0;
    for (int i = 0; i < reps; i++) {
        runs[i] = bench_golden(m, s, NULL, g, now);
        mism += runs[i].mismatches; ties += runs[i].ties;
    }
    for (int i = 1; i < reps; i++)                       // insertion sort by ms/token
        for (int k = i; k > 0 && runs[k].ms_per_token < runs[k - 1].ms_per_token; k--) {
            bench_run_t t = runs[k]; runs[k] = runs[k - 1]; runs[k - 1] = t;
        }
    bench_run_t r = runs[reps / 2];
    r.ms_min = runs[0].ms_per_token;
    r.ms_max = runs[reps - 1].ms_per_token;
    r.mismatches = mism; r.ties = ties; r.reps = reps;
    return r;
}

static void report(const bench_platform_t *p, const char *cfg, const bench_run_t *r) {
    printf("GOLDEN: %s (%d mismatches, %d ties) | %s %s | %d runs\n",
           r->mismatches ? "FAIL" : "MATCH", r->mismatches, r->ties, p->board, cfg, r->reps);
    printf("RESULT | %s %s | kernel %s | %d tok | %.1f ms/tok | %.2f tok/s | "
           "embed %.2f attn %.2f ffn %.2f head %.2f ms | median of %d, range %.1f-%.1f ms | %.2f Mcycles/tok\n",
           p->board, cfg, kern_name(), r->tokens, r->ms_per_token, r->tok_s,
           r->embed_ms, r->attn_ms, r->ffn_ms, r->head_ms,
           r->reps, r->ms_min, r->ms_max, r->ms_per_token * p->cpu_mhz / 1000.0f);
}

// ------------------------------------------------------------------ head from SRAM
// Copies a slice of head rows into the logits buffer (free between tokens) and runs
// the same kernel over it, scaled to the full head. No flash in the loop, so this is
// the pure compute cost, and 1 vs 2 cores shows how well that compute splits.
typedef struct { const uint8_t *w; uint32_t in; const float *x; float *y; } mv_t;
static void mv_job(void *a, int r0, int r1) {
    mv_t *j = a;
    kern_matvec(j->w, j->in, j->x, j->y, r0, r1);
}

static float head_sram_ms(const ple_model_t *m, ple_state_t *s, ple_parallel_fn par, ple_time_fn now) {
    const uint32_t rb = kern_row_bytes(m->D);
    const size_t cap = (size_t)m->V * 4;
    int rows = 512;
    while (rows > 16 && (size_t)rows * rb + 16 + (size_t)rows * 4 > cap) rows /= 2;
    uint8_t *buf = (uint8_t *)s->logits;
    memcpy(buf, m->tok, (size_t)rows * rb);
    mv_t j = { buf, m->D, s->x, (float *)(buf + (((size_t)rows * rb + 15) & ~(size_t)15)) };
    const int reps = (int)(m->V / rows) ? (int)(m->V / rows) : 1;
    float best = 0;
    for (int trial = 0; trial < 3; trial++) {           // best of 3: compute only, so take the min
        uint32_t t0 = now();
        for (int k = 0; k < reps; k++) {
            kern_prepare(s->x, m->D);
            if (par) par(mv_job, &j, rows);
            else kern_matvec(j.w, j.in, j.x, j.y, 0, rows);
        }
        float ms = (now() - t0) / 1000.0f * (float)m->V / ((float)reps * rows);
        if (trial == 0 || ms < best) best = ms;
    }
    return best;
}

// ------------------------------------------------------------------ long context
static void long_ctx(const bench_platform_t *p, const ple_model_t *m, ple_state_t *s,
                     const bench_golden_t *g, const char *cfg) {
    int to = LONG_TO < s->ctx - 1 ? LONG_TO : s->ctx - 1;
    if (to <= LONG_FROM) return;
    uint64_t ta = 0;
    uint32_t t0 = 0;
    int tok = g->prompt[0];
    for (int pos = 0; pos < to; pos++) {
        if (pos == LONG_FROM) t0 = p->now();
        ple_forward(m, s, tok, pos, p->now);
        if (pos >= LONG_FROM) ta += s->t_attn;
        tok = (pos + 1 < g->np) ? g->prompt[pos + 1] : ple_argmax(s->logits, m->V);
    }
    uint32_t dt = p->now() - t0;
    int n = to - LONG_FROM;
    printf("LONGCTX | %s %s | pos %d-%d | %.1f ms/tok | %.2f tok/s | attn %.2f ms\n",
           p->board, cfg, LONG_FROM, to - 1,
           dt / 1000.0f / n, n * 1e6f / dt, ta / 1000.0f / n);
}

// ------------------------------------------------------------------ everything
int bench_run_all(const bench_platform_t *p, const ple_model_t *m, ple_state_t *s,
                   const tokenizer_t *tk, const bench_golden_t *g) {
    size_t st[4];
    ple_stage_bytes(m, st);
    printf("\n=== PLE bench ===\n");
    printf("ENV | board %s | cpu %.0f MHz | flash %s | compiler gcc %s %s | kernel %s | ctx %d\n",
           p->board, p->cpu_mhz, p->flash_desc, __VERSION__, p->cflags, kern_name(), s->ctx);
    printf("MODEL | fnv1a 0x%08x | %u bytes | state %u B | %u B/tok\n",
           (unsigned)bench_fnv1a(m->base, m->size), (unsigned)m->size,
           (unsigned)ple_state_bytes(m, s->ctx), (unsigned)ple_bytes_per_token(m));

    bench_flash_t fl = bench_flash(m->base, m->size, p->now);
    printf("FLASH | sequential %.1f MB/s | random 216 B row %.1f us\n", fl.seq_mbps, fl.row_us);
    if (p->parallel) {
        float one, two;
        flash2(p, m, &one, &two);
        printf("FLASH2 | 1 core %.1f MB/s | 2 cores together %.1f MB/s | %.2fx\n",
               one, two, one > 0 ? two / one : 0);
    }
    float bw = fl.seq_mbps > 0 ? fl.seq_mbps : 1;
    printf("FLOOR | embed %.2f attn %.2f ffn %.2f head %.2f ms | total %.1f ms at %.1f MB/s\n",
           st[0] / bw / 1000.0f, st[1] / bw / 1000.0f, st[2] / bw / 1000.0f, st[3] / bw / 1000.0f,
           ple_bytes_per_token(m) / bw / 1000.0f, fl.seq_mbps);

    // Warm-up, timing discarded; prints the golden text once.
    s->parallel = NULL;
    s->head = NULL;
    bench_run_t w = bench_golden(m, s, tk, g, p->now);
    if (w.mismatches) printf("[warm-up run had %d mismatches]\n", w.mismatches);

    bench_run_t r1 = repeat(m, s, g, p->now, p->reps);
    report(p, "1-core", &r1);
    bench_run_t r2;
    memset(&r2, 0, sizeof r2);
    if (p->parallel) {
        s->parallel = p->parallel;
        r2 = repeat(m, s, g, p->now, p->reps);
        report(p, "2-core", &r2);
    }

    float h1 = head_sram_ms(m, s, NULL, p->now), h2 = 0;
    if (p->parallel) {
        h2 = head_sram_ms(m, s, p->parallel, p->now);
        printf("HEADSRAM | %s | from SRAM: 1 core %.2f ms, 2 cores %.2f ms, %.2fx | "
               "from flash: 1 core %.2f ms, 2 cores %.2f ms, %.2fx\n",
               p->board, h1, h2, h2 > 0 ? h1 / h2 : 0,
               r1.head_ms, r2.head_ms, r2.head_ms > 0 ? r1.head_ms / r2.head_ms : 0);
    } else {
        printf("HEADSRAM | %s | from SRAM: 1 core %.2f ms | from flash: 1 core %.2f ms\n",
               p->board, h1, r1.head_ms);
    }

    // Head moved by the platform (e.g. DMA into SRAM): same golden check, 1 and 2 cores.
    bench_run_t r3, r4;
    memset(&r3, 0, sizeof r3);
    memset(&r4, 0, sizeof r4);
    const char *hn = p->head_name ? p->head_name : "head";
    char c1[32], c2[32];
    snprintf(c1, sizeof c1, "1-core+%s", hn);
    snprintf(c2, sizeof c2, "2-core+%s", hn);
    if (p->head) {
        s->head = p->head;
        s->parallel = NULL;
        r3 = repeat(m, s, g, p->now, p->reps);
        report(p, c1, &r3);
        if (p->parallel) {
            s->parallel = p->parallel;
            r4 = repeat(m, s, g, p->now, p->reps);
            report(p, c2, &r4);
        }
        s->head = NULL;
        float mv = p->head_move_ms ? p->head_move_ms(m) : 0;
        float head_bytes = (float)(st[3] - 4 * m->D);          // head rows, without the norm
        printf("HEADSTREAM | %s %s | move only %.2f ms (%.1f MB/s) | 1 core %.2f ms, 2 cores %.2f ms | "
               "compute only %.2f / %.2f ms | %s\n",
               p->board, hn, mv, mv > 0 ? head_bytes / (mv * 1000.0f) : 0,
               r3.head_ms, r4.head_ms, h1, h2, p->head_desc ? p->head_desc : "");
    }

    s->parallel = p->parallel;
    s->head = p->head;
    long_ctx(p, m, s, g, p->head ? (p->parallel ? c2 : c1) : (p->parallel ? "2-core" : "1-core"));
    printf("=== bench done ===\n");
    return w.mismatches + r1.mismatches + r2.mismatches + r3.mismatches + r4.mismatches;
}

// ------------------------------------------------------------------ story
int bench_story(const ple_model_t *m, ple_state_t *s, const tokenizer_t *tk,
                const char *text, int max_tokens, int min_tokens, uint32_t seed,
                void (*emit)(const uint8_t *, int)) {
    int prompt[BENCH_MAX_PROMPT];
    int np = tokenizer_encode(tk, text, prompt, BENCH_MAX_PROMPT);
    if (np <= 0) return 0;
    uint32_t rng = seed;
    int pos = 0, tok = prompt[0], len;
    const uint8_t *b = tokenizer_bytes(tk, tok, &len); emit(b, len);
    for (int i = 0; pos < s->ctx - 1 && pos < max_tokens; i++) {
        ple_forward(m, s, tok, pos, NULL);
        int nxt = (i + 1 < np) ? prompt[i + 1] : ple_sample(s->logits, m->V, 0.8f, &rng);
        if (nxt == 0) break;                                 // <eos>
        b = tokenizer_bytes(tk, nxt, &len); emit(b, len);
        tok = nxt; pos++;
        if (pos > min_tokens && len && (b[len - 1] == '.' || b[len - 1] == '!' || b[len - 1] == '?')) break;
    }
    return pos;
}
