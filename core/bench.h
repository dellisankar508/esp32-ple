#pragma once
#include <stddef.h>
#include <stdint.h>
#include "ple.h"
#include "tokenizer.h"
// One measurement method for every target (ESP32, Pico 2 W, FVP, host), so the
// results compare like with like. Every platform fills a bench_platform_t and calls
// bench_run_all(); tools/compare_logs.py turns the tagged lines into one table.
//
// Tagged output lines (one per measurement):
//   ENV      board, measured CPU clock, flash mode/clock, compiler, flags, kernel, ctx
//   MODEL    checksum, size, state RAM, bytes per token
//   FLASH    1-core sequential MB/s, random 216 B row us
//   FLASH2   same bytes read by 1 core vs both cores at once (shared flash path?)
//   FLOOR    per-stage flash floor: stage bytes / sequential bandwidth
//   GOLDEN   correctness of the greedy decode
//   RESULT   timing per config (median of reps), with the per-stage split
//   HEADSRAM head matvec with weights copied to SRAM: pure compute, 1 vs 2 cores
//   HEADSTREAM (platforms with a head hook) head moved by the platform, e.g. DMA:
//            move-only time and MB/s, head time on 1 and 2 cores, next to compute only
//   LONGCTX  timing at positions 100..119 (attention grows with context), fastest config

#define BENCH_MAX_PROMPT 16
#define BENCH_MAX_GOLDEN 64
#define BENCH_TIE_GAP 0.05f        // logit gap below which a mismatch is float ordering

typedef struct {
    int prompt[BENCH_MAX_PROMPT], np;
    int want[BENCH_MAX_GOLDEN], nw;
} bench_golden_t;

typedef struct {
    float seq_mbps;                // sequential read bandwidth, MB/s (1e6 B/s)
    float row_us;                  // random 216-byte row read, microseconds
} bench_flash_t;

typedef struct {
    int tokens, mismatches, ties, reps;
    float ms_per_token, tok_s, ms_min, ms_max;
    float embed_ms, attn_ms, ffn_ms, head_ms;   // averaged over all tokens
} bench_run_t;

typedef struct {
    const char *board;             // short id, e.g. "esp32" or "pico2w"
    float cpu_mhz;                 // as measured or reported by the clock driver
    const char *flash_desc;        // e.g. "QIO 80 MHz"
    const char *cflags;            // e.g. "-O2"
    ple_time_fn now;               // microsecond clock
    ple_parallel_fn parallel;      // NULL on single-core targets
    int reps;                      // timing repeats per config; the median is reported
    // Optional head hook (core/head_stream.c): runs two more configs, "1-core+<name>"
    // and "2-core+<name>", and the HEADSTREAM line. Leave NULL to skip.
    ple_head_fn head;
    const char *head_name;         // e.g. "dma"
    const char *head_desc;         // free text for HEADSTREAM, e.g. mode and buffers
    float (*head_move_ms)(const ple_model_t *m);   // optional: move the head only, no math
} bench_platform_t;

// FNV-1a over the model image: printed on every target so logs prove the same
// model.bin ran everywhere.
uint32_t bench_fnv1a(const uint8_t *p, size_t n);

// Parse golden.txt: line 1 prompt ids, line 2 expected greedy ids. Tolerates \r\n.
int  bench_parse_golden(const char *txt, bench_golden_t *g);

// The full benchmark. Leaves s->parallel = p->parallel and s->head = p->head (the
// fastest setup, for the story that follows). Returns the number of golden
// mismatches over every run (0 = all runs matched).
int  bench_run_all(const bench_platform_t *p, const ple_model_t *m, ple_state_t *s,
                   const tokenizer_t *tk, const bench_golden_t *g);

// Building blocks, also usable on their own.
bench_flash_t bench_flash(const uint8_t *base, size_t size, ple_time_fn now);
bench_run_t   bench_golden(const ple_model_t *m, ple_state_t *s, const tokenizer_t *tk,
                           const bench_golden_t *g, ple_time_fn now);

// Sampled story at temperature 0.8, stops at <eos> or a sentence end after min_tokens.
int  bench_story(const ple_model_t *m, ple_state_t *s, const tokenizer_t *tk,
                 const char *prompt, int max_tokens, int min_tokens, uint32_t seed,
                 void (*emit)(const uint8_t *bytes, int len));
