// Host harness: the same core/ and kernels/ code the chips run.
//   ple_host model.bin tokenizer.bin golden.txt           full benchmark
//   ple_host model.bin tokenizer.bin --story "Once upon"  sampled story
// "2 cores" are two threads. The streamed head (core/head_stream.c) runs with a
// fake DMA: a third thread that memcpys each block after a random short delay, in
// tiny buffers, so the hand-off between movers and cores gets exercised hard.
// Host timings only show the code runs; they are not a target.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include "ple.h"
#include "head_stream.h"
#include "tokenizer.h"
#include "bench.h"

#define CTX 128

static uint8_t *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n + 1);
    if (fread(b, 1, *n, f) != *n) { perror(path); exit(1); }
    b[*n] = 0;                                   // so text files can be parsed as strings
    fclose(f);
    return b;
}

static uint32_t now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)(t.tv_sec * 1000000u + t.tv_nsec / 1000);
}

#ifndef PLE_CFLAGS
#define PLE_CFLAGS "?"
#endif

// ------------------------------------------------------------ two "cores"
// A second thread runs the upper half of each job, like core 1 on the chips.
static pthread_mutex_t pm = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t pc = PTHREAD_COND_INITIALIZER;
static ple_job_fn p_job;
static void *p_arg;
static int p_r0, p_r1, p_gen, p_done;

static void *core1(void *unused) {
    (void)unused;
    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&pm);
        while (p_gen == seen) pthread_cond_wait(&pc, &pm);
        seen = p_gen;
        ple_job_fn job = p_job; void *arg = p_arg; int r0 = p_r0, r1 = p_r1;
        pthread_mutex_unlock(&pm);
        job(arg, r0, r1);
        pthread_mutex_lock(&pm);
        p_done = seen;
        pthread_cond_broadcast(&pc);
        pthread_mutex_unlock(&pm);
    }
    return NULL;
}

static void par2(ple_job_fn job, void *arg, int n) {
    pthread_mutex_lock(&pm);
    p_job = job; p_arg = arg; p_r0 = n / 2; p_r1 = n;
    int gen = ++p_gen;
    pthread_cond_broadcast(&pc);
    pthread_mutex_unlock(&pm);
    job(arg, 0, n / 2);
    pthread_mutex_lock(&pm);
    while (p_done != gen) pthread_cond_wait(&pc, &pm);
    pthread_mutex_unlock(&pm);
}

// ------------------------------------------------------------ fake DMA
// Moves blocks strictly in order from a running cursor, like the RP2350's XIP stream
// FIFO, and fails loudly if the pipeline ever asks for anything else.
#define SIM_BUF_BYTES 1024           // 18 head rows per block: hundreds of hand-offs per token
#define SIM_NBUF 3
static uint8_t sim_buf[SIM_NBUF * SIM_BUF_BYTES] __attribute__((aligned(16)));
static pthread_mutex_t dm = PTHREAD_MUTEX_INITIALIZER, hl = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t dc = PTHREAD_COND_INITIALIZER;
static const uint8_t *d_cursor, *d_src;
static uint8_t *d_dst;
static uint32_t d_bytes;
static int d_pending;
static uint32_t d_rng = 1;

static void sim_begin(const uint8_t *src, uint32_t bytes) { (void)bytes; d_cursor = src; }

static void sim_start(const uint8_t *src, uint8_t *dst, uint32_t bytes) {
    if (src != d_cursor) { fprintf(stderr, "fake DMA: block out of order\n"); exit(3); }
    d_cursor += bytes;
    pthread_mutex_lock(&dm);
    if (d_pending) { fprintf(stderr, "fake DMA: start while busy\n"); exit(3); }
    d_src = src; d_dst = dst; d_bytes = bytes; d_pending = 1;
    pthread_cond_signal(&dc);
    pthread_mutex_unlock(&dm);
}

static void *dma_thread(void *unused) {
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&dm);
        while (!d_pending) pthread_cond_wait(&dc, &dm);
        const uint8_t *src = d_src; uint8_t *dst = d_dst; uint32_t n = d_bytes;
        d_pending = 0;
        pthread_mutex_unlock(&dm);
        d_rng = d_rng * 1103515245u + 12345u;
        if ((d_rng >> 16) % 4 == 0) {                 // sometimes slow, sometimes not
            struct timespec t = { 0, (long)((d_rng >> 8) % 20000) };
            nanosleep(&t, NULL);
        }
        memcpy(dst, src, n);
        hs_block_arrived();
    }
    return NULL;
}

static uint32_t sim_lock(void) { pthread_mutex_lock(&hl); return 0; }
static void sim_unlock(uint32_t k) { (void)k; pthread_mutex_unlock(&hl); }
static void sim_relax(void) { sched_yield(); }

static const hs_mover_t sim_mover = {
    .begin = sim_begin, .start = sim_start, .lock = sim_lock, .unlock = sim_unlock, .relax = sim_relax,
};

static float sim_move_ms(const ple_model_t *m) { return hs_move_ms(m->tok, m->V, m->D, now_us); }

static void emit(const uint8_t *b, int len) { fwrite(b, 1, len, stdout); fflush(stdout); }

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.bin tokenizer.bin (golden.txt | --story \"prompt\")\n", argv[0]);
        return 1;
    }
    size_t ms, ts;
    uint8_t *mb = slurp(argv[1], &ms), *tb = slurp(argv[2], &ts);
    ple_model_t m;
    if (ple_load(&m, mb, ms)) { fprintf(stderr, "bad model\n"); return 1; }
    tokenizer_t tk;
    if (tokenizer_load(&tk, tb, ts, malloc)) { fprintf(stderr, "bad tokenizer\n"); return 1; }
    ple_state_t s;
    if (ple_state_init(&s, &m, CTX, malloc)) { fprintf(stderr, "state alloc failed\n"); return 1; }
    pthread_t t1, t2;
    pthread_create(&t1, NULL, core1, NULL);
    pthread_create(&t2, NULL, dma_thread, NULL);
    hs_init(&sim_mover, sim_buf, SIM_BUF_BYTES, SIM_NBUF);
    printf("model V=%u D=%u L=%u H=%u F=%u P=%u | state %u B at ctx %d | %u B/token\n",
           (unsigned)m.V, (unsigned)m.D, (unsigned)m.L, (unsigned)m.H, (unsigned)m.F, (unsigned)m.P,
           (unsigned)ple_state_bytes(&m, CTX), CTX, (unsigned)ple_bytes_per_token(&m));

    if (!strcmp(argv[3], "--story")) {
        const char *prompt = argc > 4 ? argv[4] : "Once upon a time";
        int n = bench_story(&m, &s, &tk, prompt, 120, 90, 42, emit);
        printf("\n[%d tokens]\n", n);
        return 0;
    }

    size_t gs; char *gt = (char *)slurp(argv[3], &gs);
    bench_golden_t g;
    if (bench_parse_golden(gt, &g)) { fprintf(stderr, "bad golden file\n"); return 1; }
    printf("golden: %d prompt + %d tokens\n", g.np, g.nw);

    char hdesc[96];
    snprintf(hdesc, sizeof hdesc, "fake DMA thread; %d x %d B buffers, %u rows/block",
             SIM_NBUF, SIM_BUF_BYTES, (unsigned)hs_rows_per_block(m.D));
    bench_platform_t plat = {
        .board = "host", .cpu_mhz = 0, .flash_desc = "RAM (model in a heap buffer)",
        .cflags = PLE_CFLAGS, .now = now_us, .parallel = par2, .reps = 3,
        .head = hs_head, .head_name = "dmasim", .head_desc = hdesc, .head_move_ms = sim_move_ms,
    };
    return bench_run_all(&plat, &m, &s, &tk, &g) ? 2 : 0;
}
