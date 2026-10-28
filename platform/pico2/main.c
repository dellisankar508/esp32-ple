// Pico 2 W (RP2350, 2x Cortex-M33 @ 150 MHz).
// Boot: wait for USB serial, then (int8 kernels) check the kernel against the numpy
// test vectors bit for bit, then run the shared benchmark (core/bench.c), the same
// sequence the ESP32 runs, plus the DMA-streamed head (dma_head.c).
// After that: Enter = sampled story, 'b' = rerun, 'd' = DMA head on/off.
#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "pico/multicore.h"
#include "hardware/clocks.h"
#include "hardware/sync.h"
#include "hardware/structs/qmi.h"
#include "ple.h"
#include "tokenizer.h"
#include "bench.h"
#include "kern.h"
#include "head_stream.h"
#include "dma_head.h"
#include "kern_check.h"

#define CTX 128
#ifndef PLE_CFLAGS
#define PLE_CFLAGS "?"
#endif

extern const uint8_t ple_model_start[], ple_model_end[];
extern const uint8_t ple_tok_start[], ple_tok_end[];
extern const char    ple_golden_start[];
#if PLE_KERNEL_I8
extern const uint8_t ple_kvec_start[], ple_kvec_end[];
#endif

// ------------------------------------------------------------ arena (never freed)
static uint8_t arena[272 * 1024] __attribute__((aligned(16)));
static size_t arena_used;
static void *arena_alloc(size_t n) {
    size_t off = (arena_used + 15) & ~(size_t)15;
    if (off + n > sizeof arena) return NULL;
    arena_used = off + n;
    return arena + off;
}

static uint32_t now_us(void) { return time_us_32(); }

// ------------------------------------------------------------ core 1 worker
// Core 1 sleeps on the inter-core FIFO; each job is "run rows [r0, r1)".
static volatile ple_job_fn w_job;
static void *volatile w_arg;
static volatile int w_r0, w_r1;

static void core1_main(void) {
    for (;;) {
        multicore_fifo_pop_blocking();
        w_job(w_arg, w_r0, w_r1);
        __dmb();
        multicore_fifo_push_blocking(1);
    }
}

static void parallel2(ple_job_fn job, void *arg, int n) {
    int mid = n / 2;
    w_job = job; w_arg = arg; w_r0 = mid; w_r1 = n;
    __dmb();
    multicore_fifo_push_blocking(1);   // core 1: rows [mid, n)
    job(arg, 0, mid);                  // core 0: rows [0, mid)
    multicore_fifo_pop_blocking();
}

// ------------------------------------------------------------ model
static ple_model_t m;
static ple_state_t s;
static tokenizer_t tk;
static bench_golden_t golden;

static void emit(const uint8_t *b, int len) { fwrite(b, 1, len, stdout); fflush(stdout); }

// Flash interface as the boot ROM / SDK left it: data width, read command, SCK divider.
static void flash_desc(char *buf, size_t n) {
    uint32_t div = (qmi_hw->m[0].timing & QMI_M0_TIMING_CLKDIV_BITS) >> QMI_M0_TIMING_CLKDIV_LSB;
    uint32_t w = (qmi_hw->m[0].rfmt & QMI_M0_RFMT_DATA_WIDTH_BITS) >> QMI_M0_RFMT_DATA_WIDTH_LSB;
    uint32_t cmd = (qmi_hw->m[0].rcmd & QMI_M0_RCMD_PREFIX_BITS) >> QMI_M0_RCMD_PREFIX_LSB;
    if (div == 0) div = 256;
    snprintf(buf, n, "%s, cmd 0x%02x, SCK %.1f MHz (sys/%u)",
             w == 2 ? "QSPI quad" : w == 1 ? "QSPI dual" : "SPI single", (unsigned)cmd,
             clock_get_hz(clk_sys) / 1e6f / div, (unsigned)div);
}

static ple_head_fn dma_head;       // hs_head once a DMA mode has been checked, else NULL

static void run_bench(void) {
    static char fdesc[64], hdesc[128];
    flash_desc(fdesc, sizeof fdesc);
    // Time both DMA modes moving the whole head and check the bytes; keep the faster.
    dma_head = dma_head_pick(&m, hdesc, sizeof hdesc) == 0 ? hs_head : NULL;
    if (!dma_head) printf("DMA head disabled: %s\n", hdesc);
    bench_platform_t plat = {
        .board = "pico2w",
        .cpu_mhz = clock_get_hz(clk_sys) / 1e6f,
        .flash_desc = fdesc,
        .cflags = PLE_CFLAGS,
        .now = now_us,
        .parallel = parallel2,
        .reps = 3,
        .head = dma_head,
        .head_name = "dma",
        .head_desc = hdesc,
        .head_move_ms = dma_head_move_ms,
    };
    bench_run_all(&plat, &m, &s, &tk, &golden);
}

int main(void) {
    stdio_init_all();
    while (!stdio_usb_connected()) sleep_ms(100);      // so the first lines are not lost
    sleep_ms(500);

    size_t msize = (size_t)(ple_model_end - ple_model_start);
    int rc = ple_load(&m, ple_model_start, msize);
    if (rc) { printf("ple_load failed (%d): is model.bin a PLE1 file?\n", rc); for (;;) sleep_ms(1000); }
    if (tokenizer_load(&tk, ple_tok_start, (size_t)(ple_tok_end - ple_tok_start), arena_alloc)) {
        printf("tokenizer load failed\n"); for (;;) sleep_ms(1000);
    }
#if PLE_KERNEL_I8
    {   // kernel bit-exactness on the chip, in arena space the state has not claimed yet
        size_t off = (arena_used + 15) & ~(size_t)15, need = 2 * sizeof(float) * m.V;
        if (off + need > sizeof arena) printf("KERNEL | no room to check (%u B)\n", (unsigned)need);
        else kern_check(ple_model_start, msize, ple_kvec_start,
                        (size_t)(ple_kvec_end - ple_kvec_start), (float *)(arena + off));
    }
#endif
    if (ple_state_init(&s, &m, CTX, arena_alloc)) {
        printf("state alloc failed: need %u B, arena %u B\n",
               (unsigned)ple_state_bytes(&m, CTX), (unsigned)sizeof arena);
        for (;;) sleep_ms(1000);
    }
    if (bench_parse_golden(ple_golden_start, &golden)) { printf("bad golden file\n"); for (;;) sleep_ms(1000); }
    printf("model V=%u D=%u L=%u H=%u F=%u P=%u | state %u B at ctx %d | arena %u/%u B | %u B/token\n",
           (unsigned)m.V, (unsigned)m.D, (unsigned)m.L, (unsigned)m.H, (unsigned)m.F, (unsigned)m.P,
           (unsigned)ple_state_bytes(&m, CTX), CTX, (unsigned)arena_used, (unsigned)sizeof arena,
           (unsigned)ple_bytes_per_token(&m));

    multicore_launch_core1(core1_main);
    dma_head_init();                    // core 0: the DMA IRQ is handled here
    run_bench();

    uint32_t seed = time_us_32();
    for (;;) {
        printf("\n[Enter] story   [b] rerun benchmark   [d] DMA head: %s\n",
               s.head ? "on" : dma_head ? "off" : "unavailable");
        int c;
        do { c = getchar_timeout_us(1000000); } while (c == PICO_ERROR_TIMEOUT);
        if (c == 'b' || c == 'B') { run_bench(); continue; }
        if (c == 'd' || c == 'D') { s.head = s.head ? NULL : dma_head; continue; }
        s.parallel = parallel2;
        printf("\n--- ");
        uint32_t t0 = time_us_32();
        int n = bench_story(&m, &s, &tk, "Once upon a time", 120, 90, seed += 7919, emit);
        uint32_t dt = time_us_32() - t0;
        printf("\n[%d tokens, %.2f tok/s]\n", n, n * 1e6f / dt);
    }
}
