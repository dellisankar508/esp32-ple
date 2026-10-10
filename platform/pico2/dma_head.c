// DMA mover for the streamed output head (core/head_stream.c), RP2350.
#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"
#include "hardware/dma.h"
#include "hardware/irq.h"
#include "hardware/sync.h"
#include "hardware/structs/xip_ctrl.h"
#include "hardware/regs/addressmap.h"
#include "head_stream.h"
#include "kern.h"
#include "dma_head.h"

// 6 x 8 KB: 146 head rows per block (56 B each), 57 blocks for the 8192-row head.
// Small blocks keep the pipeline fill (first block) and drain (last block) short.
#ifndef DH_BUF_BYTES
#define DH_BUF_BYTES (8 * 1024)
#endif
#ifndef DH_NBUF
#define DH_NBUF 6
#endif

typedef enum { DH_STREAM = 0, DH_DIRECT = 1 } dh_mode_t;
static const char *const dh_names[] = { "stream", "direct" };

static uint8_t dh_buf[DH_NBUF * DH_BUF_BYTES] __attribute__((aligned(16)));
static int dh_ch = -1;
static spin_lock_t *dh_lock;
static dh_mode_t dh_mode = DH_STREAM;
static dma_channel_config dh_cfg[2];

static uint32_t now_us(void) { return time_us_32(); }

// ------------------------------------------------------------ mover callbacks
static void dh_begin(const uint8_t *src, uint32_t bytes) {
    if (dh_mode != DH_STREAM) return;
    xip_ctrl_hw->stream_ctr = 0;                              // halt any old stream
    while (!(xip_ctrl_hw->stat & XIP_STAT_FIFO_EMPTY_BITS))   // and drain its FIFO
        (void)xip_ctrl_hw->stream_fifo;
    xip_ctrl_hw->stream_addr = (uint32_t)src;
    xip_ctrl_hw->stream_ctr = bytes / 4;                      // starts reading now
}

static void dh_start(const uint8_t *src, uint8_t *dst, uint32_t bytes) {
    const volatile void *from = dh_mode == DH_STREAM
        ? (const volatile void *)XIP_AUX_BASE                 // stream FIFO, fast port
        : (const volatile void *)((uint32_t)src - XIP_BASE + XIP_NOCACHE_NOALLOC_BASE);
    dma_channel_configure((uint)dh_ch, &dh_cfg[dh_mode], dst, from, bytes / 4, true);
}

static uint32_t dh_lock_fn(void) { return spin_lock_blocking(dh_lock); }
static void dh_unlock_fn(uint32_t saved) { spin_unlock(dh_lock, saved); }

static void dh_abort(void) {
    dma_irqn_set_channel_enabled(0, (uint)dh_ch, false);    // no completion IRQ from the abort
    dma_channel_abort((uint)dh_ch);
    dma_irqn_acknowledge_channel(0, (uint)dh_ch);
    dma_irqn_set_channel_enabled(0, (uint)dh_ch, true);
    xip_ctrl_hw->stream_ctr = 0;
    while (!(xip_ctrl_hw->stat & XIP_STAT_FIFO_EMPTY_BITS))
        (void)xip_ctrl_hw->stream_fifo;
}

static const hs_mover_t dh_mover = {
    .begin = dh_begin, .start = dh_start, .lock = dh_lock_fn, .unlock = dh_unlock_fn,
    .abort = dh_abort,
};

static void dh_irq(void) {
    if (!dma_irqn_get_channel_status(0, (uint)dh_ch)) return;
    dma_irqn_acknowledge_channel(0, (uint)dh_ch);
    hs_block_arrived();
}

// ------------------------------------------------------------ setup
void dma_head_init(void) {
    dh_ch = dma_claim_unused_channel(true);
    dh_lock = spin_lock_init((uint)spin_lock_claim_unused(true));

    dma_channel_config c = dma_channel_get_default_config((uint)dh_ch);   // 32-bit, unpaced
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
    channel_config_set_write_increment(&c, true);
    channel_config_set_read_increment(&c, false);                       // stream: one FIFO address
    channel_config_set_dreq(&c, DREQ_XIP_STREAM);
    dh_cfg[DH_STREAM] = c;
    c = dma_channel_get_default_config((uint)dh_ch);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
    channel_config_set_write_increment(&c, true);
    channel_config_set_read_increment(&c, true);                        // direct: walk the alias
    dh_cfg[DH_DIRECT] = c;

    dma_irqn_set_channel_enabled(0, (uint)dh_ch, true);
    irq_add_shared_handler(DMA_IRQ_0, dh_irq, PICO_SHARED_IRQ_HANDLER_DEFAULT_ORDER_PRIORITY);
    irq_set_enabled(DMA_IRQ_0, true);

    hs_init(&dh_mover, dh_buf, DH_BUF_BYTES, DH_NBUF);
}

float dma_head_move_ms(const ple_model_t *m) {
    return hs_move_ms(m->tok, m->V, m->D, now_us);
}

int dma_head_pick(const ple_model_t *m, char *desc, size_t n) {
    float ms[2];
    const float bytes = (float)m->V * kern_row_bytes(m->D);     // head rows
    for (int k = 0; k < 2; k++) {
        dh_mode = (dh_mode_t)k;
        ms[k] = -1;
        for (int t = 0; t < 3; t++) {                            // best of 3
            float v = dma_head_move_ms(m);
            if (v < 0) { ms[k] = -1; break; }                    // wrong bytes: mode unusable
            if (ms[k] < 0 || v < ms[k]) ms[k] = v;
        }
    }
    int ok0 = ms[0] > 0, ok1 = ms[1] > 0;
    dh_mode = (ok0 && (!ok1 || ms[0] <= ms[1])) ? DH_STREAM : DH_DIRECT;
    char a[24], b[24];
    if (ok0) snprintf(a, sizeof a, "%.1f MB/s", bytes / (ms[0] * 1000.0f)); else snprintf(a, sizeof a, "BAD");
    if (ok1) snprintf(b, sizeof b, "%.1f MB/s", bytes / (ms[1] * 1000.0f)); else snprintf(b, sizeof b, "BAD");
    snprintf(desc, n, "mode %s; stream %s, direct %s; %d x %u B buffers, %u rows/block",
             (ok0 || ok1) ? dh_names[dh_mode] : "none", a, b, DH_NBUF, (unsigned)DH_BUF_BYTES,
             (unsigned)hs_rows_per_block(m->D));
    return (ok0 || ok1) ? 0 : -1;
}
