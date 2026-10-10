#pragma once
#include <stdint.h>
#include "ple.h"
// Streamed output head.
//
// The head is the biggest read of every token (459 KB of the 741 KB). Read straight
// from XIP flash, the cores stall on flash, and on the RP2350 two cores reading at
// once share one flash path (FLASH2 0.37x), so the second core adds little. Here a
// mover (DMA on the Pico) copies the head rows block by block into a ring of SRAM
// buffers while the cores compute on blocks that have already arrived. Flash time
// and math overlap, and the cores never touch flash during the head.
//
// Protocol:
//   - one mover, one block in flight, blocks moved in order (so the Pico can use the
//     XIP stream FIFO, which reads one linear range)
//   - block b lands in buffer b % nbuf; it starts once every core has finished block
//     b - nbuf, the previous block in that buffer
//   - each block's rows are split in half between the two cores
//   - the mover calls hs_block_arrived() when a block is in SRAM (the DMA IRQ on the
//     Pico); that starts the next block if its buffer is free. Each core runs the same
//     check after finishing a block, because finishing may free a buffer.
//   - `par` must run both halves at the same time (two cores or two threads): each
//     core waits for blocks that the other core's progress unlocks.

typedef struct {
    // Optional: the whole range [src, src + bytes) is about to be moved, block by block.
    void (*begin)(const uint8_t *src, uint32_t bytes);
    // Start moving one block. Must not wait for it; call hs_block_arrived() once the
    // bytes are in dst. src is the block's address in the weight image.
    void (*start)(const uint8_t *src, uint8_t *dst, uint32_t bytes);
    // Mutual exclusion between both cores and the completion callback. On the Pico a
    // spin lock that also masks interrupts on the calling core.
    uint32_t (*lock)(void);
    void (*unlock)(uint32_t saved);
    void (*relax)(void);           // optional, called while a core waits (host: sched_yield)
    void (*abort)(void);           // optional: stop a block that will never arrive
} hs_mover_t;

// bufs: nbuf buffers of buf_bytes each, 4-byte aligned, nbuf >= 2. Returns 0 on success.
int   hs_init(const hs_mover_t *mv, uint8_t *bufs, uint32_t buf_bytes, int nbuf);

// A ple_head_fn: set s->head = hs_head.
void  hs_head(ple_parallel_fn par, const uint8_t *w, uint32_t rows, uint32_t in,
              const float *x, float *y);

// Called by the mover when the block it was started on is in SRAM.
void  hs_block_arrived(void);

// Moves the head with no math, on one core. Returns milliseconds, or -1 if the
// blocks left in the buffers differ from the source or a block took longer than
// 200 ms (the mover is broken). Run this once before using hs_head: the head path
// itself has no timeout.
float hs_move_ms(const uint8_t *w, uint32_t rows, uint32_t in, ple_time_fn now);

// Rows per block for a given input width (0 if hs_init was not called).
uint32_t hs_rows_per_block(uint32_t in);
