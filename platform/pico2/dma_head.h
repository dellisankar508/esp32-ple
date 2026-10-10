#pragma once
#include <stddef.h>
#include "ple.h"
// DMA mover for core/head_stream.c on the RP2350.
//
// Two ways to get the head rows from flash into SRAM:
//   stream  the XIP streaming engine reads one linear range into a FIFO in the
//           background; DMA drains it (via the XIP_AUX port) into the buffers,
//           paced by DREQ_XIP_STREAM. Made for exactly this job.
//   direct  plain DMA reads from the uncached flash alias (0x14000000), unpaced.
// dma_head_pick() times both moving the whole head, checks the bytes, and keeps the
// faster one.

void  dma_head_init(void);       // on core 0: claims a channel, DMA_IRQ_0, a spin lock
// Measures both modes, keeps the faster correct one, writes a one-line description.
// Returns 0 when at least one mode works.
int   dma_head_pick(const ple_model_t *m, char *desc, size_t n);
float dma_head_move_ms(const ple_model_t *m);    // move-only time, current mode
