#pragma once
#include <stddef.h>
#include <stdint.h>

// Two-bank arena: the D0WD's free DRAM is split into a ~172 KB main bank and a
// ~111 KB D/IRAM bank. We take the largest block from each at boot (minus a
// small margin for the IDF) and hand out 16-byte aligned chunks from whichever
// bank fits. Nothing else mallocs during inference.
#define ARENA_BANKS 2

void   arena_init(size_t margin_bytes);    // margin left free in each bank
void  *arena_alloc(size_t bytes);          // aborts on exhaustion
void  *arena_alloc_bank(int bank, size_t bytes); // force a bank (e.g. hot data in bank 0)
size_t arena_capacity(int bank);
size_t arena_used(int bank);
size_t arena_remaining(int bank);
size_t arena_largest_free(void);
void   arena_reset(void);
void   arena_report(void);
