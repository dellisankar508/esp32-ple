#pragma once
#include <stddef.h>
#include <stdint.h>
// Kernel bit-exactness check against tools/kernel_vectors.py (kvec.bin).
// Runs the linked kernel on every case twice, once over all rows and once split in
// two row ranges (as two cores would), and compares every output float bit for bit
// with the numpy answer. Prints one KCHECK line per failing case and a summary:
//   KERNEL | <name> | BITEXACT 42/42 cases | 1123584 outputs | split rows: same
// Returns the number of failing cases (0 = bit-exact), or -1 for a bad kvec file.
// `y` needs room for the largest case's rows (the head: vocab floats), twice.
int kern_check(const uint8_t *model, size_t model_size, const uint8_t *kvec, size_t kvec_size,
               float *y);
