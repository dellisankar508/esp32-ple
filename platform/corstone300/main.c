// PLE on Arm M-profile, bare metal: Corstone-300 FVP (Cortex-M55 + Helium) and QEMU
// (mps3-an547 for the M55, mps2-an505 for an M33). Output goes through semihosting.
//
// What it checks and measures, all on the target:
//  1. KERNEL  the linked kernel against numpy test vectors, every output bit for bit
//  2. GOLDEN  greedy decode against golden_i8.txt (or golden.txt for the float kernel)
//  3. LOGITS  FNV-1a over every logit of that decode. Two kernels that print the same
//             hash on the same target produced a bit-identical model run.
//  4. PMU     per token and for one head matvec: cycles and Helium instructions from
//             the Armv8.1-M PMU when PLE_PMU=1 (FVP, where cycles = instructions); in
//             QEMU the insncount plugin counts the regions between the ple_mark_*
//             functions instead.
// The FVP and QEMU are not cycle-accurate: instruction counts compare kernels, they are
// not timings.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ple.h"
#include "tokenizer.h"
#include "bench.h"
#include "kern.h"
#include "kern_check.h"

#ifndef PLE_BOARD
#define PLE_BOARD "armv8m"
#endif
#define CTX 128

extern const uint8_t ple_model_start[], ple_model_end[], ple_kvec_start[], ple_kvec_end[];
extern const uint8_t ple_tok_start[], ple_tok_end[];
extern const char ple_golden_start[];

static float kcheck_buf[2 * 8192];

// ---------------------------------------------------------------- region markers
// Empty, never inlined: the QEMU plugin counts instructions between begin and end.
__attribute__((noinline)) void ple_mark_tok_begin(void)  { __asm volatile("" ::: "memory"); }
__attribute__((noinline)) void ple_mark_tok_end(void)    { __asm volatile("" ::: "memory"); }
__attribute__((noinline)) void ple_mark_head_begin(void) { __asm volatile("" ::: "memory"); }
__attribute__((noinline)) void ple_mark_head_end(void)   { __asm volatile("" ::: "memory"); }

// ---------------------------------------------------------------- PMU (Armv8.1-M)
// Helium instructions (MVE_INST_RETIRED) and the cycle counter. Event counters are
// 16 bits wide on the Cortex-M55, so the count uses a pair: counter 0 counts the
// event and counter 1, set to CHAIN, counts its overflows. The cycle counter is 32
// bits. On the FVP it advances one per instruction (the model isn't cycle-accurate),
// so there it is the instruction count; on silicon it counts real cycles.
// (INST_RETIRED isn't used: on FVP 11.27 its CHAIN partner never advanced.)
#if PLE_PMU
#define REG(a) (*(volatile uint32_t *)(a))
#define DEMCR          REG(0xE000EDFC)
#define PMU_EVCNTR(n)  REG(0xE0003000 + 4 * (n))
#define PMU_CCNTR      REG(0xE000307C)
#define PMU_EVTYPER(n) REG(0xE0003400 + 4 * (n))
#define PMU_CNTENSET   REG(0xE0003C00)
#define PMU_CTRL       REG(0xE0003E04)
#define EV_CHAIN            0x001E
#define EV_MVE_INST_RETIRED 0x0200
static void pmu_init(void) {
    DEMCR |= 1u << 24;                                   // TRCENA: PMU and DWT on
    PMU_EVTYPER(0) = EV_MVE_INST_RETIRED;
    PMU_EVTYPER(1) = EV_CHAIN;                           // counters 0+1: Helium instructions
    PMU_CNTENSET = (1u << 31) | 3u;                      // cycle counter + counters 0, 1
    PMU_CTRL = 1u | 2u | 4u;                             // enable, reset counters, reset cycles
}
// The chained pair as one 32-bit value; re-read if the high half moved in between.
static uint32_t pmu_mve(void) {
    uint32_t hi, lo;
    do { hi = PMU_EVCNTR(1) & 0xFFFFu; lo = PMU_EVCNTR(0) & 0xFFFFu; }
    while (hi != (PMU_EVCNTR(1) & 0xFFFFu));
    return hi << 16 | lo;
}
typedef struct { uint32_t mve, cyc; } pmu_t;
static pmu_t pmu_read(void) { pmu_t p = { pmu_mve(), PMU_CCNTR }; return p; }
static pmu_t pmu_diff(pmu_t a, pmu_t b) { pmu_t d = { b.mve - a.mve, b.cyc - a.cyc }; return d; }
#endif

int main(void) {
    printf("=== PLE on %s, kernel %s ===\n", PLE_BOARD, kern_name());
#if PLE_PMU
    pmu_init();
#endif
    ple_model_t m;
    size_t msize = (size_t)(ple_model_end - ple_model_start);
    if (ple_load(&m, ple_model_start, msize)) { printf("bad model\n"); return 1; }
    printf("MODEL | fnv1a 0x%08lx | %u bytes | V=%u D=%u L=%u F=%u P=%u\n",
           (unsigned long)bench_fnv1a(ple_model_start, msize), (unsigned)msize,
           (unsigned)m.V, (unsigned)m.D, (unsigned)m.L, (unsigned)m.F, (unsigned)m.P);
    if (m.V > sizeof kcheck_buf / sizeof kcheck_buf[0] / 2) { printf("vocab too big for kcheck_buf\n"); return 1; }

    // 1. kernel bit-exactness (int8 kernels only: the float kernel has no exact contract)
#if PLE_KERNEL_I8
    int kbad = kern_check(ple_model_start, msize, ple_kvec_start,
                          (size_t)(ple_kvec_end - ple_kvec_start), kcheck_buf);
#else
    int kbad = 0;
    printf("KERNEL | %s | float activations: checked by the golden decode only\n", kern_name());
#endif

    // 2 + 3. golden decode, hashing every logit
    tokenizer_t tk;
    ple_state_t s;
    bench_golden_t g;
    if (tokenizer_load(&tk, ple_tok_start, (size_t)(ple_tok_end - ple_tok_start), malloc) ||
        ple_state_init(&s, &m, CTX, malloc) || bench_parse_golden(ple_golden_start, &g)) {
        printf("setup failed (tokenizer, state or golden)\n");
        return 1;
    }
    int tok = g.prompt[0], pos = 0, mism = 0, ties = 0, len;
    const uint8_t *tb = tokenizer_bytes(&tk, tok, &len);
    fwrite(tb, 1, (size_t)len, stdout);
    uint32_t h = 2166136261u;
#if PLE_PMU
    pmu_t t0 = pmu_read();
#endif
    for (int i = 0; i < g.np + g.nw - 1 && pos < CTX - 1; i++) {
        ple_mark_tok_begin();
        ple_forward(&m, &s, tok, pos, NULL);
        ple_mark_tok_end();
        const uint8_t *lb = (const uint8_t *)s.logits;
        for (size_t k = 0; k < 4 * (size_t)m.V; k++) { h ^= lb[k]; h *= 16777619u; }
        int nxt = (i + 1 < g.np) ? g.prompt[i + 1] : ple_argmax(s.logits, (int)m.V);
        if (i + 1 >= g.np) {
            int want = g.want[i + 1 - g.np];
            if (nxt != want) {
                if (s.logits[nxt] - s.logits[want] > BENCH_TIE_GAP) mism++; else ties++;
                nxt = want;                                  // stay on the golden path
            }
        }
        const uint8_t *b = tokenizer_bytes(&tk, nxt, &len);
        fwrite(b, 1, (size_t)len, stdout);
        tok = nxt; pos++;
    }
#if PLE_PMU
    pmu_t tt = pmu_diff(t0, pmu_read());
#endif
    printf("\nGOLDEN: %s (%d mismatches, %d ties) | %s | kernel %s | %d tokens\n",
           mism ? "MISMATCH" : "MATCH", mism, ties, PLE_BOARD, kern_name(), pos);
    printf("LOGITS | fnv1a 0x%08lx | %d tokens x %u logits | kernel %s\n",
           (unsigned long)h, pos, (unsigned)m.V, kern_name());

    // 4. one head matvec on its own (the biggest matvec: vocab x d_model)
#if PLE_PMU
    pmu_t h0 = pmu_read();
#endif
    ple_mark_head_begin();
    kern_prepare(s.x, m.D);
    kern_matvec(m.tok, m.D, s.x, s.logits, 0, (int)m.V);
    ple_mark_head_end();
#if PLE_PMU
    pmu_t hh = pmu_diff(h0, pmu_read());
    if (!tt.cyc) printf("PMU | not available on this target (QEMU: use make count)\n");
    else printf("PMU | per token: %lu cycles, %lu MVE inst | head matvec %ux%u: %lu cycles, %lu MVE inst"
                " | FVP: cycles = instructions\n",
                (unsigned long)(tt.cyc / (uint32_t)pos), (unsigned long)(tt.mve / (uint32_t)pos),
                (unsigned)m.V, (unsigned)m.D, (unsigned long)hh.cyc, (unsigned long)hh.mve);
#endif
    printf("=== done ===\n");
    return (kbad || mism) ? 1 : 0;
}
