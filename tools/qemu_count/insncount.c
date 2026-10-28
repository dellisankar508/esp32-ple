/*
 * QEMU TCG plugin: exact executed-instruction counts for marked regions of a
 * bare-metal Arm M-profile program (platform/corstone300). QEMU is not cycle-accurate,
 * so these are instruction counts, not timings; they compare kernels exactly.
 *
 * Arguments (repeatable):
 *   region=NAME:BEGIN:END   count from each execution of address BEGIN to the next END
 *   kernel=LO:HI            addresses [LO, HI) count as "kernel" instructions
 * Per region it prints: runs, instructions, kernel instructions, Helium (MVE)
 * instructions. MVE is recognised by encoding: T32 Advanced-SIMD-like space
 * (0xEF/0xFF first halfword), coprocessor space with coprocessor 14/15, and the
 * core-register transfers VMOV (to or from a vector lane) and VDUP. This matches the
 * Cortex-M55's MVE_INST_RETIRED count on Arm's Corstone-300 FVP exactly.
 *
 * Build (QEMU's include/qemu/qemu-plugin.h, same version as qemu-system-arm):
 *   gcc -shared -fPIC -O2 -I<qemu>/include/qemu $(pkg-config --cflags glib-2.0) \
 *       -o libinsncount.so insncount.c
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

#define MAXR 8
#define MAXK 16
typedef struct { uint64_t total, kernel, mve; } ctr_t;
static ctr_t now;                                  // single vCPU: plain counters
static struct { char name[32]; uint64_t begin, end; ctr_t start, sum; uint64_t runs; int open; } reg[MAXR];
static int nreg;
static struct { uint64_t lo, hi; } kr[MAXK];
static int nk;

static int is_mve(const struct qemu_plugin_insn *insn) {
    if (qemu_plugin_insn_size(insn) != 4) return 0;
    const uint8_t *b = qemu_plugin_insn_data(insn);
    uint16_t h1 = (uint16_t)(b[0] | b[1] << 8), h2 = (uint16_t)(b[2] | b[3] << 8);
    if ((h1 & 0xEF00) == 0xEF00) return 1;                            // vector data-processing
    if ((h1 & 0xEC00) == 0xEC00 && ((h2 >> 9) & 7) == 7) return 1;     // cp14/15: loads, MLADAV, ...
    if ((h1 & 0xFF00) == 0xEE00 && (h2 & 0x0F1F) == 0x0B10) return 1;  // VMOV to/from a lane, VDUP
    return 0;
}

static void on_begin(unsigned int vcpu, void *ud) {
    (void)vcpu; int i = (int)(intptr_t)ud;
    reg[i].start = now; reg[i].open = 1;
}
static void on_end(unsigned int vcpu, void *ud) {
    (void)vcpu; int i = (int)(intptr_t)ud;
    if (!reg[i].open) return;
    reg[i].sum.total += now.total - reg[i].start.total;
    reg[i].sum.kernel += now.kernel - reg[i].start.kernel;
    reg[i].sum.mve += now.mve - reg[i].start.mve;
    reg[i].runs++; reg[i].open = 0;
}

static void on_tb(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
    (void)id;
    for (size_t n = 0; n < qemu_plugin_tb_n_insns(tb); n++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, n);
        uint64_t pc = qemu_plugin_insn_vaddr(insn);
        for (int i = 0; i < nreg; i++) {          // markers first, so a region excludes them
            if (pc == reg[i].begin) qemu_plugin_register_vcpu_insn_exec_cb(insn, on_begin, QEMU_PLUGIN_CB_NO_REGS, (void *)(intptr_t)i);
            if (pc == reg[i].end) qemu_plugin_register_vcpu_insn_exec_cb(insn, on_end, QEMU_PLUGIN_CB_NO_REGS, (void *)(intptr_t)i);
        }
        qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64, &now.total, 1);
        for (int k = 0; k < nk; k++)
            if (pc >= kr[k].lo && pc < kr[k].hi) {
                qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64, &now.kernel, 1);
                break;
            }
        if (is_mve(insn)) qemu_plugin_register_vcpu_insn_exec_inline(insn, QEMU_PLUGIN_INLINE_ADD_U64, &now.mve, 1);
    }
}

static void plugin_exit(qemu_plugin_id_t id, void *p) {
    (void)id; (void)p;
    for (int i = 0; i < nreg; i++) {
        char *s = g_strdup_printf("QCOUNT | %s | runs %" PRIu64 " | insns %" PRIu64 " | kernel %" PRIu64 " | mve %" PRIu64 "\n",
                                  reg[i].name, reg[i].runs, reg[i].sum.total, reg[i].sum.kernel, reg[i].sum.mve);
        qemu_plugin_outs(s);
        g_free(s);
    }
    char *s = g_strdup_printf("QCOUNT | whole run | insns %" PRIu64 "\n", now.total);
    qemu_plugin_outs(s);
    g_free(s);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv) {
    (void)info;
    for (int a = 0; a < argc; a++) {
        if (!strncmp(argv[a], "region=", 7) && nreg < MAXR) {
            char *p = argv[a] + 7, *c1 = strchr(p, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
            if (!c2) return -1;
            snprintf(reg[nreg].name, sizeof reg[nreg].name, "%.*s", (int)(c1 - p), p);
            reg[nreg].begin = strtoull(c1 + 1, NULL, 0);
            reg[nreg].end = strtoull(c2 + 1, NULL, 0);
            nreg++;
        } else if (!strncmp(argv[a], "kernel=", 7) && nk < MAXK) {
            char *p = argv[a] + 7, *c = strchr(p, ':');
            if (!c) return -1;
            kr[nk].lo = strtoull(p, NULL, 0);
            kr[nk].hi = strtoull(c + 1, NULL, 0);
            nk++;
        } else {
            fprintf(stderr, "insncount: unknown argument %s\n", argv[a]);
            return -1;
        }
    }
    qemu_plugin_register_vcpu_tb_trans_cb(id, on_tb);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    return 0;
}
