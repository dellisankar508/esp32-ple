#include "head_stream.h"
#include "kern.h"
#include "ple_port.h"
#include <string.h>

// Shared between both cores and the mover's completion callback.
//   issued, busy: only touched with the mover lock held
//   fetched:      written with the lock held, read by the cores without it
//   done[c]:      written by core c only, read by kick_locked()
static struct {
    const hs_mover_t *mv;
    uint8_t *bufs;
    uint32_t buf_bytes, nbuf;
    const uint8_t *src;            // this call
    const float *x;
    float *y;
    uint32_t in, rb, rows, rpb, nblk;
    int ncores, compute;
    ple_time_fn now;               // set only by hs_move_ms: enables the timeout
    uint32_t t_wait;
    volatile int failed;
    uint32_t issued, busy;
    uint32_t fetched;              // blocks now in SRAM
    uint32_t done[2];              // blocks finished by core 0 / core 1
} H;

#define LOAD(p)     __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define STORE(p, v) __atomic_store_n((p), (v), __ATOMIC_RELEASE)

static inline uint32_t rows_in(uint32_t b) {
    uint32_t r0 = b * H.rpb;
    return H.rows - r0 < H.rpb ? H.rows - r0 : H.rpb;
}
static inline uint8_t *buf_of(uint32_t b) { return H.bufs + (size_t)(b % H.nbuf) * H.buf_bytes; }
static inline const uint8_t *src_of(uint32_t b) { return H.src + (size_t)b * H.rpb * H.rb; }

uint32_t hs_rows_per_block(uint32_t in) {
    return H.mv ? (H.buf_bytes / kern_row_bytes(in)) & ~1u : 0;
}

int hs_init(const hs_mover_t *mv, uint8_t *bufs, uint32_t buf_bytes, int nbuf) {
    if (!mv || !mv->start || !mv->lock || !mv->unlock || !bufs || nbuf < 2 ||
        ((uintptr_t)bufs & 3) || (buf_bytes & 3)) return -1;
    memset(&H, 0, sizeof H);
    H.mv = mv; H.bufs = bufs; H.buf_bytes = buf_bytes; H.nbuf = (uint32_t)nbuf;
    return 0;
}

// Start the next block if the mover is idle and the block's buffer is free. Lock held.
static void kick_locked(void) {
    if (H.busy || H.issued >= H.nblk) return;
    uint32_t b = H.issued;
    if (b >= H.nbuf) {
        uint32_t prev = b - H.nbuf;                 // previous block in this buffer
        if (LOAD(&H.done[0]) <= prev) return;
        if (H.ncores > 1 && LOAD(&H.done[1]) <= prev) return;
    }
    H.busy = 1;
    H.issued = b + 1;
    H.mv->start(src_of(b), buf_of(b), rows_in(b) * H.rb);
}

static void kick(void) {
    uint32_t k = H.mv->lock();
    kick_locked();
    H.mv->unlock(k);
}

void hs_block_arrived(void) {
    uint32_t k = H.mv->lock();
    H.busy = 0;
    STORE(&H.fetched, H.fetched + 1);
    kick_locked();
    H.mv->unlock(k);
}

static PLE_HOT void work(int c) {
    for (uint32_t b = 0; b < H.nblk; b++) {
        if (H.now) H.t_wait = H.now();
        while (LOAD(&H.fetched) <= b) {
            if (H.mv->relax) H.mv->relax();
            if (H.now && H.now() - H.t_wait > 200000u) { H.failed = 1; return; }
        }
        uint32_t n = rows_in(b), mid = H.ncores > 1 ? n / 2 : n;
        uint32_t r0 = c ? mid : 0, r1 = c ? n : mid;
        if (H.compute && r1 > r0)
            kern_matvec(buf_of(b), H.in, H.x, H.y + (size_t)b * H.rpb, (int)r0, (int)r1);
        STORE(&H.done[c], b + 1);
        kick();
    }
}

static void job(void *arg, int r0, int r1) { (void)arg; (void)r1; work(r0); }

// Under the lock, so a completion callback still finishing the previous call
// (host threads) is done before the counters reset.
static int setup(const uint8_t *w, uint32_t rows, uint32_t in, const float *x, float *y,
                 int ncores, int compute) {
    uint32_t k = H.mv->lock();
    H.src = w; H.rows = rows; H.in = in; H.x = x; H.y = y;
    H.rb = kern_row_bytes(in);
    H.rpb = (H.buf_bytes / H.rb) & ~1u;
    H.nblk = H.rpb ? (rows + H.rpb - 1) / H.rpb : 0;
    H.ncores = ncores; H.compute = compute;
    H.now = NULL; H.failed = 0;
    H.issued = H.busy = 0;
    STORE(&H.fetched, 0);
    STORE(&H.done[0], 0);
    STORE(&H.done[1], 0);
    H.mv->unlock(k);
    return H.rpb >= 2 ? 0 : -1;
}

void hs_head(ple_parallel_fn par, const uint8_t *w, uint32_t rows, uint32_t in,
             const float *x, float *y) {
    if (!H.mv || setup(w, rows, in, x, y, par ? 2 : 1, 1)) {   // no buffers: plain path
        kern_matvec(w, in, x, y, 0, (int)rows);
        return;
    }
    if (H.mv->begin) H.mv->begin(w, rows * H.rb);
    kick();
    if (par) par(job, NULL, 2);
    else work(0);
}

float hs_move_ms(const uint8_t *w, uint32_t rows, uint32_t in, ple_time_fn now) {
    if (!H.mv || !now || setup(w, rows, in, NULL, NULL, 1, 0)) return -1;
    H.now = now;
    uint32_t t0 = now();
    if (H.mv->begin) H.mv->begin(w, rows * H.rb);
    kick();
    work(0);
    uint32_t dt = now() - t0;
    H.now = NULL;
    if (H.failed) {                                  // a block never arrived
        uint32_t k = H.mv->lock();
        H.issued = H.nblk;                           // no more starts
        H.busy = 0;
        H.mv->unlock(k);
        if (H.mv->abort) H.mv->abort();
        return -1;
    }
    // The last nbuf blocks are still in the buffers: they must equal the source.
    uint32_t first = H.nblk > H.nbuf ? H.nblk - H.nbuf : 0;
    for (uint32_t b = first; b < H.nblk; b++)
        if (memcmp(buf_of(b), src_of(b), rows_in(b) * H.rb)) return -1;
    return dt / 1000.0f;
}
