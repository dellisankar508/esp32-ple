#include "ple.h"
#include "f16.h"
#include "kern.h"
#include "ple_port.h"
#include <math.h>
#include <string.h>

static inline uint32_t row_bytes(uint32_t in) { return kern_row_bytes(in); }
static inline uint32_t clk(ple_time_fn f) { return f ? f() : 0; }

// ---------------------------------------------------------------- load
int ple_load(ple_model_t *m, const uint8_t *b, size_t size) {
    if (size < 64 || memcmp(b, "PLE1", 4) != 0) return -1;
    const uint32_t *h = (const uint32_t *)b;
    m->V = h[2]; m->D = h[3]; m->L = h[4]; m->H = h[5]; m->F = h[6]; m->P = h[7]; m->C = h[8]; m->G = h[9];
    uint32_t off_tok = h[10], off_ple = h[11], off_layers = h[12], off_norm = h[13], total = h[14];
    if (m->G != 32 || m->L > PLE_MAX_LAYERS || total > size) return -2;
    m->size = total;
    m->base = b; m->tok = b + off_tok; m->ple = b + off_ple;
    m->norm = (const float *)(b + off_norm);
    const uint8_t *p = b + off_layers;
    for (uint32_t l = 0; l < m->L; l++) {
        m->layer[l].ple_proj = p; p += m->D * row_bytes(m->P);
        m->layer[l].n1 = (const float *)p; p += 4 * m->D;
        m->layer[l].qkv = p; p += 3 * m->D * row_bytes(m->D);
        m->layer[l].o   = p; p += m->D * row_bytes(m->D);
        m->layer[l].n2 = (const float *)p; p += 4 * m->D;
        m->layer[l].w1  = p; p += m->F * row_bytes(m->D);
        m->layer[l].w3  = p; p += m->F * row_bytes(m->D);
        m->layer[l].w2  = p; p += m->D * row_bytes(m->F);
    }
    return (p == b + off_norm) ? 0 : -3;
}

// ---------------------------------------------------------------- state
size_t ple_state_bytes(const ple_model_t *m, int ctx) {
    size_t hd2 = m->D / m->H / 2;
    size_t f = m->D * 5 + 3 * m->D + ctx + 2 * m->F + m->P + m->V      // activations + logits
             + 2 * m->L * ctx * m->H                                    // kv scales
             + 2 * ctx * hd2;                                           // rope table
    return f * 4 + (size_t)2 * m->L * ctx * m->D;                       // + int8 KV
}

// Weight bytes one token streams from flash, split by timed stage. These set the
// flash floor: the time a stage would take if its math were free.
void ple_stage_bytes(const ple_model_t *m, size_t out[4]) {
    size_t rbD = row_bytes(m->D), rbP = row_bytes(m->P), rbF = row_bytes(m->F);
    size_t ple_slice = m->P / 2 + 2 * (m->P / 32);       // nibbles + fp16 scales of one layer's slice
    out[0] = (size_t)m->L * (ple_slice + (size_t)m->D * rbP);                          // embed
    out[1] = (size_t)m->L * (4 * m->D + (size_t)4 * m->D * rbD);                       // attn: n1, qkv(3D), o(D)
    out[2] = (size_t)m->L * (4 * m->D + (size_t)2 * m->F * rbD + (size_t)m->D * rbF);  // ffn: n2, w1, w3, w2
    out[3] = (size_t)m->V * rbD + 4 * m->D;                                            // head + final norm
}

size_t ple_bytes_per_token(const ple_model_t *m) {
    size_t st[4];
    ple_stage_bytes(m, st);
    return st[0] + st[1] + st[2] + st[3] + row_bytes(m->D);    // + the token's embedding row
}

int ple_state_init(ple_state_t *s, const ple_model_t *m, int ctx, ple_alloc_fn alloc) {
    memset(s, 0, sizeof *s);
    s->ctx = ctx;
    uint32_t hd = m->D / m->H, hd2 = hd / 2;
    s->kc = alloc((size_t)m->L * ctx * m->D);          // biggest blocks first
    s->vc = alloc((size_t)m->L * ctx * m->D);
    s->logits = alloc(m->V * 4);
    s->ks = alloc((size_t)m->L * ctx * m->H * 4);
    s->vs = alloc((size_t)m->L * ctx * m->H * 4);
    s->rope_cos = alloc((size_t)ctx * hd2 * 4);
    s->rope_sin = alloc((size_t)ctx * hd2 * 4);
    s->h = alloc(m->D * 4); s->x = alloc(m->D * 4); s->tmp = alloc(m->D * 4);
    s->qkv = alloc(3 * m->D * 4);
    s->att = alloc(ctx * 4);
    s->f1 = alloc(m->F * 4); s->f3 = alloc(m->F * 4);
    s->ple_row = alloc(m->P * 4);
    if (!(s->kc && s->vc && s->logits && s->ks && s->vs && s->rope_cos && s->rope_sin &&
          s->h && s->x && s->tmp && s->qkv && s->att && s->f1 && s->f3 && s->ple_row)) return -1;
    for (int p = 0; p < ctx; p++)
        for (uint32_t i = 0; i < hd2; i++) {
            float a = p * powf(10000.f, -(float)(2 * i) / hd);
            s->rope_cos[p * hd2 + i] = cosf(a);
            s->rope_sin[p * hd2 + i] = sinf(a);
        }
    return 0;
}

// ---------------------------------------------------------------- matvec via backend
static inline void matvec(const uint8_t *w, uint32_t out, uint32_t in, const float *x, float *y) {
    kern_prepare(x, in);
    kern_matvec(w, in, x, y, 0, (int)out);
}

typedef struct { const uint8_t *w; uint32_t in; const float *x; float *y; } job_t;
static void head_job(void *arg, int r0, int r1) {
    job_t *j = arg; kern_matvec(j->w, j->in, j->x, j->y, r0, r1);
}

static PLE_HOT void deq_row(const uint8_t *w, uint32_t in, uint32_t row, uint32_t c0, uint32_t n, float *out) {
    const uint8_t *r = w + row * row_bytes(in);
    const uint16_t *sc = (const uint16_t *)(r + in / 2);
    for (uint32_t c = c0; c < c0 + n; c++) {
        uint8_t b = r[c / 2];
        int q = (c & 1) ? ((int)((int8_t)b) >> 4) : ((int)((int8_t)(b << 4)) >> 4);
        out[c - c0] = q * f16_to_f32(sc[c / 32]);
    }
}

static void rmsnorm(float *o, const float *x, const float *w, uint32_t n) {
    float ss = 0.f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + 1e-6f);
    for (uint32_t i = 0; i < n; i++) o[i] = w[i] * x[i] * inv;
}

static inline void rope(float *v, uint32_t H, uint32_t hd, const float *cs, const float *sn) {
    for (uint32_t h = 0; h < H; h++) {
        float *p = v + h * hd;
        for (uint32_t i = 0; i < hd / 2; i++) {
            float x1 = p[2 * i], x2 = p[2 * i + 1];
            p[2 * i]     = x1 * cs[i] - x2 * sn[i];
            p[2 * i + 1] = x1 * sn[i] + x2 * cs[i];
        }
    }
}

// quantize one head's hd floats to int8 with a shared scale
static inline void q8_head(const float *v, uint32_t hd, int8_t *out, float *scale) {
    float mx = 0.f;
    for (uint32_t i = 0; i < hd; i++) { float a = fabsf(v[i]); if (a > mx) mx = a; }
    float s = mx / 127.f, inv = mx > 0.f ? 127.f / mx : 0.f;
    *scale = s;
    for (uint32_t i = 0; i < hd; i++) out[i] = (int8_t)lrintf(v[i] * inv);
}

// ---------------------------------------------------------------- forward
void ple_forward(const ple_model_t *m, ple_state_t *s, int token, int pos, ple_time_fn now) {
    const uint32_t D = m->D, H = m->H, hd = D / H, hd2 = hd / 2, F = m->F, P = m->P, L = m->L;
    const float scale = 1.0f / sqrtf((float)hd);
    const float *cs = s->rope_cos + pos * hd2, *sn = s->rope_sin + pos * hd2;
    s->t_embed = s->t_attn = s->t_ffn = s->t_head = 0;

    deq_row(m->tok, D, token, 0, D, s->h);
    for (uint32_t l = 0; l < L; l++) {
        uint32_t ta = clk(now);
        deq_row(m->ple, L * P, token, l * P, P, s->ple_row);
        matvec(m->layer[l].ple_proj, D, P, s->ple_row, s->tmp);
        for (uint32_t i = 0; i < D; i++) s->h[i] += s->tmp[i];
        uint32_t tb = clk(now); s->t_embed += tb - ta;

        rmsnorm(s->x, s->h, m->layer[l].n1, D);
        matvec(m->layer[l].qkv, 3 * D, D, s->x, s->qkv);
        float *q = s->qkv, *k = s->qkv + D, *v = s->qkv + 2 * D;
        rope(q, H, hd, cs, sn); rope(k, H, hd, cs, sn);

        size_t base = ((size_t)l * s->ctx + pos);
        for (uint32_t h = 0; h < H; h++) {
            q8_head(k + h * hd, hd, s->kc + base * D + h * hd, &s->ks[base * H + h]);
            q8_head(v + h * hd, hd, s->vc + base * D + h * hd, &s->vs[base * H + h]);
        }
        for (uint32_t h = 0; h < H; h++) {
            const float *qh = q + h * hd;
            float mx = -1e30f;
            for (int t = 0; t <= pos; t++) {
                size_t bt = (size_t)l * s->ctx + t;
                const int8_t *kt = s->kc + bt * D + h * hd;
                float fd = 0.f;                       // q float x K int8, one scale per head
                for (uint32_t i = 0; i < hd; i++) fd += qh[i] * kt[i];
                s->att[t] = fd * s->ks[bt * H + h] * scale;
                if (s->att[t] > mx) mx = s->att[t];
            }
            float sum = 0.f;
            for (int t = 0; t <= pos; t++) { s->att[t] = expf(s->att[t] - mx); sum += s->att[t]; }
            float inv = 1.f / sum;
            float *o = s->x + h * hd;
            for (uint32_t i = 0; i < hd; i++) o[i] = 0.f;
            for (int t = 0; t <= pos; t++) {
                size_t bt = (size_t)l * s->ctx + t;
                const int8_t *vt = s->vc + bt * D + h * hd;
                float p = s->att[t] * inv * s->vs[bt * H + h];
                for (uint32_t i = 0; i < hd; i++) o[i] += p * vt[i];
            }
        }
        matvec(m->layer[l].o, D, D, s->x, s->tmp);
        for (uint32_t i = 0; i < D; i++) s->h[i] += s->tmp[i];
        uint32_t tc = clk(now); s->t_attn += tc - tb;

        rmsnorm(s->x, s->h, m->layer[l].n2, D);
        matvec(m->layer[l].w1, F, D, s->x, s->f1);
        matvec(m->layer[l].w3, F, D, s->x, s->f3);
        for (uint32_t i = 0; i < F; i++) { float a = s->f1[i]; s->f1[i] = a / (1.0f + expf(-a)) * s->f3[i]; }
        matvec(m->layer[l].w2, D, F, s->f1, s->tmp);
        for (uint32_t i = 0; i < D; i++) s->h[i] += s->tmp[i];
        s->t_ffn += clk(now) - tc;
    }
    uint32_t th = clk(now);
    rmsnorm(s->x, s->h, m->norm, D);
    if (s->head) {
        kern_prepare(s->x, D);
        s->head(s->parallel, m->tok, m->V, D, s->x, s->logits);
    } else if (s->parallel) {
        kern_prepare(s->x, D);                     // before any core touches a row range
        job_t j = { m->tok, D, s->x, s->logits };
        s->parallel(head_job, &j, (int)m->V);
    } else matvec(m->tok, m->V, D, s->x, s->logits);
    s->t_head = clk(now) - th;
}

int ple_argmax(const float *lg, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (lg[i] > lg[b]) b = i;
    return b;
}

int ple_sample(const float *lg, int n, float temp, uint32_t *rng) {
    if (temp <= 0.f) return ple_argmax(lg, n);
    float mx = lg[ple_argmax(lg, n)], sum = 0.f;
    for (int i = 0; i < n; i++) sum += expf((lg[i] - mx) / temp);
    *rng = *rng * 1664525u + 1013904223u;
    float r = (*rng >> 8) / 16777216.0f * sum, c = 0.f;
    for (int i = 0; i < n; i++) { c += expf((lg[i] - mx) / temp); if (c >= r) return i; }
    return n - 1;
}
