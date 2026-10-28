"""
Export ple_d0wd96.pt -> model.bin (PLE1, int4 group-32) + tokenizer.bin + golden.txt

    python export_ple.py                 # uses ple_d0wd96.pt + tokenizer.json in cwd
    python export_ple.py --check         # also runs a numpy reference forward on the exported file
    python export_ple.py --regolden ../model             # numpy only: rewrite golden.txt
    python export_ple.py --regolden ../model --act i8    # numpy only: write golden_i8.txt

PLE1 binary layout (little-endian)
----------------------------------
Header, 64 bytes:
  0  char[4]  "PLE1"
  4  u32      version = 1
  8  u32      vocab, d_model, n_layers, n_heads, ffn_hidden, ple_dim, ctx, group   (8 x u32 = 32 B)
  40 u32      off_tok, off_ple, off_layers, off_norm, total_size                    (5 x u32 = 20 B)
  60 u32      reserved
Every weight matrix W[out, in] is stored row-packed, one row after another:
  row = int4 nibbles (in/2 bytes, low nibble = even column, two's complement -8..7)
        followed by fp16 scales, one per group of 32 columns (in/16 bytes),
        padded to a multiple of 4 bytes.        row_bytes = align4(in/2 + in/16)
  value = nibble * scale[col // 32]
Sections:
  off_tok    tok_emb  [vocab, d_model]           (also the tied output head)
  off_ple    ple      [vocab, n_layers*ple_dim]  layer l uses columns [l*ple_dim, (l+1)*ple_dim)
  off_layers per layer, in order: ple_proj[d,ple] n1(f32 d) qkv[3d,d] o[d,d] n2(f32 d) w1[f,d] w3[f,d] w2[d,f]
  off_norm   final norm (f32 d)
Each section starts on a 16-byte boundary.
"""
import argparse, json, os, struct, sys
import numpy as np

GROUP = 32
NORM_EPS = 1e-6


# ---------------------------------------------------------------- quantization
def quant_rows(w: np.ndarray) -> bytes:
    """w: [out, in] float32 -> row-packed int4 + fp16 scales, padded to 4 bytes."""
    out, inn = w.shape
    assert inn % GROUP == 0, f"in={inn} not a multiple of {GROUP}"
    g = w.reshape(out, inn // GROUP, GROUP)
    scale = np.abs(g).max(-1) / 7.0                      # [out, groups]
    scale = np.where(scale == 0, 1e-8, scale).astype(np.float16)
    q = np.clip(np.round(g / scale[..., None].astype(np.float32)), -8, 7).astype(np.int8)
    q = q.reshape(out, inn)
    lo, hi = (q[:, 0::2] & 0xF).astype(np.uint8), (q[:, 1::2] & 0xF).astype(np.uint8)
    packed = (lo | (hi << 4)).astype(np.uint8)          # [out, in/2]
    row_bytes = align4(inn // 2 + inn // 16)
    buf = np.zeros((out, row_bytes), np.uint8)
    buf[:, :inn // 2] = packed
    buf[:, inn // 2:inn // 2 + inn // 16] = scale.view(np.uint8).reshape(out, -1)
    return buf.tobytes()


def dequant_rows(buf: np.ndarray, out: int, inn: int) -> np.ndarray:
    """Inverse of quant_rows (reference for the C kernel)."""
    row_bytes = align4(inn // 2 + inn // 16)
    rows = buf.reshape(out, row_bytes)
    packed = rows[:, :inn // 2]
    lo = (packed & 0xF).astype(np.int8); hi = (packed >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo); hi = np.where(hi > 7, hi - 16, hi)
    q = np.empty((out, inn), np.int8); q[:, 0::2] = lo; q[:, 1::2] = hi
    scale = rows[:, inn // 2:inn // 2 + inn // 16].copy().view(np.float16).astype(np.float32)
    return (q.astype(np.float32).reshape(out, inn // GROUP, GROUP) * scale[..., None]).reshape(out, inn)


def dequant_parts(buf: np.ndarray, out: int, inn: int):
    """Like dequant_rows, but keeps the parts: int4 values as int8 [out, in] and the
    fp16 group scales widened to float32 [out, in/32] (fp16 -> fp32 is exact)."""
    row_bytes = align4(inn // 2 + inn // 16)
    rows = buf.reshape(out, row_bytes)
    packed = rows[:, :inn // 2]
    lo = (packed & 0xF).astype(np.int8); hi = (packed >> 4).astype(np.int8)
    lo = np.where(lo > 7, lo - 16, lo); hi = np.where(hi > 7, hi - 16, hi)
    q = np.empty((out, inn), np.int8); q[:, 0::2] = lo; q[:, 1::2] = hi
    scale = rows[:, inn // 2:inn // 2 + inn // 16].copy().view(np.float16).astype(np.float32)
    return q, scale


# ---------------------------------------------------------------- int8-activation matvec (I8 v1)
# The contract every int8 kernel (kernels/i8ref.c, dsp.c, mve.c) must meet bit for bit.
# For y = W x with W int4 group-32 (fp16 scales s) and x float32:
#  1. per group g of 32 inputs: m = max|x|; m < 2^-60 -> sx = 0, q = 0 (see below);
#     else inv = 127/m, sx = m/127, q = clamp(rint(x * inv), -127, 127)   (round half to even)
#  2. per row r and group g: d = sum of w * q over the 32 columns, exact in int32
#  3. y_r = (...((0 + d_0 * c_0) + d_1 * c_1) ...) with c_g = s_rg * sx_g, groups in order,
#     every * and + rounded to float32, no fused multiply-add
# Every step is float32 or exact integer math, so numpy and C agree to the last bit.
# The 2^-60 cut-off keeps every product in step 3 a normal float (fp16 scales are at
# least 2^-24), so vector units that flush subnormals to zero still agree exactly.
F32 = np.float32


def act_q8(x: np.ndarray):
    """x [T, in] float32 -> q int8 [T, in], sx float32 [T, in/32]  (step 1)."""
    T, inn = x.shape
    g = x.astype(F32).reshape(T, inn // GROUP, GROUP)
    m = np.abs(g).max(-1)
    nz = m >= F32(2.0 ** -60)
    inv = np.where(nz, F32(127) / np.where(nz, m, F32(1)), F32(0)).astype(F32)
    sx = np.where(nz, m / F32(127), F32(0)).astype(F32)
    q = np.clip(np.rint(g * inv[..., None]), -127, 127).astype(np.int8)
    return q.reshape(T, inn), sx


def matvec_i8(wq: np.ndarray, ws: np.ndarray, x: np.ndarray) -> np.ndarray:
    """I8 v1 matvec for a batch of inputs. wq int8 [out, in], ws float32 [out, in/32],
    x float32 [T, in] -> y float32 [T, out]."""
    q, sx = act_q8(x)
    T, inn = q.shape
    ng = inn // GROUP
    q = q.reshape(T, ng, GROUP).astype(np.int32)
    w = wq.reshape(wq.shape[0], ng, GROUP).astype(np.int32)
    acc = np.zeros((T, wq.shape[0]), F32)
    for g in range(ng):                                   # groups in order, as in C
        d = (q[:, g, :] @ w[:, g, :].T).astype(F32)       # exact int32 sums, then to float
        c = (ws[None, :, g] * sx[:, g, None]).astype(F32)
        acc = (acc + (d * c).astype(F32)).astype(F32)
    return acc


def align4(n):  return (n + 3) & ~3
def align16(n): return (n + 15) & ~15


# ---------------------------------------------------------------- export
def export(state: dict, cfg: dict, out_path: str):
    V, D, L, H, F, P, C = (cfg[k] for k in ("vocab", "d_model", "n_layers", "n_heads", "ffn_hidden", "ple_dim", "ctx"))
    f32 = lambda k: state[k].astype(np.float32)

    sections = []                                         # (name, bytes)
    sections.append(("tok", quant_rows(f32("tok.weight"))))
    sections.append(("ple", quant_rows(f32("ple.weight"))))
    layers = b""
    for l in range(L):
        p = f"blocks.{l}."
        layers += quant_rows(f32(p + "ple_proj.weight"))
        layers += f32(p + "n1.w").tobytes()
        layers += quant_rows(f32(p + "qkv.weight"))
        layers += quant_rows(f32(p + "o.weight"))
        layers += f32(p + "n2.w").tobytes()
        layers += quant_rows(f32(p + "w1.weight"))
        layers += quant_rows(f32(p + "w3.weight"))
        layers += quant_rows(f32(p + "w2.weight"))
    sections.append(("layers", layers))
    sections.append(("norm", f32("norm.w").tobytes()))

    off, offsets, body = 64, {}, b""
    for name, blob in sections:
        off = align16(off)
        body += b"\0" * (off - 64 - len(body))
        offsets[name] = off
        body += blob
        off += len(blob)
    total = 64 + len(body)

    header = b"PLE1" + struct.pack("<I", 1)
    header += struct.pack("<8I", V, D, L, H, F, P, C, GROUP)
    header += struct.pack("<5I", offsets["tok"], offsets["ple"], offsets["layers"], offsets["norm"], total)
    header += struct.pack("<I", 0)
    assert len(header) == 64
    with open(out_path, "wb") as fh:
        fh.write(header + body)
    print(f"{out_path}: {total} bytes ({total/1024/1024:.2f} MB)  "
          f"tok {len(sections[0][1])//1024} KB, ple {len(sections[1][1])//1024} KB, "
          f"layers {len(layers)//1024} KB")
    return total


# ---------------------------------------------------------------- tokenizer
def bytelevel_map():
    """GPT-2 byte<->unicode table used by the ByteLevel pre-tokenizer."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def export_tokenizer(tok_json: str, out_path: str):
    """tokenizer.bin: u32 vocab, then per token: u8 len, raw bytes."""
    vocab = json.load(open(tok_json, encoding="utf-8"))["model"]["vocab"]
    u2b = bytelevel_map()
    inv = {i: t for t, i in vocab.items()}
    n = max(inv) + 1
    out = struct.pack("<I", n)
    for i in range(n):
        t = inv.get(i, "")
        raw = bytes(u2b.get(ch, ord("?")) for ch in t) if t != "<eos>" else b""
        out += struct.pack("<B", len(raw)) + raw
    open(out_path, "wb").write(out)
    print(f"{out_path}: {len(out)} bytes, {n} tokens")


# ---------------------------------------------------------------- numpy reference forward
class Ref:
    """Dequantized reference. Same math the C runtime must reproduce, int8 KV cache included.
    act="f32": float activations, matching kernels/scalar.c.
    act="i8":  int8 activations (I8 v1 above), matching kernels/i8ref.c, dsp.c and mve.c."""
    def __init__(self, path, act="f32"):
        assert act in ("f32", "i8")
        self.act = act
        b = np.fromfile(path, np.uint8)
        assert b[:4].tobytes() == b"PLE1"
        (self.V, self.D, self.L, self.H, self.F, self.P, self.C, g) = struct.unpack_from("<8I", b, 8)
        ot, op, ol, on, tot = struct.unpack_from("<5I", b, 40)
        assert tot == len(b) and g == GROUP
        V, D, L, F, P = self.V, self.D, self.L, self.F, self.P
        rb = lambda inn: align4(inn // 2 + inn // 16)
        self.tok = dequant_rows(b[ot:ot + V * rb(D)], V, D)
        self.tok_i8 = dequant_parts(b[ot:ot + V * rb(D)], V, D) if act == "i8" else None
        self.ple = dequant_rows(b[op:op + V * rb(L * P)], V, L * P)
        self.layers = []
        o = ol
        for _ in range(L):
            lay = {}
            def mat(name, out, inn):
                nonlocal o
                n = out * rb(inn); lay[name] = dequant_rows(b[o:o + n], out, inn)
                if act == "i8":
                    lay[name + "_i8"] = dequant_parts(b[o:o + n], out, inn)
                o += n
            def vec(name):
                nonlocal o
                lay[name] = b[o:o + 4 * D].view(np.float32).copy(); o += 4 * D
            mat("ple_proj", D, P); vec("n1"); mat("qkv", 3 * D, D); mat("o", D, D)
            vec("n2"); mat("w1", F, D); mat("w3", F, D); mat("w2", D, F)
            self.layers.append(lay)
        self.norm = b[on:on + 4 * D].view(np.float32).copy()
        hd = D // self.H
        inv = 1.0 / (10000 ** (np.arange(0, hd, 2) / hd))
        f = np.outer(np.arange(self.C), inv)
        self.cos, self.sin = np.cos(f), np.sin(f)

    @staticmethod
    def q8(x):
        """Per (position, head) int8 with scale max|x|/127, round-half-even (C: lrintf)."""
        mx = np.abs(x).max(-1, keepdims=True)
        inv = np.where(mx > 0, 127.0 / np.where(mx > 0, mx, 1), 0).astype(np.float32)
        return np.rint(x * inv).astype(np.float32) * (mx / 127.0).astype(np.float32)

    @staticmethod
    def rms(x, w): return w * x / np.sqrt((x * x).mean(-1, keepdims=True) + NORM_EPS)

    def rope(self, x, pos):                              # x [T, H, hd]
        c, s = self.cos[pos][:, None], self.sin[pos][:, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return np.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).reshape(x.shape)

    def mv(self, lay, name, x):
        """One weight matvec, through whichever kernel contract this reference models."""
        if self.act == "i8":
            return matvec_i8(*lay[name + "_i8"], x)
        return x @ lay[name].T

    def logits(self, ids):
        T, D, H = len(ids), self.D, self.H
        hd = D // H
        h = self.tok[ids]
        pos = np.arange(T)
        for l, lay in enumerate(self.layers):
            h = h + self.mv(lay, "ple_proj", self.ple[ids][:, l * self.P:(l + 1) * self.P])
            qkv = self.mv(lay, "qkv", self.rms(h, lay["n1"]))
            q, k, v = (qkv[:, i * D:(i + 1) * D].reshape(T, H, hd) for i in range(3))
            q, k = self.rope(q, pos), self.rope(k, pos)
            k, v = self.q8(k), self.q8(v)                # int8 KV cache, as in core/ple.c
            att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
            att += np.triu(np.full((T, T), -1e9), 1)[None]
            att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
            a = np.einsum("hts,shd->thd", att, v).reshape(T, D)
            h = h + self.mv(lay, "o", a)
            x = self.rms(h, lay["n2"])
            s = self.mv(lay, "w1", x)
            h = h + self.mv(lay, "w2", (s / (1 + np.exp(-s))) * self.mv(lay, "w3", x))
        x = self.rms(h, self.norm)
        if self.act == "i8":
            return matvec_i8(*self.tok_i8, x)
        return x @ self.tok.T

    def greedy(self, ids, n):
        ids = list(ids)
        for _ in range(n):
            ids.append(int(self.logits(np.array(ids[-self.C:])).argmax(-1)[-1]))
        return ids


# ---------------------------------------------------------------- regolden (numpy only)
def decode_bin(tok_bin: str, ids):
    """Decode token ids with our own tokenizer.bin (no tokenizers library needed)."""
    b = open(tok_bin, "rb").read()
    n = struct.unpack_from("<I", b)[0]
    o, table = 4, []
    for _ in range(n):
        ln = b[o]; table.append(b[o + 1:o + 1 + ln]); o += 1 + ln
    return b"".join(table[i] for i in ids).decode("utf-8", "replace")


def golden_name(act: str) -> str:
    return "golden.txt" if act == "f32" else f"golden_{act}.txt"


def regolden(model_dir: str, n_tokens: int, act: str = "f32"):
    """Rewrite MODEL_DIR/golden.txt (act f32) or golden_i8.txt (act i8) from an existing
    model.bin with the current reference. Takes the prompt ids from golden.txt, so neither
    torch nor tokenizers is needed. For i8 it also compares against the float golden."""
    mb, tb = os.path.join(model_dir, "model.bin"), os.path.join(model_dir, "tokenizer.bin")
    base = open(os.path.join(model_dir, "golden.txt")).read().split("\n")
    ids = [int(t) for t in base[0].split()]
    out = os.path.join(model_dir, golden_name(act))
    prev = []
    if os.path.exists(out):
        old = open(out).read().split("\n")
        prev = [int(t) for t in old[1].split()] if len(old) > 1 else []
    gen = Ref(mb, act).greedy(ids, n_tokens)[len(ids):]
    if os.path.exists(tb):
        print(f"greedy (int4 weights, {act} activations, int8 KV):", decode_bin(tb, ids + gen))
    changed = sum(1 for x, y in zip(prev, gen) if x != y)
    with open(out, "w", newline="\n") as fh:
        fh.write(" ".join(map(str, ids)) + "\n" + " ".join(map(str, gen)) + "\n")
    print(f"{out}: {len(ids)} prompt + {len(gen)} tokens, "
          f"{changed} of {min(len(prev), len(gen))} differ from the previous {golden_name(act)}")
    if act != "f32":
        f = [int(t) for t in base[1].split()]
        first = next((i for i, (x, y) in enumerate(zip(f, gen)) if x != y), None)
        print(f"vs float golden: {sum(1 for x, y in zip(f, gen) if x == y)} of {min(len(f), len(gen))} "
              f"tokens equal, first divergence at {'none' if first is None else first}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ple_d0wd96.pt")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out", default="model.bin")
    ap.add_argument("--check", action="store_true", help="run the numpy reference on the exported file")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--golden_tokens", type=int, default=48)
    ap.add_argument("--regolden", metavar="MODEL_DIR",
                    help="only rewrite MODEL_DIR/golden.txt from MODEL_DIR/model.bin (numpy only)")
    ap.add_argument("--act", choices=("f32", "i8"), default="f32",
                    help="with --regolden: f32 -> golden.txt, i8 -> golden_i8.txt (int8 activations)")
    a = ap.parse_args()

    if a.regolden:
        regolden(a.regolden, a.golden_tokens, a.act)
        return

    import torch
    ck = torch.load(a.ckpt, map_location="cpu")
    state = {k: v.numpy() for k, v in ck["state"].items() if k not in ("cos", "sin")}
    total = export(state, ck["cfg"], a.out)
    if total > 0x2F0000:
        sys.exit(f"ERROR: {total} B exceeds the 0x2F0000 model partition")
    export_tokenizer(a.tokenizer, "tokenizer.bin")

    if a.check:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tokenizer)
        ref = Ref(a.out)
        ids = tk.encode(a.prompt).ids
        gen = ref.greedy(ids, a.golden_tokens)
        print("greedy (int4 reference):", tk.decode(gen))
        with open("golden.txt", "w") as fh:
            fh.write(" ".join(map(str, ids)) + "\n" + " ".join(map(str, gen[len(ids):])) + "\n")
        print(f"golden.txt: {len(ids)} prompt tokens + {a.golden_tokens} greedy tokens")
        gen8 = Ref(a.out, "i8").greedy(ids, a.golden_tokens)
        with open("golden_i8.txt", "w") as fh:
            fh.write(" ".join(map(str, ids)) + "\n" + " ".join(map(str, gen8[len(ids):])) + "\n")
        print("greedy (int8 activations):", tk.decode(gen8))
        print(f"golden_i8.txt: {sum(1 for x, y in zip(gen[len(ids):], gen8[len(ids):]) if x == y)} "
              f"of {a.golden_tokens} tokens equal to the float golden")


if __name__ == "__main__":
    main()
