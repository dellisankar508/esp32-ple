"""
Export ple_d0wd96.pt -> model.bin (PLE1, int4 group-32) + tokenizer.bin + golden.txt

    python export_ple.py                 # uses ple_d0wd96.pt + tokenizer.json in cwd
    python export_ple.py --check         # also runs a numpy reference forward on the exported file

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
    """Dequantized reference. Same math the C runtime must reproduce, int8 KV cache included."""
    def __init__(self, path):
        b = np.fromfile(path, np.uint8)
        assert b[:4].tobytes() == b"PLE1"
        (self.V, self.D, self.L, self.H, self.F, self.P, self.C, g) = struct.unpack_from("<8I", b, 8)
        ot, op, ol, on, tot = struct.unpack_from("<5I", b, 40)
        assert tot == len(b) and g == GROUP
        V, D, L, F, P = self.V, self.D, self.L, self.F, self.P
        rb = lambda inn: align4(inn // 2 + inn // 16)
        self.tok = dequant_rows(b[ot:ot + V * rb(D)], V, D)
        self.ple = dequant_rows(b[op:op + V * rb(L * P)], V, L * P)
        self.layers = []
        o = ol
        for _ in range(L):
            lay = {}
            def mat(name, out, inn):
                nonlocal o
                n = out * rb(inn); lay[name] = dequant_rows(b[o:o + n], out, inn); o += n
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

    def logits(self, ids):
        T, D, H = len(ids), self.D, self.H
        hd = D // H
        h = self.tok[ids]
        pos = np.arange(T)
        for l, lay in enumerate(self.layers):
            h = h + self.ple[ids][:, l * self.P:(l + 1) * self.P] @ lay["ple_proj"].T
            qkv = self.rms(h, lay["n1"]) @ lay["qkv"].T
            q, k, v = (qkv[:, i * D:(i + 1) * D].reshape(T, H, hd) for i in range(3))
            q, k = self.rope(q, pos), self.rope(k, pos)
            k, v = self.q8(k), self.q8(v)                # int8 KV cache, as in core/ple.c
            att = np.einsum("thd,shd->hts", q, k) / np.sqrt(hd)
            att += np.triu(np.full((T, T), -1e9), 1)[None]
            att = np.exp(att - att.max(-1, keepdims=True)); att /= att.sum(-1, keepdims=True)
            a = np.einsum("hts,shd->thd", att, v).reshape(T, D)
            h = h + a @ lay["o"].T
            x = self.rms(h, lay["n2"])
            s = x @ lay["w1"].T
            h = h + ((s / (1 + np.exp(-s))) * (x @ lay["w3"].T)) @ lay["w2"].T
        return self.rms(h, self.norm) @ self.tok.T

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


def regolden(model_dir: str, n_tokens: int):
    """Rewrite golden.txt from an existing model.bin with the current reference.
    Reuses the prompt ids already in golden.txt, so neither torch nor tokenizers is needed."""
    mb, gt = os.path.join(model_dir, "model.bin"), os.path.join(model_dir, "golden.txt")
    tb = os.path.join(model_dir, "tokenizer.bin")
    old = open(gt).read().split("\n")
    ids = [int(t) for t in old[0].split()]
    prev = [int(t) for t in old[1].split()] if len(old) > 1 else []
    gen = Ref(mb).greedy(ids, n_tokens)[len(ids):]
    if os.path.exists(tb):
        print("greedy (int4 reference, int8 KV):", decode_bin(tb, ids + gen))
    changed = sum(1 for x, y in zip(prev, gen) if x != y)
    with open(gt, "w", newline="\n") as fh:
        fh.write(" ".join(map(str, ids)) + "\n" + " ".join(map(str, gen)) + "\n")
    print(f"{gt}: {len(ids)} prompt + {len(gen)} tokens, "
          f"{changed} of {min(len(prev), len(gen))} differ from the old golden")


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
    a = ap.parse_args()

    if a.regolden:
        regolden(a.regolden, a.golden_tokens)
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


if __name__ == "__main__":
    main()
