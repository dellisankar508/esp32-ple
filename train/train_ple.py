"""
Minimal Per-Layer-Embedding (PLE) language model, trained from scratch on TinyStories.

    pip install torch datasets tokenizers
    python train_ple_example.py            # ~20-40 min on a single GPU, runs on CPU too (slowly)

The PLE idea in one line:
    h_l = h_l + proj_l( PLE_l[token] )     # PLE_l is a [vocab, ple_dim] lookup table per layer
Everything else is a plain decoder-only transformer (RMSNorm, RoPE, SwiGLU, tied head).
"""
import math, time, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

# ---------------- config: "D0WD-96", sized for ESP32 4 MB flash ----------------
@dataclass
class Cfg:
    vocab: int = 8192
    d_model: int = 96
    n_layers: int = 6
    n_heads: int = 4
    ffn_hidden: int = 128
    ple_dim: int = 64
    ctx: int = 128

# ---------------- model ----------------
class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Parameter(torch.ones(d))
    def forward(self, x): return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

def rope(x, cos, sin):                       # x: [B, H, T, hd]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.ple_proj = nn.Linear(c.ple_dim, c.d_model, bias=False)      # <- PLE injection
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.o = nn.Linear(c.d_model, c.d_model, bias=False)
        self.w1 = nn.Linear(c.d_model, c.ffn_hidden, bias=False)          # SwiGLU
        self.w3 = nn.Linear(c.d_model, c.ffn_hidden, bias=False)
        self.w2 = nn.Linear(c.ffn_hidden, c.d_model, bias=False)

    def forward(self, h, ple_row, cos, sin):
        B, T, D = h.shape
        H, hd = self.c.n_heads, D // self.c.n_heads
        h = h + self.ple_proj(ple_row)                                   # add this layer's PLE
        q, k, v = self.qkv(self.n1(h)).view(B, T, 3, H, hd).transpose(1, 3).unbind(2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.o(a.transpose(1, 2).reshape(B, T, D))
        x = self.n2(h)
        return h + self.w2(F.silu(self.w1(x)) * self.w3(x))

class PLELM(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d_model)                        # normal token embedding
        self.ple = nn.Embedding(c.vocab, c.n_layers * c.ple_dim)           # ONE big table, sliced per layer
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model)
        hd = c.d_model // c.n_heads
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        t = torch.arange(c.ctx).float()
        f = torch.outer(t, inv)
        self.register_buffer("cos", f.cos()[None, None]); self.register_buffer("sin", f.sin()[None, None])

    def forward(self, idx):
        B, T = idx.shape
        h = self.tok(idx)
        ple = self.ple(idx).view(B, T, self.c.n_layers, self.c.ple_dim)    # one lookup, no matmul
        cos, sin = self.cos[:, :, :T], self.sin[:, :, :T]
        for l, blk in enumerate(self.blocks):
            h = blk(h, ple[:, :, l], cos, sin)
        return self.norm(h) @ self.tok.weight.T                            # tied head

    def param_report(self):
        ple = self.ple.weight.numel(); tok = self.tok.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        core = total - ple - tok
        print(f"total {total/1e6:.2f}M | PLE table {ple/1e6:.2f}M ({100*ple/total:.0f}%) "
              f"| token emb/head {tok/1e6:.2f}M | dense core {core/1e3:.0f}K")

# ---------------- data ----------------
def build_data(c: Cfg, n_train_stories=200_000):
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    ds = load_dataset("roneneldan/TinyStories", split="train")
    texts = [ds[i]["text"] for i in range(n_train_stories)]

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=c.vocab, special_tokens=["<eos>"]))
    tok.save("tokenizer.json")

    eos = tok.token_to_id("<eos>")
    ids = []
    for t in texts:
        ids.extend(tok.encode(t).ids); ids.append(eos)
    return torch.tensor(ids, dtype=torch.long), tok

def batches(data, c: Cfg, bs, device):
    while True:
        ix = torch.randint(0, len(data) - c.ctx - 1, (bs,))
        x = torch.stack([data[i:i + c.ctx] for i in ix])
        y = torch.stack([data[i + 1:i + c.ctx + 1] for i in ix])
        yield x.to(device), y.to(device)

# ---------------- train ----------------
def main():
    c = Cfg()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data, tok = build_data(c)
    model = PLELM(c).to(dev)
    model.param_report()

    steps, bs, lr = 6000, 64, 3e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    gen = batches(data, c, bs, dev)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = next(gen)
        loss = F.cross_entropy(model(x).view(-1, c.vocab), y.view(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            print(f"step {step:5d} loss {loss.item():.3f} ppl {math.exp(loss.item()):.1f} {time.time()-t0:.0f}s")
    torch.save({"cfg": c.__dict__, "state": model.state_dict()}, "ple_d0wd96.pt")

    # sample
    model.eval()
    ids = torch.tensor([tok.encode("Once upon a time").ids], device=dev)
    with torch.no_grad():
        for _ in range(80):
            logits = model(ids[:, -c.ctx:])[:, -1] / 0.8
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            ids = torch.cat([ids, nxt], 1)
    print(tok.decode(ids[0].tolist()))

if __name__ == "__main__":
    main()
