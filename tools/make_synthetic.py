"""
Build a synthetic PLE1 model with random weights, plus a tokenizer.bin and a
golden.txt from the numpy int4 reference. Lets `make test` check the C runtime
end to end without the trained model or torch.

    python tools/make_synthetic.py out_dir
"""
import os, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
import export_ple as E   # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)

cfg = dict(vocab=8192, d_model=96, n_layers=6, n_heads=4, ffn_hidden=128, ple_dim=64, ctx=128)
V, D, L, F, P = 8192, 96, 6, 128, 64
r = np.random.default_rng(1)
st = {"tok.weight": r.normal(0, .3, (V, D)), "ple.weight": r.normal(0, .3, (V, L * P)),
      "norm.w": r.uniform(.5, 1.5, D)}
for l in range(L):
    p = f"blocks.{l}."
    st[p + "ple_proj.weight"] = r.normal(0, .3, (D, P))
    st[p + "n1.w"] = r.uniform(.5, 1.5, D); st[p + "n2.w"] = r.uniform(.5, 1.5, D)
    st[p + "qkv.weight"] = r.normal(0, .3, (3 * D, D)); st[p + "o.weight"] = r.normal(0, .3, (D, D))
    st[p + "w1.weight"] = r.normal(0, .3, (F, D)); st[p + "w3.weight"] = r.normal(0, .3, (F, D))
    st[p + "w2.weight"] = r.normal(0, .3, (D, F))
st = {k: v.astype(np.float32) for k, v in st.items()}
E.export(st, cfg, os.path.join(out, "model.bin"))

tok = struct.pack("<I", V)                       # token i decodes to "i "
for i in range(V):
    b = (str(i) + " ").encode()
    tok += struct.pack("<B", len(b)) + b
open(os.path.join(out, "tokenizer.bin"), "wb").write(tok)

ref = E.Ref(os.path.join(out, "model.bin"))
ids = [5, 17, 300, 4000]
gen = ref.greedy(ids, 40)
with open(os.path.join(out, "golden.txt"), "w") as fh:
    fh.write(" ".join(map(str, ids)) + "\n" + " ".join(map(str, gen[len(ids):])) + "\n")
print(f"synthetic model, tokenizer and golden ({len(gen) - len(ids)} tokens) in {out}")
