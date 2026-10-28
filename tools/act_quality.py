"""
What do int8 activations cost? Compares the float-activation and int8-activation
references (train/export_ple.py, Ref act="f32" vs act="i8") on the same text.

Stories are sampled from the float model, then both references score every position
of them (teacher forcing), so they always see identical inputs:
  - top-1 agreement: how often both pick the same next token
  - KL(f32 || i8):   how far the int8 next-token distribution moves, in nats
  - perplexity of the text under each (the text comes from the float model, which
    gives it a small home advantage)

    python tools/act_quality.py model/model.bin [--stories 8] [--tokens 120]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
import export_ple as E   # noqa: E402


def sample(ref, ids, n, temp, rng):
    ids = list(ids)
    for _ in range(n):
        lg = ref.logits(np.array(ids[-ref.C:]))[-1].astype(np.float64) / temp
        p = np.exp(lg - lg.max()); p /= p.sum()
        ids.append(int(rng.choice(len(p), p=p)))
    return ids


def logp(lg):
    lg = lg.astype(np.float64)
    return lg - lg.max(-1, keepdims=True) - np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--stories", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--prompt", default="297 312 124 261", help="prompt token ids ('Once upon a time')")
    a = ap.parse_args()
    f, q = E.Ref(a.model, "f32"), E.Ref(a.model, "i8")
    prompt = [int(t) for t in a.prompt.split()]
    rng = np.random.default_rng(0)
    agree = total = 0
    kl, nll_f, nll_q = [], [], []
    for s in range(a.stories):
        ids = np.array(sample(f, prompt, a.tokens, 0.8, rng)[:f.C])
        lf, lq = logp(f.logits(ids[:-1])), logp(q.logits(ids[:-1]))
        nxt = ids[1:]
        agree += int((lf.argmax(-1) == lq.argmax(-1)).sum()); total += len(nxt)
        kl.extend((np.exp(lf) * (lf - lq)).sum(-1))
        nll_f.extend(-lf[np.arange(len(nxt)), nxt]); nll_q.extend(-lq[np.arange(len(nxt)), nxt])
        print(f"story {s + 1}: top-1 agreement {100 * agree / total:.2f}% so far")
    print(f"ACTQ | {a.stories} stories, {total} positions | top-1 agreement {100 * agree / total:.2f}% | "
          f"mean KL(f32||i8) {np.mean(kl):.5f} nats, max {np.max(kl):.4f} | "
          f"perplexity f32 {np.exp(np.mean(nll_f)):.3f}, i8 {np.exp(np.mean(nll_q)):.3f}")


if __name__ == "__main__":
    main()
