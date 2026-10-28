"""
Kernel test vectors for the int8-activation contract (I8 v1, train/export_ple.py).

Takes real weight matrices from a model.bin, feeds each a set of input vectors
(ordinary, outliers, an all-zero group, exact .5 rounding ties, tiny values) and
stores the numpy answer. core/kern_check.c runs a kernel on the same inputs and
compares every output float bit for bit.

    python tools/kernel_vectors.py model/model.bin model/kvec.bin

kvec.bin (little-endian): "KVC1", u32 n_cases, then per case:
    char name[24], u32 offset (bytes from the start of model.bin), u32 rows, u32 in,
    f32 x[in], f32 y[rows]
"""
import os, struct, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
import export_ple as E   # noqa: E402

F32 = np.float32


def matrices(b):
    """(name, offset, rows, in) of the weight matrices exercised, in model.bin order."""
    V, D, L, H, F, P, C, G = struct.unpack_from("<8I", b, 8)
    ot, op, ol, on, tot = struct.unpack_from("<5I", b, 40)
    rb = lambda inn: E.align4(inn // 2 + inn // 16)
    out, o = [], ol
    for l in range(L):
        lay = {}
        for name, rows, inn in (("ple_proj", D, P), ("n1", 0, 0), ("qkv", 3 * D, D), ("o", D, D),
                                ("n2", 0, 0), ("w1", F, D), ("w3", F, D), ("w2", D, F)):
            if rows == 0:
                o += 4 * D
                continue
            lay[name] = (o, rows, inn)
            o += rows * rb(inn)
        if l == 0: out += [("L0.ple_proj",) + lay["ple_proj"], ("L0.qkv",) + lay["qkv"]]
        if l == 1: out += [("L1.o",) + lay["o"], ("L1.w1",) + lay["w1"]]
        if l == L - 1: out += [(f"L{l}.w3",) + lay["w3"], (f"L{l}.w2",) + lay["w2"]]
    out.append(("head",) + (ot, V, D))
    return out, rb


def inputs(inn, rng):
    """Named input vectors that poke at every step of the contract."""
    ng = inn // E.GROUP
    xs = [("normal", rng.normal(0, 1, inn)),
          ("outliers", np.where(rng.random(inn) < 0.03, rng.normal(0, 30, inn), rng.normal(0, 0.1, inn)))]
    z = rng.normal(0, 1, inn); z[E.GROUP:2 * E.GROUP if ng > 1 else E.GROUP] = 0
    if ng == 1: z[:] = 0
    xs.append(("zero-group", z))
    # Each group's max is exactly 127, so inv = 1 and x * inv = x: values k + 0.5 are
    # exact rounding ties, which must go to the even neighbour (2.5 -> 2, -3.5 -> -4).
    t = (rng.integers(-60, 60, inn) + 0.5)
    t[::E.GROUP] = 127.0
    xs.append(("ties", t))
    xs.append(("tiny", rng.normal(0, 1, inn) * 1e-16))
    xs.append(("vanishing", rng.normal(0, 1, inn) * 1e-21))   # below 2^-60: a zero group
    return [(n, x.astype(F32)) for n, x in xs]


def main():
    mpath, kpath = sys.argv[1], sys.argv[2]
    b = np.fromfile(mpath, np.uint8)
    mats, rb = matrices(b)
    rng = np.random.default_rng(7)
    blob, n = b"", 0
    for name, off, rows, inn in mats:
        q, s = E.dequant_parts(b[off:off + rows * rb(inn)], rows, inn)
        for kind, x in inputs(inn, rng):
            y = E.matvec_i8(q, s, x[None, :])[0]
            tag = f"{name}/{kind}"[:23].encode()
            blob += struct.pack("<24s3I", tag, off, rows, inn) + x.tobytes() + y.astype(F32).tobytes()
            n += 1
    open(kpath, "wb").write(b"KVC1" + struct.pack("<I", n) + blob)
    print(f"{kpath}: {n} cases over {len(mats)} matrices, {len(blob) // 1024} KB")


if __name__ == "__main__":
    main()
