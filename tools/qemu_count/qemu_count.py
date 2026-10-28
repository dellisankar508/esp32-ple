"""
Exact instruction counts for a platform/corstone300 ELF, in QEMU with the insncount
plugin (tools/qemu_count/insncount.c, built on first use).

    python3 tools/qemu_count/qemu_count.py --machine mps3-an547 build/sse300-mve.elf

Regions come from the ple_mark_* functions in platform/corstone300/main.c:
  token  one ple_forward() each (all golden tokens)    head  one 8192 x 96 head matvec
"kernel" = instructions inside kern_matvec / kern_prepare (the linked kernel).
QEMU is not cycle-accurate: these numbers compare kernels, they are not timings.
Needs QEMU's include/qemu/qemu-plugin.h for the same QEMU version: set QEMU_PLUGIN_INC.
"""
import argparse, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def symbols(elf):
    out = subprocess.run(["arm-none-eabi-nm", "-S", elf], capture_output=True, text=True, check=True).stdout
    syms = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4:
            syms[p[3]] = (int(p[0], 16) & ~1, int(p[1], 16))
        elif len(p) == 3 and p[2] not in syms:
            syms[p[2]] = (int(p[0], 16) & ~1, 0)
    return syms


def build_plugin():
    so = os.path.join(HERE, "libinsncount.so")
    src = os.path.join(HERE, "insncount.c")
    if os.path.exists(so) and os.path.getmtime(so) > os.path.getmtime(src):
        return so
    inc = os.environ.get("QEMU_PLUGIN_INC")
    if not inc:
        sys.exit("set QEMU_PLUGIN_INC to the directory holding qemu-plugin.h (QEMU source: include/qemu)")
    glib = subprocess.run(["pkg-config", "--cflags", "glib-2.0"], capture_output=True, text=True, check=True).stdout.split()
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-I" + inc] + glib + ["-o", so, src], check=True)
    return so


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("elf")
    ap.add_argument("--machine", default="mps3-an547")
    a = ap.parse_args()
    s = symbols(a.elf)
    args = []
    for name, b, e in (("token", "ple_mark_tok_begin", "ple_mark_tok_end"),
                       ("head", "ple_mark_head_begin", "ple_mark_head_end")):
        args.append(f"region={name}:{s[b][0]:#x}:{s[e][0]:#x}")
    for f in ("kern_matvec", "kern_prepare"):
        lo, size = s[f]
        args.append(f"kernel={lo:#x}:{lo + size:#x}")
    plugin = build_plugin() + "," + ",".join(args)
    r = subprocess.run(["qemu-system-arm", "-M", a.machine, "-display", "none", "-serial", "null",
                        "-monitor", "none", "-semihosting-config", "enable=on,target=native",
                        "-plugin", plugin, "-d", "plugin", "-kernel", a.elf], capture_output=True, text=True, timeout=3600)
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if line.startswith(("KERNEL", "GOLDEN", "LOGITS", "===")):
            print(line)
    res = {}
    for m in re.finditer(r"QCOUNT \| (\w+) \| runs (\d+) \| insns (\d+) \| kernel (\d+) \| mve (\d+)", out):
        res[m.group(1)] = tuple(int(x) for x in m.group(2, 3, 4, 5))
    if "token" not in res:
        sys.exit("no counts; QEMU output:\n" + out[-2000:])
    kern = re.search(r"kernel (\w+) ===", out)
    kname = kern.group(1) if kern else "?"
    n, t, k, v = res["token"]
    hn, ht, hk, hv = res["head"]
    print(f"COUNT | {a.machine} | kernel {kname} | per token: {t / n:.0f} insns, {k / n:.0f} in the kernel, "
          f"{v / n:.0f} MVE | head matvec: {ht} insns, {hk} in the kernel, {hv} MVE")


if __name__ == "__main__":
    main()
