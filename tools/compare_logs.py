"""
Turn boot logs from the shared benchmark into one comparison table (Markdown).

    python3 tools/compare_logs.py esp32.log pico2w.log > comparison.md

Save each board's serial output to a file first, e.g.
    idf.py monitor | tee esp32.log
    python3 -m serial.tools.miniterm /dev/ttyACM0 115200 | tee pico2w.log
Only the tagged lines (ENV, MODEL, FLASH, ...) are read; everything else is ignored.
If a log holds several benchmark runs ('b' on the Pico), the last one is used.
"""
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUM = r"([0-9.]+)"


def parse(path):
    # Logs copied out of a terminal window often lose their line breaks (each line
    # padded with spaces instead), so no pattern below relies on a newline.
    text = ANSI.sub("", open(path, encoding="utf-8", errors="replace").read()).replace("\r", "")
    blocks = text.split("=== PLE bench ===")
    if len(blocks) < 2:
        sys.exit(f"{path}: no '=== PLE bench ===' section; is this a log from the new firmware?")
    b = blocks[-1]
    d = {"file": path}

    def grab(pattern, keys, cast=float):
        m = re.search(pattern, b)
        if m:
            for k, v in zip(keys, m.groups()):
                d[k] = cast(v) if cast and v is not None else v

    grab(r"ENV \| board (\S+) \| cpu " + NUM + r" MHz \| flash (.+?) \| compiler (.+?) \| kernel (\S+) \| ctx (\d+)",
         ["board", "mhz", "flash", "compiler", "kernel", "ctx"], cast=None)
    d["mhz"] = float(d.get("mhz") or 0)
    grab(r"MODEL \| fnv1a (0x[0-9a-f]+) \| (\d+) bytes \| state (\d+) B \| (\d+) B/tok",
         ["fnv", "model_bytes", "state", "bpt"], cast=None)
    grab(r"FLASH \| sequential " + NUM + r" MB/s \| random 216 B row " + NUM + " us", ["seq", "row_us"])
    grab(r"FLASH2 \| 1 core " + NUM + r" MB/s \| 2 cores together " + NUM + r" MB/s \| " + NUM + "x",
         ["f2_one", "f2_two", "f2_ratio"])
    grab(r"FLOOR \| embed " + NUM + " attn " + NUM + " ffn " + NUM + " head " + NUM + r" ms \| total " + NUM,
         ["fl_embed", "fl_attn", "fl_ffn", "fl_head", "fl_total"])
    # Plain configs, then the ones with a platform head hook ("1-core+dma", ...).
    hook = re.search(r"RESULT \| \S+ 1-core\+(\S+) \|", b)
    d["hook"] = hook.group(1) if hook else None
    cfgs = [("1-core", "1"), ("2-core", "2")]
    if hook:
        cfgs += [("1-core+" + hook.group(1), "1x"), ("2-core+" + hook.group(1), "2x")]
    for cfg, tag in cfgs:
        c = re.escape(cfg)
        m = re.search(r"GOLDEN: (\w+) \((\d+) mismatches, (\d+) ties\) \| \S+ " + c + r" \|", b)
        if m:
            d[f"g{tag}"] = f"{m.group(1)} ({m.group(2)} mismatches, {m.group(3)} ties)"
        grab(r"RESULT \| \S+ " + c + r" \| kernel \S+ \| \d+ tok \| " + NUM + r" ms/tok \| " + NUM +
             r" tok/s \| embed " + NUM + " attn " + NUM + " ffn " + NUM + " head " + NUM +
             r" ms \| median of (\d+), range " + NUM + "-" + NUM + r" ms \| " + NUM + " Mcycles",
             [f"ms{tag}", f"tps{tag}", f"embed{tag}", f"attn{tag}", f"ffn{tag}", f"head{tag}",
              f"reps{tag}", f"min{tag}", f"max{tag}", f"mcyc{tag}"])
    grab(r"HEADSRAM \| \S+ \| from SRAM: 1 core " + NUM + " ms(?:, 2 cores " + NUM + " ms, " + NUM + "x)?",
         ["hs1", "hs2", "hs_ratio"])
    grab(r"from flash: 1 core " + NUM + " ms(?:, 2 cores " + NUM + " ms, " + NUM + "x)?",
         ["hf1", "hf2", "hf_ratio"])
    grab(r"HEADSTREAM \| \S+ \S+ \| move only " + NUM + r" ms \(" + NUM + r" MB/s\) \| 1 core " + NUM +
         r" ms, 2 cores " + NUM + r" ms \| compute only " + NUM + " / " + NUM + r" ms \| (.*?)(?:\s{2,}|\n|$)",
         ["mv_ms", "mv_mbps", "hx1", "hx2", "hc1", "hc2", "hdesc"], cast=None)
    for k in ("mv_ms", "mv_mbps", "hx1", "hx2", "hc1", "hc2"):
        if k in d:
            d[k] = float(d[k])
    grab(r"LONGCTX \| \S+ (\S+) \| pos \S+ \| " + NUM + r" ms/tok \| " + NUM + r" tok/s \| attn " + NUM + " ms",
         ["lc_cfg", "lc_ms", "lc_tps", "lc_attn"], cast=None)
    for k in ("lc_ms", "lc_tps", "lc_attn"):
        if k in d:
            d[k] = float(d[k])
    return d


def f(v, fmt="{:.1f}"):
    return "–" if v is None else fmt.format(v)


def x_floor(ms, floor):
    if ms is None or not floor:
        return f(ms, "{:.2f}")
    return f"{ms:.2f} ({ms / floor:.1f}x floor)"


def main(paths):
    logs = [parse(p) for p in paths]
    g = lambda d, k: d.get(k)
    rows = [
        ("**Setup**", None),
        ("CPU clock (MHz)", lambda d: f(g(d, "mhz"), "{:.0f}")),
        ("Flash interface", lambda d: g(d, "flash") or "–"),
        ("Compiler, flags", lambda d: g(d, "compiler") or "–"),
        ("Kernel", lambda d: g(d, "kernel") or "–"),
        ("Model checksum", lambda d: g(d, "fnv") or "–"),
        ("State RAM (B)", lambda d: g(d, "state") or "–"),
        ("**Correctness**", None),
        ("Golden, 1 core", lambda d: g(d, "g1") or "–"),
        ("Golden, 2 cores", lambda d: g(d, "g2") or "–"),
        ("**Flash**", None),
        ("Sequential read (MB/s)", lambda d: f(g(d, "seq"))),
        ("Random 216 B row (µs)", lambda d: f(g(d, "row_us"))),
        ("Same bytes, 1 core vs 2 at once (MB/s)",
         lambda d: f"{d['f2_one']:.1f} vs {d['f2_two']:.1f} ({d['f2_ratio']:.2f}x)" if "f2_one" in d else "–"),
        ("Bytes per token", lambda d: g(d, "bpt") or "–"),
        ("Flash floor per token (ms)", lambda d: f(g(d, "fl_total"))),
        ("**Speed (median of runs)**", None),
        ("ms/token, 1 core", lambda d: f"{d['ms1']:.1f} ({d['min1']:.1f}-{d['max1']:.1f})" if "ms1" in d else "–"),
        ("ms/token, 2 cores", lambda d: f"{d['ms2']:.1f} ({d['min2']:.1f}-{d['max2']:.1f})" if "ms2" in d else "–"),
        ("tok/s, 2 cores", lambda d: f(g(d, "tps2"), "{:.2f}")),
        ("Measured / floor, 2 cores",
         lambda d: f"{d['ms2'] / d['fl_total']:.1f}x" if "ms2" in d and d.get("fl_total") else "–"),
        ("Mcycles/token, 1 core (clock-independent)", lambda d: f(g(d, "mcyc1"), "{:.2f}")),
        ("**Per stage, 2 cores (ms)**", None),
        ("Embed", lambda d: x_floor(g(d, "embed2"), g(d, "fl_embed"))),
        ("Attention", lambda d: x_floor(g(d, "attn2"), g(d, "fl_attn"))),
        ("FFN", lambda d: x_floor(g(d, "ffn2"), g(d, "fl_ffn"))),
        ("Head", lambda d: x_floor(g(d, "head2"), g(d, "fl_head"))),
        ("**Head: compute vs flash (ms)**", None),
        ("Head from SRAM, 1 core / 2 cores",
         lambda d: f"{d['hs1']:.2f} / {d['hs2']:.2f} ({d['hs_ratio']:.2f}x)" if g(d, "hs2") else f(g(d, "hs1"), "{:.2f}")),
        ("Head from flash, 1 core / 2 cores",
         lambda d: f"{d['hf1']:.2f} / {d['hf2']:.2f} ({d['hf_ratio']:.2f}x)" if g(d, "hf2") else f(g(d, "hf1"), "{:.2f}")),
        ("Flash cost in head, 2 cores (flash - SRAM)",
         lambda d: f"{d['hf2'] - d['hs2']:.2f}" if g(d, "hf2") and g(d, "hs2") else "–"),
        ("**Head streamed by the platform (DMA on the Pico)**", None),
        ("How", lambda d: (d.get("hdesc") or "–").strip().replace("|", ";") if d.get("hook") else "–"),
        ("Move head only, no math (ms, MB/s)",
         lambda d: f"{d['mv_ms']:.2f} ({d['mv_mbps']:.1f} MB/s)" if "mv_ms" in d else "–"),
        ("Golden, 1 core / 2 cores",
         lambda d: f"{g(d, 'g1x') or '–'} / {g(d, 'g2x') or '–'}" if d.get("hook") else "–"),
        ("Head, 1 core / 2 cores (ms)",
         lambda d: f"{d['hx1']:.2f} / {d['hx2']:.2f}" if "hx1" in d else "–"),
        ("ms/token, 1 core / 2 cores",
         lambda d: f"{d['ms1x']:.1f} / {d['ms2x']:.1f}" if "ms1x" in d and "ms2x" in d else "–"),
        ("tok/s, 2 cores", lambda d: f(g(d, "tps2x"), "{:.2f}")),
        ("Speed-up vs plain 2 cores",
         lambda d: f"{d['ms2'] / d['ms2x']:.2f}x" if "ms2x" in d and "ms2" in d else "–"),
        ("**Long context (positions 100-119)**", None),
        ("Config", lambda d: g(d, "lc_cfg") or "–"),
        ("ms/token", lambda d: f(g(d, "lc_ms"))),
        ("Attention (ms), here vs 51-token average",
         lambda d: f"{d['lc_attn']:.2f} vs {d.get('attn2x', d['attn2']):.2f}" if "lc_attn" in d and "attn2" in d else f(g(d, "lc_attn"), "{:.2f}")),
    ]
    names = [d.get("board", d["file"]) for d in logs]
    print("| Metric | " + " | ".join(names) + " |")
    print("| --- | " + " | ".join("---" for _ in names) + " |")
    for label, fn in rows:
        if fn is None:
            print(f"| {label} | " + " | ".join("" for _ in names) + " |")
        else:
            print(f"| {label} | " + " | ".join(fn(d) for d in logs) + " |")
    fnvs = {d.get("fnv") for d in logs}
    print()
    print("Same model on every board: " + ("yes" if len(fnvs) == 1 and None not in fnvs else "NO, checksums differ"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
