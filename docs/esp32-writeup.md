# A 4.4M-parameter LLM on a classic ESP32 with no PSRAM

The board is an ESP32-D0WD-V3: two Xtensa cores at 240 MHz, 520 KB of SRAM, 4 MB
of flash, no PSRAM and no vector unit. On it, a language model I trained from
scratch writes short stories at **15.9 tokens per second**, and every token it
produces matches a Python reference of the same model exactly.

The whole design comes from one decision: **the weights never enter RAM.** The
model lives in flash and is read again for every token. Everything else follows
from making that read as small and as fast as possible.

## The budget

| Resource | What the chip has | What it rules out |
| --- | --- | --- |
| Flash | 4 MB, including the 1 MB app | a model stored as 16-bit floats (8.8 MB) |
| RAM | 520 KB, about 260 KB usable, in two pieces | loading even one token's worth of weights (741 KB) |
| Flash bandwidth | 23.7 MB/s, memory-mapped, in order | reading megabytes per token |

A language model writes one token at a time and touches nearly every weight for
each one, so the number that matters is **bytes read per token**. Divide it by the
flash bandwidth and you get the fastest a token could ever be if the math were
free. I call that the flash floor.

## A model designed for the chip

No existing model fits these limits, so I chose the shape first and trained to it:
vocabulary 8,192, width 96, 6 layers, 4 attention heads, a SwiGLU FFN of 128, and
a context of 128 tokens. Trained on TinyStories on a free Colab GPU in about 30
minutes (final loss 2.436). TinyStories is a deliberately narrow world: small
models only become fluent when the job is narrow.

| Part | Parameters | Read per token |
| --- | --- | --- |
| Embedding table, reused as the output head (tied) | 786,432 | one row to embed, all 8,192 rows to score |
| Per-Layer Embedding (PLE) table | 3,145,728 | one 216-byte row |
| 6 layers: attention, FFN, PLE projections | 480,384 | all of them |
| **Total** | **4.41M** | **741,264 bytes** |

Three choices make it fit:

- **4-bit weights.** Each weight is an int4 with one fp16 scale per group of 32,
  about 4.5 bits per weight. The whole model is 2,510,528 bytes, and a 96-wide row
  is 56 bytes. Groups of 32 keep one large weight from crushing the precision of
  its whole row.
- **Per-Layer Embeddings**, the idea from Google's Gemma 3n (and slvDev's
  esp32-ai, which inspired this). Each token has its own extra 64 numbers per
  layer, kept in a 1.77 MB table. Only the current token's row is read: 71% of the
  parameters cost 216 bytes per token, about 0.03% of the traffic. The model knows
  more without being slower, which is exactly the trade a flash-bound chip wants.
- **A tied head.** Scoring the next token reuses the embedding table, which saves
  459 KB of flash. The price: the head reads all 459 KB for every token, 62% of the
  per-token traffic.

## Streaming from flash

The model sits in its own 2.94 MB flash partition, written once with `esptool`.
At boot, `esp_partition_mmap()` maps it into the address space, so the C code
reads weights through a plain pointer and the flash cache fetches them behind the
scenes. The file is laid out in exactly the order the forward pass reads it, so
almost every read is sequential, the fast pattern.

At 23.7 MB/s, 741,264 bytes set a floor of **31.3 ms per token**. The chip
measures 63.0 ms, twice the floor, so the ESP32 is compute-bound: the scalar int4
math costs about 9 cycles per weight, and there is no SIMD to share it out.

## RAM: 222 KB, and the KV cache is most of it

| Part | Bytes |
| --- | --- |
| KV cache, int8 (2 x 6 layers x 128 positions x 96) | 147,456 |
| Logits (8,192 floats) | 32,768 |
| KV scales, one per layer, position and head | 24,576 |
| RoPE sine and cosine table | 12,288 |
| Working vectors | about 4,100 |

The first runtime stored the KV cache as 16-bit floats: 64 positions filled
147 KB, and stories stopped mid-sentence at token 64. Storing keys and values as
int8 with one scale per head doubled the context to 128 tokens in the same
147 KB. Everything is carved once at boot from an arena that never frees, in two
pieces, because the largest single free block on this chip was 172 KB.

## Correctness before speed

A tiny model writes plausible text even with a bug in the math, so reading the
stories proves nothing. `export_ple.py` runs the exact int4 model in numpy,
including the int8 KV cache rounding, and writes a prompt plus its greedy
continuation to `golden.txt`. Every build on every board decodes the same prompt
and must reproduce it token for token. When the top two scores differ by less
than 0.05, a different float addition order can flip the winner; that counts as a
tie, not an error.

## From 9.3 to 15.9 tokens per second

The first working runtime ran at 9.33 tokens per second. Four changes, each aimed
at the biggest measured cost, took it to 15.9:

1. The int8 KV cache (above), which also made attention cheaper to read.
2. A precomputed RoPE table instead of sine and cosine per token.
3. A kernel that loads eight int4 weights per 32-bit word and unpacks them with
   shifts.
4. The output head split across both cores, with core 1 running a worker task.

Where the time goes now (two cores, the golden run):

| Stage | Flash floor | Measured | Measured / floor |
| --- | --- | --- | --- |
| Embed and PLE | 0.88 ms | 2.37 ms | 2.7x |
| Attention | 5.54 ms | 16.72 ms | 3.0x |
| FFN | 5.48 ms | 14.23 ms | 2.6x |
| Head | 19.37 ms | 29.25 ms | 1.5x |
| **Token** | **31.3 ms** | **63.0 ms** | **2.0x** |

The head is closest to its floor because it is the only stage on two cores.
Computed from SRAM instead of flash, the head takes 28.9 ms on one core and
19.7 ms on two: the math splits 1.47x. Read from flash, it goes from 45.9 to
29.3 ms, 1.57x, because each ESP32 core has its own flash cache, so two cores
actually read flash 14% faster together than one alone.

**Two corrections** to numbers I posted earlier: a token reads 741 KB, not
0.9 MB, so the floor is 31 ms, not 45 ms. And splitting the head across two cores
gave 1.57x, not 2x; my first comparison had a different kernel on each side.

## What the Arm port showed so far

The same runtime now runs on a Raspberry Pi Pico 2 W (two Cortex-M33 cores at
150 MHz). Per clock cycle its cores match the ESP32's. What differed was the
memory system: its two cores share one path to flash, and reading together they
got only 0.37x the bandwidth of one core alone. Streaming the head from flash into
SRAM with DMA, while both cores compute from SRAM, took the Pico from 7.9 to
12.9 tokens per second. At 22 ms, its head is now faster than the ESP32's. That
gets its own post.

## Next

Helium, Arm's vector extension, multiplies 16 int8 pairs per instruction. Next I
build an int8-activation reference on the host, then a Helium kernel on Arm's
Corstone-300 simulator (a Cortex-M55), checked against the same golden file. The
open question: with vector math, does the flash become the limit for the first
time?

Code, build steps and the benchmark: [README](../README.md).
