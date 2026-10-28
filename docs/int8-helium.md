# int8 activations, the Cortex-M33 DSP kernel and Helium

On a Raspberry Pi Pico 2 W, the int8 DSP kernel takes the model from **12.9 to 19.3
tokens per second** (77.7 to 51.9 ms per token), and the chip itself checks the kernel
against numpy **bit for bit**. Running the flash at 75 MHz, the Pico SDK's own setting
for this board, takes it to **25.0 tokens per second (40.1 ms)**. At 150 MHz the Pico
is now 1.57x faster than the 240 MHz ESP32 (63.0 ms per token, float kernel).

On a simulated Cortex-M55, the Helium kernel runs a token in **1.26M instructions,
7.7x fewer than the float kernel**, and every logit of a 51-token decode has the same
bits as a plain C reference. Arm's own Corstone-300 FVP and QEMU agree on every one
of those bits. Quantizing the activations to int8, which is what makes this
possible, costs almost nothing in quality.

Times come from the Pico. Instruction counts come from QEMU (`mps3-an547` for the
M55, `mps2-an505` for the M33) and Arm's Corstone-300 FVP, which agree to within a
few instructions. They compare kernels; they are not timings.

## Why int8 activations

The ESP32 kernel multiplies int4 weights by float inputs. A vector unit can't do
much with that: Helium's fast instructions multiply sixteen int8 pairs at once,
and the M33's SMLAD multiplies two int16 pairs. Both need integer inputs. So each
matvec now quantizes its input vector first, per group of 32 values, with one
float scale per group (the same blocking as the weights, and the same scheme as
llama.cpp's Q8_0).

## A contract defined to the last bit

The float path could only promise token-exact output, because the C code and numpy
add floats in a different order. The int8 path is specified completely, so every
kernel can be held to the reference's exact bits. "I8 v1", in
`train/export_ple.py` (`matvec_i8`):

1. **Quantize** each group of 32 inputs: `m = max|x|`, `q = rint(x * 127/m)`
   (round half to even), scale `sx = m/127`. Groups with `m < 2^-60` become zero.
2. **Integer dot** per row and group: 32 int4 weights times 32 int8 inputs, exact
   in int32.
3. **Float tail**: `acc = acc + d * (s * sx)`, groups in order, every operation
   rounded to float32, no fused multiply-add.

Two details make it hold on real hardware:

- **No fused multiply-add.** Kernels are built with `-ffp-contract=off`. An FMA
  rounds once where the reference rounds twice.
- **No subnormal floats.** The `2^-60` cut-off keeps every value in step 3 a normal
  float, so a vector unit that flushes subnormals to zero still agrees.

## Proving it

- **Kernel test vectors** (`tools/kernel_vectors.py`): 42 cases on 7 real weight
  matrices, from the 64-wide PLE projection to the 8192-row output head. The inputs
  cover ordinary values, outliers, an all-zero group, exact .5 rounding ties, tiny
  values and vanishing ones. `core/kern_check.c` compares every output float bit for
  bit, and runs each case a second time split into two row ranges, as two cores
  would.
- **The test catches real bugs.** Three deliberately wrong kernels all fail it:
  rounding half away from zero (7 of 42 cases fail), multiplying the scales in a
  different order (28), and a fused multiply-add in the float tail (27).
- **Whole-model hash.** The Arm runs hash every logit of the golden decode. On each
  core, the plain C kernel and the SIMD kernel print the same hash: `0xe5f66a1b` on
  the M55 (`i8ref`, `mve`), `0x5d2815eb` on the M33 (`i8ref`, `dsp`). The two cores'
  hashes differ from each other; the likely cause is the non-kernel code (norms,
  softmax), which each core gets compiled with its own maths library; not traced yet.
- **On the chip.** The Pico runs the same 42 cases at boot, before the benchmark:
  `KERNEL | dsp | BITEXACT 42/42 cases`.
- **Golden decode.** Every build matches `golden_i8.txt`, with 1 to 3 near-ties
  (logit gap under 0.05). Those come from the code outside the kernels (norms,
  softmax, RoPE), which is still plain float and differs from numpy in the last
  bits. The kernels themselves are bit-exact.

## What int8 costs in quality

`tools/act_quality.py` samples 8 stories from the float model and has both
references score all 984 positions:

| Measure | Result |
| --- | --- |
| Same top-1 next token | 98.88% |
| Mean KL(float \|\| int8) | 0.00034 nats (max 0.0031) |
| Perplexity of the text | 5.013 float, 5.019 int8 |

On the 48-token golden story, int8 matches float for 45 tokens, then ends "Her mom
said" instead of "Her mom was very happy".

## The kernels

All four read the weights exactly as `model.bin` stores them, so nothing changes in
the model file or the flash layout.

**`i8ref.c`**: plain C. It defines the answer.

**`dsp.c`** (Cortex-M33, M4, M7). A 32-bit word holds 8 weights, each byte an even
and an odd column.

- `L = (w << 4) & 0xF0F0F0F0` and `H = w & 0xF0F0F0F0` give every weight times 16
  in its own byte. The mask stays in a register, so each is one instruction.
- `SXTB16` sign-extends bytes 0 and 2 into two 16-bit lanes. Its built-in rotate
  reaches bytes 1 and 3; GCC doesn't fold that rotate by itself, so the kernel
  writes it in assembly.
- `kern_prepare` packs the int8 inputs into matching 16-bit pairs.
- Four SMLADs then do 8 multiply-adds. The sum is 16 times the dot product, so the
  16 is divided out of the input scale instead, which is exact.
- Result: 13 instructions per 8 weights in the inner loop; about 79 per group of 32
  once the float tail is included.

**`mve.c`** (Cortex-M55, M85).

- **The dot product.** One 16-byte load holds a whole group of 32 weights.
  `b << 4` and `b & 0xF0` give the even and odd weights times 16, still as int8, and
  two `VMLADAV`s multiply-accumulate them against the de-interleaved inputs.
- **The float tail, after the first version.** That version spent 28 instructions
  per group, of which only 7 were Helium: the scalar float tail had become the
  bottleneck. The current kernel takes four rows at a time, so the tail runs as
  vector float maths on four lanes, in the same order of operations per lane.
- **Result.** The output head went from 682K to 463K instructions.
- **One instruction avoided.** The natural way to widen the four fp16 scales is
  Helium's `VCVTB.F32.F16`. The GNU assembler here encodes it wrongly: its own
  disassembler can't decode the result, and QEMU faults on it. binutils has a commit
  fixing the encoding of exactly this instruction. The kernel converts the scales
  with the scalar instruction instead, which every toolchain gets right.

## Results: instruction counts

| Core | Kernel | Instructions / token | In the kernel | Output head (8192 x 96) |
| --- | --- | --- | --- | --- |
| Cortex-M55 | float | 9.76M | 9.34M | 5.80M |
| Cortex-M55 | int8 plain C | 8.48M | 8.05M | 4.95M |
| Cortex-M55 | int8 Helium | **1.26M** | **0.83M** | **0.46M** |
| Cortex-M33 | float | 9.92M | 9.37M | 5.82M |
| Cortex-M33 | int8 plain C | 9.05M | 8.51M | 5.24M |
| Cortex-M33 | int8 DSP | **3.77M** | **3.23M** | **1.95M** |

Helium executes 11x fewer kernel instructions than the float kernel, and 7.7x fewer
for the whole token.

The next bottleneck is outside the kernels. With Helium, a third of each token's
instructions are now attention, softmax and the norms: 0.43M, and attention grows
with the context.

### On Arm's Corstone-300 FVP

The same two ELFs (`mve` and `i8ref`), run on Arm's Fixed Virtual Platform for the
Corstone-300 (version 11.27), give the same answer as in QEMU:

| Check | `mve` (Helium) | `i8ref` (plain C) |
| --- | --- | --- |
| Kernel vs numpy | BITEXACT 42/42 | BITEXACT 42/42 |
| Golden decode | MATCH | MATCH |
| Hash of all 51 x 8192 logits | `0xe5f66a1b` | `0xe5f66a1b` |
| Output head, FVP cycle counter | 463,283 | 4,950,400 |
| Output head, QEMU instructions | 463,277 | 4,950,394 |
| Whole run, FVP instructions | 92.8M | 524.8M |

- **Two independent models of the Cortex-M55 agree on every bit.** QEMU and Arm's
  Fast Model implement the core separately, and both print the same hash over every
  logit of the run, for both kernels.
- **The FVP's cycle counter counts one per instruction.** The FVP is not
  cycle-accurate, so its cycle count equals the instruction count. It matches QEMU's
  count for the output head to within 6 instructions; the difference is the
  measurement code around the region.
- **The whole run** includes the kernel check and hashing the logits. Helium runs it
  in 5.7x fewer instructions than the plain C kernel.

Neither QEMU nor the FVP models timing. They execute instructions, not cycles, and
the Cortex-M55 runs most 128-bit Helium instructions over two cycles,
overlapping neighbours ("beats"). On real boards the weights also stream from
flash, so a faster kernel moves the bottleneck towards flash. The Pico shows how
much.

## On silicon: Pico 2 W

Same model, same benchmark (`core/bench.c`), Cortex-M33 at 150 MHz. Every run
matches its golden file; the float kernel is the v0.1.0 release.

| Configuration | Float kernel, flash 50 MHz | int8 DSP, flash 50 MHz | int8 DSP, flash 75 MHz |
| --- | --- | --- | --- |
| 1 core, head read from flash | 139.7 ms | 74.3 ms | 57.8 ms |
| 2 cores, head read from flash | 126.7 ms | 98.0 ms | 61.6 ms |
| 2 cores, head streamed by DMA (best) | 77.7 ms (12.9 tok/s) | 51.9 ms (19.3 tok/s) | **40.1 ms (25.0 tok/s)** |
| Layers: embed + attention + FFN | 55.1 ms | 30.9 ms | 24.6 ms |
| Head maths only (weights in SRAM), 2 cores | 20.9 ms | 8.8 ms | 8.8 ms |

The int8 kernel alone is 1.50x; with the faster flash, 1.94x over v0.1.0.

**Faster flash.** The boot ROM sets the flash clock to sys/3 (50 MHz). The Pico SDK's
board file for the Pico 2 W specifies sys/2 (75 MHz), applied by its "boot stage 2"
XIP setup, which a default RP2350 build doesn't run. `-DPLE_FAST_FLASH=ON` (now the
default) embeds it. The clock went up 1.5x and bandwidth 1.36 to 1.38x:

| Flash | 50 MHz | 75 MHz |
| --- | --- | --- |
| CPU sequential read | 20.5 MB/s | 27.8 MB/s |
| DMA moving the head | 22.8 MB/s | 31.5 MB/s |
| Two cores reading at once, vs one | 0.38x | 0.52x |
| Random 216-byte row | 16.9 µs | 13.8 µs |

The reads are correct: the model checksum over all 2.5 MB, read through XIP at
75 MHz, matches, the DMA path checks the bytes it moves, and every golden run
matches. (I had estimated the head at 13.5 ms if bandwidth scaled with the clock;
it came in at 14.8 ms.)

Where the 40.1 ms go, against the flash floor (bytes read / 27.8 MB/s):

| Stage | Measured | Flash floor |
| --- | --- | --- |
| Embed | 1.8 ms | 0.8 ms |
| Attention | 12.8 ms | 4.7 ms |
| FFN | 10.0 ms | 4.7 ms |
| Head, DMA-streamed | 14.8 ms | 16.5 ms (DMA alone: 14.6 ms) |
| **Token** | **40.1 ms** | **26.6 ms** |

What it shows:

- **The head is balanced on the transfer.** With two cores it takes 14.8 ms, and the
  DMA alone needs 14.6 ms to move its 459 KB; the maths (8.8 ms) is hidden under it.
  With one core the maths is now the longer of the two: 16.9 ms against 16.8 ms of
  maths alone.
- **The layers are 61% of the token, and there flash and maths add up.** They run on
  one core, straight from flash: the core stalls on each flash read, then computes.
  24.6 ms measured against a 10.1 ms floor leaves about 14.5 ms of maths that never
  overlaps the reading. The same pattern held at 50 MHz (30.9 ms against 13.8).
- **Two cores without DMA are still slower than one** (61.6 against 57.8 ms), though
  less so: with faster flash, two cores reading together keep 0.52x of one core's
  bandwidth instead of 0.38x.
- **Attention grows with context, and that part is pure maths.** At positions 100
  to 119 it takes 8.6 ms more than on average, at 50 MHz and at 75 MHz alike: about
  0.1 ms more per position. The q·K and softmax·V loops over the KV cache are still
  scalar float.

## Next

1. **Stream the layer weights with DMA too.** `model.bin` stores them in exactly the
   order the forward pass reads them, so one sequential stream covers all six layers
   (282 KB, 9.0 ms at the DMA's 31.5 MB/s). Flash and maths would then overlap, as
   they do in the head, and core 1, idle during the layers, can take half the maths.
   Estimate: 25 to 30 ms per token, 33 to 40 tokens/s. The floor for the whole token
   at the DMA's rate is 23.5 ms.
2. **int8 attention.** Quantize q per head and use SMLAD (M33) or VMLADAV (Helium)
   against the int8 K cache, which already has the right format.
3. **Cycles on real Helium silicon.** Instruction counts can't show how the M55
   overlaps Helium instructions. A Cortex-M55 or M85 board would, for example the
   Himax WiseEye2 on Seeed's Grove Vision AI V2 or the Renesas RA8M1.
