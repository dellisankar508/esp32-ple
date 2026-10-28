# esp32-ple

A 4.4M-parameter language model running entirely on microcontrollers: a classic
ESP32 with no PSRAM, and a Raspberry Pi Pico 2 W. No PC, no cloud. The same C
runtime runs on every target, and every target reproduces a Python reference
token for token.

| Target | Core | Result |
| --- | --- | --- |
| ESP32-D0WD | Xtensa LX6, 2x 240 MHz, no PSRAM, no SIMD | 63.0 ms/token, 15.9 tok/s, golden MATCH |
| Raspberry Pi Pico 2 W | Arm Cortex-M33, 2x 150 MHz | **40.1 ms/token, 25.0 tok/s**: int8 DSP kernel, DMA-streamed head, flash at 75 MHz (v0.1.0 float kernel: 77.7 ms, 12.9 tok/s); kernel bit-exact on the chip, golden MATCH |
| Corstone-300 (Cortex-M55 + Helium) | simulated: QEMU `mps3-an547` now, Arm's FVP next | int8 Helium kernel bit-exact; 1.26M instructions per token, 7.7x fewer than the float kernel (QEMU counts, not timings) |

**How it fits:**

- **Weights never enter RAM.** The 2.5 MB model stays in flash and is streamed
  for every token (741 KB per token). RAM holds only the KV cache and activations:
  222 KB.
- **4-bit weights** (int4, one fp16 scale per 32 weights) make the model fit the
  4 MB flash and cut the bytes read per token.
- **Per-Layer Embeddings (PLE)**, the idea from Google's Gemma 3n: 71% of the
  parameters sit in a table of which each token reads one 216-byte row.
- **int8 KV cache** with a scale per head: 128 tokens of context in 147 KB.
- **On the Pico 2 W, DMA streams the output head** from flash into SRAM while both
  cores compute, which took it from 7.9 to 12.9 tok/s.
- **int8 activations, bit-exact everywhere** (new): the inputs of every matvec are
  quantized to int8 per 32 values, so Arm's integer SIMD can do the math. The int8
  path is defined down to the last rounding step, so the plain C, the Cortex-M33 DSP
  and the Helium kernels all produce the numpy reference's exact bits. It costs
  almost nothing (98.9% top-1 agreement with float, perplexity 5.013 -> 5.019), and
  took the Pico 2 W from 12.9 to 19.3 tok/s; flash at 75 MHz then took it to 25.0.

The full story, with the measurements behind each decision:
[docs/esp32-writeup.md](docs/esp32-writeup.md). The int8, DSP and Helium kernels:
[docs/int8-helium.md](docs/int8-helium.md).

Inspired by [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai).

## Layout

```
core/       ple.c (model), tokenizer.c, bench.c (one measurement method for all targets),
            head_stream.c (output head streamed through SRAM buffers by a platform mover)
            kern_check.c (a kernel against numpy test vectors, bit for bit)
kernels/    kern.h (the matvec interface), scalar.c (int4 x float, the ESP32 kernel),
            q8.h + i8ref.c (int8 activations, plain C), dsp.c (Cortex-M33 SMLAD), mve.c (Helium)
platform/   esp32/ (ESP-IDF + OLED demo), pico2/ (Pico SDK, dma_head.c = the DMA mover),
            corstone300/ (bare metal: Corstone-300 FVP or QEMU, Cortex-M55 and M33)
host/       ple_host: same core on your PC; `make test` = golden check on a synthetic model
train/      train_ple.py (PyTorch, TinyStories), export_ple.py (PLE1 int4 export + numpy
            reference, float or int8 activations)
tools/      make_synthetic.py, kernel_vectors.py, act_quality.py, qemu_count/ (exact
            instruction counts in QEMU), compare_logs.py, ttf2font.py
model/      your model.bin, tokenizer.bin, golden.txt, golden_i8.txt, kvec.bin (not committed)
```

## 1. Model files

The quickest way: download `model.bin`, `tokenizer.bin` and `golden.txt` from the
[latest release](../../releases/latest) into `model/`.

Or train your own with `train/train_ple.py` (Colab T4: about 30 minutes), then:

```
python train/export_ple.py --ckpt ple_d0wd96.pt --tokenizer tokenizer.json --check
cp model.bin tokenizer.bin golden.txt model/
```

`--check` runs the numpy int4 reference, which models the int8 KV cache exactly
as `core/ple.c` does, and writes `golden.txt` (float activations) and `golden_i8.txt`
(int8 activations). For the int8 kernels you also need the kernel test vectors.
Both work from `model.bin` alone, numpy only:

```
python train/export_ple.py --regolden model --act i8      # golden_i8.txt
python tools/kernel_vectors.py model/model.bin model/kvec.bin
```

## 2. Host check

```
cd host
make test        # synthetic model: GOLDEN: MATCH on 1 and 2 threads, plain and streamed head
make tsan        # the same under ThreadSanitizer (checks the streamed-head hand-off)
make run         # your model/ files
make story
make test KERNEL=i8ref   # int8 activations, checked against golden_i8.txt
make ktest KERNEL=i8ref  # the kernel alone vs numpy, every output bit for bit
```

On the host, "2 cores" are two threads, and the streamed head runs with a fake
DMA: a third thread copies each block after a random short delay, into three
1 KB buffers, so the hand-off between mover and cores runs hundreds of times
per token. The timings are meaningless; the golden check is the point.

## 3. ESP32-D0WD

```
cd platform/esp32
./flash_model.sh                     # model/model.bin -> model partition at 0x110000
idf.py set-target esp32
idf.py build flash monitor
```

Boot runs the shared benchmark (see below, about 40 s), then starts the OLED
demo (BOOT button = story). Press the board's EN/RST button to run it again.
OLED: SDA GPIO21, SCL GPIO22, address 0x3C.

## 4. Raspberry Pi Pico 2 W

Needs the Arm GNU Toolchain 13.2 or newer (`arm-none-eabi-gcc`), CMake and the
Pico SDK 2.x:

```
git clone --branch 2.2.0 https://github.com/raspberrypi/pico-sdk.git
cd pico-sdk && git submodule update --init lib/tinyusb && cd ..
export PICO_SDK_PATH=$PWD/pico-sdk

cmake -S platform/pico2 -B build-pico          # -DPICO_BOARD=pico2 for the non-W board
cmake --build build-pico -j

# int8 activations with the Cortex-M33 DSP kernel (needs golden_i8.txt and kvec.bin):
cmake -S platform/pico2 -B build-pico-dsp -DPLE_KERNEL=dsp
cmake --build build-pico-dsp -j
```

With an int8 kernel (`dsp` or `i8ref`), boot first checks the kernel against the
numpy test vectors on the chip (`KERNEL | dsp | BITEXACT 42/42 cases`).

The flash runs at the Pico SDK's setting for this board, sys/2 = 75 MHz, instead of
the boot ROM's sys/3 = 50 MHz (`PLE_FAST_FLASH`, on by default; measured 20.5 -> 27.8
MB/s). The `ENV` and `FLASH` lines show what you got, and the `MODEL` checksum shows
the reads are correct. `-DPLE_FAST_FLASH=OFF` keeps the boot ROM's 50 MHz.

The first configure builds picotool from source (about a minute). Then hold
BOOTSEL, plug in USB, and copy `build-pico/ple_pico2.uf2` to the RP2350 drive.
The UF2 carries the firmware and the model, so the copy takes a little while.
Open the serial port:

```
python -m serial.tools.miniterm /dev/ttyACM0 115200
```

The board waits for the serial connection, then prints the same benchmark as
the ESP32. Afterwards: Enter = story, `b` = run the benchmark again.

How the Pico build is laid out:
- `model.bin` is embedded in a `.flashdata` section, so it stays in flash and
  streams through the XIP cache, like the ESP32's mmapped partition.
- All code runs from SRAM (`copy_to_ram`), so the flash only carries weights.
- The output head is split across both Cortex-M33 cores via the inter-core FIFO.
- DMA-streamed head (`dma_head.c` + `core/head_stream.c`): DMA copies the head rows
  from flash into six 8 KB SRAM buffers while both cores compute on the blocks
  that have already arrived, so the cores never wait on flash during the head.
  Two DMA sources are timed at boot, and the faster one that returns the right
  bytes is kept: the XIP streaming FIFO (`DREQ_XIP_STREAM`, read via `XIP_AUX`)
  or plain DMA from the uncached flash alias. Press `d` to switch it off and on
  for stories.

Why: the two-board comparison showed the RP2350's cores are as fast per clock as
the ESP32's, but both cores reading flash at once share one path (FLASH2 0.37x),
so the plain 2-core head spends 50 of its 71 ms waiting on flash. Measured with
the DMA head: the stream engine moves the head at 22.8 MB/s (the CPU reads 20.5),
the head drops from 70.9 to 22.0 ms (pure compute is 20.9), and a token from
126.6 to 77.7 ms.

## 5. Cortex-M55 + Helium: Corstone-300 FVP (or QEMU)

`platform/corstone300` is a bare-metal build for Arm's Corstone-300 (Cortex-M55 with
Helium). It runs on Arm's free Fixed Virtual Platform, and on QEMU's `mps3-an547`.
Output goes through semihosting. Each run prints:

- `KERNEL`: the kernel against the numpy test vectors, every output bit for bit
- `GOLDEN`: the greedy decode against `golden_i8.txt`
- `LOGITS`: a hash of every logit of that decode. Two kernels with the same hash on
  the same core produced a bit-identical run of the whole model.
- `PMU` (FVP): instructions, Helium instructions and cycles per token, from the
  Cortex-M55's performance monitor

```
cd platform/corstone300
make KERNEL=mve                 # also: i8ref (plain C int8), scalar (float)
make fvp KERNEL=mve             # FVP_Corstone_SSE-300_Ethos-U55 on your PATH
make qemu KERNEL=mve            # or QEMU (apt install qemu-system-arm)
make count KERNEL=mve           # exact instruction counts in QEMU (tools/qemu_count)
make qemu KERNEL=dsp BOARD=an505   # the Pico's M33 kernel, on QEMU's Cortex-M33
```

Needs `arm-none-eabi-gcc` 13 or newer with newlib. Get the FVP from Arm's
[Ecosystem FVPs](https://developer.arm.com/downloads/-/arm-ecosystem-fvps) page
(Corstone-300, Linux; it runs in WSL). The FVP and QEMU are not cycle-accurate, so
their numbers compare kernels by instructions; they are not timings.

Instructions per token in QEMU (51-token golden decode, 4.4M-parameter model):

| Core | Kernel | Instructions / token | In the kernel | Golden |
| --- | --- | --- | --- | --- |
| Cortex-M55 | `scalar` (float activations) | 9.76M | 9.34M | MATCH |
| Cortex-M55 | `i8ref` (int8, plain C) | 8.48M | 8.05M | MATCH, bit-exact |
| Cortex-M55 | `mve` (int8, Helium) | **1.26M** | **0.83M** | MATCH, bit-exact, same logits as `i8ref` |
| Cortex-M33 | `scalar` | 9.92M | 9.37M | MATCH |
| Cortex-M33 | `i8ref` | 9.05M | 8.51M | MATCH, bit-exact |
| Cortex-M33 | `dsp` (int8, SMLAD) | **3.77M** | **3.23M** | MATCH, bit-exact, same logits as `i8ref` |

## The benchmark

Every target runs the same `bench_run_all()` from `core/bench.c` at boot, so the
numbers compare like with like. It prints one tagged line per measurement:

| Line | What it measures |
| --- | --- |
| `ENV` | board, CPU clock (measured on the ESP32, from the clock driver on the Pico), flash mode and clock, compiler and flags |
| `MODEL` | checksum of `model.bin` (must be equal on every board), state RAM, bytes read per token |
| `FLASH` | sequential read MB/s, random 216-byte row time |
| `FLASH2` | the same bytes read by one core, then by both cores at once |
| `FLOOR` | per-stage flash floor: the bytes a stage reads / sequential bandwidth |
| `KERNEL` | (int8 kernels) the kernel against the numpy test vectors in `kvec.bin`, bit for bit |
| `GOLDEN` | greedy decode matches `golden.txt` (`golden_i8.txt` for int8 kernels), over every run |
| `RESULT` | ms/token and the per-stage split, 1 core and 2 cores, median of 3 runs |
| `HEADSRAM` | the output head with its weights copied to SRAM (pure compute), 1 vs 2 cores, next to the same head read from flash |
| `RESULT ... +dma` | (Pico) the same runs with the DMA-streamed head, 1 and 2 cores, each with its own `GOLDEN` line |
| `HEADSTREAM` | (Pico) moving the head with DMA and no math (ms, MB/s), the streamed head on 1 and 2 cores, compute-only times, the DMA mode chosen |
| `LONGCTX` | speed at positions 100-119, where attention costs the most, fastest config |

`floor` = the fastest a token could be if the math were free. Measured well
above floor means compute-bound; close to it means flash-bound.

The ESP32 takes about 40 s and the Pico 2 W about 100 s. Copy each board's output
(from `=== PLE bench ===` to `=== bench done ===`) into a text file, then:

```
python3 tools/compare_logs.py esp32.log pico2w.log > comparison.md
```

Host timings (`make test`) only check that the code runs; they are not a target.

## Licence

MIT. Poppins font (OLED demo) under the SIL Open Font License.
