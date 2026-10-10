# esp32-ple

A 4.4M-parameter language model running entirely on microcontrollers: a classic
ESP32 with no PSRAM, and a Raspberry Pi Pico 2 W. No PC, no cloud. The same C
runtime runs on every target, and every target reproduces a Python reference
token for token.

| Target | Core | Result |
| --- | --- | --- |
| ESP32-D0WD | Xtensa LX6, 2x 240 MHz, no PSRAM, no SIMD | 63.0 ms/token, 15.9 tok/s, golden MATCH |
| Raspberry Pi Pico 2 W | Arm Cortex-M33, 2x 150 MHz | 77.7 ms/token, 12.9 tok/s with the DMA-streamed head (126.6 ms, 7.9 tok/s without), golden MATCH |
| Corstone-300 FVP | Arm Cortex-M55 + Helium (simulated) | next: Helium kernel, instruction counts |

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

The full story, with the measurements behind each decision:
[docs/esp32-writeup.md](docs/esp32-writeup.md).

Inspired by [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai).

## Layout

```
core/       ple.c (model), tokenizer.c, bench.c (one measurement method for all targets),
            head_stream.c (output head streamed through SRAM buffers by a platform mover)
kernels/    kern.h (the matvec interface), scalar.c (int4 x float, the ESP32 kernel)
platform/   esp32/ (ESP-IDF + OLED demo), pico2/ (Pico SDK, dma_head.c = the DMA mover)
host/       ple_host: same core on your PC; `make test` = golden check on a synthetic model
train/      train_ple.py (PyTorch, TinyStories), export_ple.py (PLE1 int4 export + numpy reference)
tools/      make_synthetic.py, ttf2font.py
model/      your model.bin, tokenizer.bin, golden.txt (not committed)
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
as `core/ple.c` does, and writes `golden.txt`.

## 2. Host check

```
cd host
make test        # synthetic model: GOLDEN: MATCH on 1 and 2 threads, plain and streamed head
make tsan        # the same under ThreadSanitizer (checks the streamed-head hand-off)
make run         # your model/ files
make story
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
```

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
| `GOLDEN` | greedy decode matches `golden.txt`, over every run |
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
