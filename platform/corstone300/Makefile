# Bare-metal Arm M-profile build: Corstone-300 FVP (Cortex-M55 + Helium) and QEMU.
# Needs the Arm GNU Toolchain (arm-none-eabi-gcc 13+, with newlib).
#
#   make KERNEL=mve                  Cortex-M55 + Helium kernel   -> build/sse300-mve.elf
#   make KERNEL=i8ref                Cortex-M55, plain C int8     -> build/sse300-i8ref.elf
#   make KERNEL=scalar               Cortex-M55, float kernel (checked against golden.txt)
#   make fvp KERNEL=mve              run on the Corstone-300 FVP (FVP_Corstone_SSE-300_Ethos-U55)
#   make qemu KERNEL=mve             run in QEMU (qemu-system-arm, machine mps3-an547)
#   make count KERNEL=mve            QEMU + tools/qemu_count: exact instruction counts
#   make qemu KERNEL=dsp BOARD=an505 the Pico's Cortex-M33 DSP kernel, in QEMU mps2-an505
#
# Model files come from MODEL_DIR (model.bin, tokenizer.bin, golden.txt, golden_i8.txt,
# kvec.bin); `make -C ../../host ktest` creates kvec.bin if it is missing.
KERNEL    ?= mve
BOARD     ?= sse300
MODEL_DIR ?= ../../model
OPT       ?= -O2
FVP       ?= FVP_Corstone_SSE-300_Ethos-U55
CC         = arm-none-eabi-gcc

ROOT      := ../..
MODEL_ABS := $(abspath $(MODEL_DIR))
I8_KERNELS = i8ref dsp mve
GOLDEN     = $(if $(filter $(I8_KERNELS),$(KERNEL)),golden_i8.txt,golden.txt)

ifeq ($(BOARD),sse300)
  CPU      = -mcpu=cortex-m55 -mfloat-abi=hard
  LDSCRIPT = sse300.ld
  QEMU_M   = mps3-an547
  PMU     ?= 1
else ifeq ($(BOARD),an505)
  CPU      = -mcpu=cortex-m33 -mfloat-abi=hard -mfpu=fpv5-sp-d16 -mfp16-format=ieee
  LDSCRIPT = an505.ld
  QEMU_M   = mps2-an505
  PMU     ?= 0
else
  $(error BOARD must be sse300 or an505)
endif

ELF     = build/$(BOARD)-$(KERNEL).elf
CFLAGS  = $(CPU) $(OPT) -std=c11 -ffp-contract=off -ffunction-sections -fdata-sections -g -Wall -Wextra \
          -I$(ROOT)/core -I$(ROOT)/kernels -DPLE_BOARD='"$(BOARD)"' -DPLE_PMU=$(PMU) \
          -DPLE_KERNEL_I8=$(if $(filter $(I8_KERNELS),$(KERNEL)),1,0)
ASFLAGS = $(CPU) -DPLE_MODEL_BIN='"$(MODEL_ABS)/model.bin"' -DPLE_KVEC_BIN='"$(MODEL_ABS)/kvec.bin"' \
          -DPLE_TOK_BIN='"$(MODEL_ABS)/tokenizer.bin"' -DPLE_GOLDEN_TXT='"$(MODEL_ABS)/$(GOLDEN)"'
LDFLAGS = $(CPU) --specs=rdimon.specs -nostartfiles -T $(LDSCRIPT) -Wl,--gc-sections -Wl,-Map=$(ELF:.elf=.map)
SRC     = startup.c main.c $(ROOT)/core/ple.c $(ROOT)/core/tokenizer.c $(ROOT)/core/bench.c \
          $(ROOT)/core/kern_check.c $(ROOT)/kernels/$(KERNEL).c
BLOBS   = $(MODEL_ABS)/model.bin $(MODEL_ABS)/kvec.bin $(MODEL_ABS)/tokenizer.bin $(MODEL_ABS)/$(GOLDEN)

$(ELF): $(SRC) blobs.S $(BLOBS) $(LDSCRIPT) sections.ld $(ROOT)/core/*.h $(ROOT)/kernels/*.h | build
	$(CC) $(CFLAGS) $(ASFLAGS) $(LDFLAGS) -o $@ $(SRC) blobs.S -lm
	arm-none-eabi-size $@

build:
	mkdir -p build

$(MODEL_ABS)/kvec.bin: $(MODEL_ABS)/model.bin
	python3 $(ROOT)/tools/kernel_vectors.py $< $@

# Semihosting prints to stdout; the run ends when main returns.
qemu: $(ELF)
	qemu-system-arm -M $(QEMU_M) -display none -serial null -monitor none \
	    -semihosting-config enable=on,target=native -kernel $(ELF)

count: $(ELF)
	python3 $(ROOT)/tools/qemu_count/qemu_count.py --machine $(QEMU_M) $(ELF)

# Corstone-300 FVP, no visualisation or telnet; --stat prints the instructions simulated.
fvp: $(ELF)
	$(FVP) -a $(ELF) --stat --timelimit 600 \
	    -C mps3_board.visualisation.disable-visualisation=1 \
	    -C mps3_board.telnetterminal0.start_telnet=0 \
	    -C mps3_board.uart0.out_file=- \
	    -C cpu0.semihosting-enable=1

clean:
	rm -rf build

.PHONY: qemu count fvp clean
