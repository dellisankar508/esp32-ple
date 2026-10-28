# Host build of the portable runtime.
#   make test                    synthetic model, golden check (needs python3 + numpy)
#   make tsan                    the same under ThreadSanitizer (streamed-head races)
#   make run MODEL_DIR=../model  your trained model.bin / tokenizer.bin / golden.txt
#   make KERNEL=scalar           kernel backend (kernels/<name>.c)
#   make ktest KERNEL=i8ref      kernel bit-exactness check against numpy (int8 kernels)
#
# Int8-activation kernels (I8_KERNELS) are checked against golden_i8.txt, the float
# kernel against golden.txt. Kernels are always built with -ffp-contract=off: the int8
# contract is bit-exact, and a fused multiply-add would round differently.
KERNEL    ?= scalar
MODEL_DIR ?= ../model
OPT       ?= -O2
I8_KERNELS = i8ref dsp mve
GOLDEN     = $(if $(filter $(I8_KERNELS),$(KERNEL)),golden_i8.txt,golden.txt)
BIN        = ple_host$(if $(filter scalar,$(KERNEL)),,_$(KERNEL))
CFLAGS    ?= $(OPT) -Wall -Wextra -std=c11 -ffp-contract=off -D_POSIX_C_SOURCE=200809L -DPLE_CFLAGS='"$(OPT)"'
INC        = -I../core -I../kernels
SRC        = main_host.c ../core/ple.c ../core/tokenizer.c ../core/bench.c ../core/head_stream.c \
             ../kernels/$(KERNEL).c
HDR        = ../core/*.h ../kernels/*.h

$(BIN): $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) -lm -pthread

# Same build under ThreadSanitizer: checks the streamed-head hand-off for data races.
$(BIN)_tsan: $(SRC) $(HDR)
	$(CC) -O1 -g -fsanitize=thread -Wall -std=c11 -ffp-contract=off -D_POSIX_C_SOURCE=200809L \
	    -DPLE_CFLAGS='"-O1 tsan"' $(INC) -o $@ $(SRC) -lm -pthread

kern_test_$(KERNEL): kern_test.c ../core/kern_check.c ../kernels/$(KERNEL).c $(HDR)
	$(CC) $(CFLAGS) $(INC) -o $@ kern_test.c ../core/kern_check.c ../kernels/$(KERNEL).c -lm

synthetic/model.bin: ../tools/make_synthetic.py ../train/export_ple.py ../tools/kernel_vectors.py
	python3 ../tools/make_synthetic.py synthetic

$(MODEL_DIR)/kvec.bin: $(MODEL_DIR)/model.bin ../tools/kernel_vectors.py ../train/export_ple.py
	python3 ../tools/kernel_vectors.py $< $@

test: $(BIN) synthetic/model.bin
	./$(BIN) synthetic/model.bin synthetic/tokenizer.bin synthetic/$(GOLDEN)

tsan: $(BIN)_tsan synthetic/model.bin
	./$(BIN)_tsan synthetic/model.bin synthetic/tokenizer.bin synthetic/$(GOLDEN)

run: $(BIN)
	./$(BIN) $(MODEL_DIR)/model.bin $(MODEL_DIR)/tokenizer.bin $(MODEL_DIR)/$(GOLDEN)

story: $(BIN)
	./$(BIN) $(MODEL_DIR)/model.bin $(MODEL_DIR)/tokenizer.bin --story "Once upon a time"

# Kernel check: the synthetic model, then your model if model.bin is there.
ktest: kern_test_$(KERNEL) synthetic/model.bin
	./kern_test_$(KERNEL) synthetic/model.bin synthetic/kvec.bin
	@if [ -f $(MODEL_DIR)/model.bin ]; then $(MAKE) --no-print-directory $(MODEL_DIR)/kvec.bin && \
	    ./kern_test_$(KERNEL) $(MODEL_DIR)/model.bin $(MODEL_DIR)/kvec.bin; fi

clean:
	rm -rf ple_host ple_host_* kern_test_* synthetic

.PHONY: test tsan run story ktest clean
