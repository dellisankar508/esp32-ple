# Host build of the portable runtime.
#   make test                    synthetic model, golden check (needs python3 + numpy)
#   make tsan                    the same under ThreadSanitizer (streamed-head races)
#   make run MODEL_DIR=../model  your trained model.bin / tokenizer.bin / golden.txt
#   make KERNEL=scalar           kernel backend (kernels/<name>.c)
KERNEL    ?= scalar
MODEL_DIR ?= ../model
OPT       ?= -O2
CFLAGS    ?= $(OPT) -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L -DPLE_CFLAGS='"$(OPT)"' 
INC        = -I../core -I../kernels
SRC        = main_host.c ../core/ple.c ../core/tokenizer.c ../core/bench.c ../core/head_stream.c \
             ../kernels/$(KERNEL).c

ple_host: $(SRC) ../core/*.h ../kernels/kern.h
	$(CC) $(CFLAGS) $(INC) -o $@ $(SRC) -lm -pthread

# Same build under ThreadSanitizer: checks the streamed-head hand-off for data races.
ple_host_tsan: $(SRC) ../core/*.h ../kernels/kern.h
	$(CC) -O1 -g -fsanitize=thread -Wall -std=c11 -D_POSIX_C_SOURCE=200809L -DPLE_CFLAGS='"-O1 tsan"' \
	    $(INC) -o $@ $(SRC) -lm -pthread

synthetic/model.bin: ../tools/make_synthetic.py ../train/export_ple.py
	python3 ../tools/make_synthetic.py synthetic

test: ple_host synthetic/model.bin
	./ple_host synthetic/model.bin synthetic/tokenizer.bin synthetic/golden.txt

tsan: ple_host_tsan synthetic/model.bin
	./ple_host_tsan synthetic/model.bin synthetic/tokenizer.bin synthetic/golden.txt

run: ple_host
	./ple_host $(MODEL_DIR)/model.bin $(MODEL_DIR)/tokenizer.bin $(MODEL_DIR)/golden.txt

story: ple_host
	./ple_host $(MODEL_DIR)/model.bin $(MODEL_DIR)/tokenizer.bin --story "Once upon a time"

clean:
	rm -rf ple_host ple_host_tsan synthetic

.PHONY: test tsan run story clean
