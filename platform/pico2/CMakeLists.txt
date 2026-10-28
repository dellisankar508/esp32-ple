cmake_minimum_required(VERSION 3.13)

# Board: Pico 2 W by default; -DPICO_BOARD=pico2 for the non-wireless board.
if(NOT DEFINED PICO_BOARD)
    set(PICO_BOARD pico2_w)
endif()
set(PICO_PLATFORM rp2350-arm-s)

include(pico_sdk_import.cmake)
project(ple_pico2 C CXX ASM)
set(CMAKE_C_STANDARD 11)
pico_sdk_init()

# Where model.bin, tokenizer.bin and golden.txt live (from train/export_ple.py).
set(PLE_MODEL_DIR "${CMAKE_CURRENT_LIST_DIR}/../../model" CACHE PATH "Directory with model.bin, tokenizer.bin, golden.txt")
set(PLE_KERNEL scalar CACHE STRING "Kernel backend in kernels/<name>.c: scalar (float) or i8ref / dsp (int8)")
set(PLE_OPT -O2 CACHE STRING "Optimisation flag for the PLE code (ESP32 uses -O2)")
option(PLE_COPY_TO_RAM "Run all code from SRAM so flash only streams weights" ON)
option(PLE_FAST_FLASH "Flash at clk_sys / PICO_FLASH_SPI_CLKDIV (2 on the Pico 2 / 2 W: 75 MHz) instead of the boot ROM's sys/3" ON)

# Int8-activation kernels are checked against golden_i8.txt and, at boot, against the
# numpy kernel test vectors in kvec.bin (bit for bit). The float kernel uses golden.txt.
set(PLE_I8_KERNELS i8ref dsp)
if(PLE_KERNEL IN_LIST PLE_I8_KERNELS)
    set(PLE_KERNEL_I8 1)
    set(PLE_GOLDEN golden_i8.txt)
    set(PLE_FILES model.bin tokenizer.bin golden_i8.txt kvec.bin)
else()
    set(PLE_KERNEL_I8 0)
    set(PLE_GOLDEN golden.txt)
    set(PLE_FILES model.bin tokenizer.bin golden.txt)
endif()
foreach(f ${PLE_FILES})
    if(NOT EXISTS "${PLE_MODEL_DIR}/${f}")
        message(FATAL_ERROR "Missing ${PLE_MODEL_DIR}/${f}. Copy it from your train folder, "
                            "or set -DPLE_MODEL_DIR=<dir>. golden_i8.txt: "
                            "python3 train/export_ple.py --regolden model --act i8; kvec.bin: "
                            "python3 tools/kernel_vectors.py model/model.bin model/kvec.bin")
    endif()
endforeach()
get_filename_component(PLE_MODEL_DIR_ABS "${PLE_MODEL_DIR}" ABSOLUTE)
configure_file(blobs.S.in ${CMAKE_CURRENT_BINARY_DIR}/blobs.S @ONLY)

set(ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)
add_executable(ple_pico2
    main.c
    dma_head.c
    ${CMAKE_CURRENT_BINARY_DIR}/blobs.S
    ${ROOT}/core/ple.c
    ${ROOT}/core/tokenizer.c
    ${ROOT}/core/bench.c
    ${ROOT}/core/head_stream.c
    ${ROOT}/core/kern_check.c
    ${ROOT}/kernels/${PLE_KERNEL}.c
)
list(TRANSFORM PLE_FILES PREPEND "${PLE_MODEL_DIR_ABS}/" OUTPUT_VARIABLE PLE_FILE_PATHS)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/blobs.S PROPERTIES OBJECT_DEPENDS "${PLE_FILE_PATHS}")
# The int8 contract is bit-exact: no fused multiply-add in the kernel, and the FPU's
# fp16 conversion for the weight scales (exact, like the software one).
if(PLE_KERNEL_I8)
    set_source_files_properties(${ROOT}/kernels/${PLE_KERNEL}.c PROPERTIES
        COMPILE_OPTIONS "-ffp-contract=off;-mfp16-format=ieee")
endif()

target_include_directories(ple_pico2 PRIVATE ${ROOT}/core ${ROOT}/kernels)
target_compile_definitions(ple_pico2 PRIVATE
    PLE_HOT_SECTION=".time_critical.ple"     # kernels in SRAM even without copy_to_ram
    PLE_CFLAGS="${PLE_OPT}"
    PLE_KERNEL_I8=${PLE_KERNEL_I8}
)
target_compile_options(ple_pico2 PRIVATE ${PLE_OPT} -Wall)
target_link_libraries(ple_pico2 pico_stdlib pico_multicore hardware_dma)

if(PLE_COPY_TO_RAM)
    pico_set_binary_type(ple_pico2 copy_to_ram)
endif()

# The boot ROM sets the flash up conservatively (here sys/3 = 50 MHz). Embedding the
# SDK's XIP setup ("boot stage 2") makes crt0 run it again before main, with the board
# header's PICO_FLASH_SPI_CLKDIV and PICO_FLASH_SPI_RXDELAY (2 and 2 for the Pico 2 W).
# The ENV and FLASH lines of the benchmark show the clock and bandwidth you got, and the
# MODEL checksum (all 2.5 MB read through XIP) shows the reads are correct.
# Measured on a Pico 2 W: 20.5 -> 27.8 MB/s, 51.9 -> 40.1 ms/token. -DPLE_FAST_FLASH=OFF
# keeps the boot ROM's setting.
if(PLE_FAST_FLASH)
    target_compile_definitions(ple_pico2 PRIVATE PICO_EMBED_XIP_SETUP=1)
endif()

pico_enable_stdio_usb(ple_pico2 1)
pico_enable_stdio_uart(ple_pico2 0)
pico_add_extra_outputs(ple_pico2)       # .uf2 for drag-and-drop
