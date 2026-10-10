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
set(PLE_KERNEL scalar CACHE STRING "Kernel backend in kernels/<name>.c")
set(PLE_OPT -O2 CACHE STRING "Optimisation flag for the PLE code (ESP32 uses -O2)")
option(PLE_COPY_TO_RAM "Run all code from SRAM so flash only streams weights" ON)

foreach(f model.bin tokenizer.bin golden.txt)
    if(NOT EXISTS "${PLE_MODEL_DIR}/${f}")
        message(FATAL_ERROR "Missing ${PLE_MODEL_DIR}/${f}. Copy it from your train folder, "
                            "or set -DPLE_MODEL_DIR=<dir>.")
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
    ${ROOT}/kernels/${PLE_KERNEL}.c
)
set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/blobs.S PROPERTIES OBJECT_DEPENDS
    "${PLE_MODEL_DIR_ABS}/model.bin;${PLE_MODEL_DIR_ABS}/tokenizer.bin;${PLE_MODEL_DIR_ABS}/golden.txt")

target_include_directories(ple_pico2 PRIVATE ${ROOT}/core ${ROOT}/kernels)
target_compile_definitions(ple_pico2 PRIVATE
    PLE_HOT_SECTION=".time_critical.ple"     # kernels in SRAM even without copy_to_ram
    PLE_CFLAGS="${PLE_OPT}"
)
target_compile_options(ple_pico2 PRIVATE ${PLE_OPT} -Wall)
target_link_libraries(ple_pico2 pico_stdlib pico_multicore hardware_dma)

if(PLE_COPY_TO_RAM)
    pico_set_binary_type(ple_pico2 copy_to_ram)
endif()

pico_enable_stdio_usb(ple_pico2 1)
pico_enable_stdio_uart(ple_pico2 0)
pico_add_extra_outputs(ple_pico2)       # .uf2 for drag-and-drop
