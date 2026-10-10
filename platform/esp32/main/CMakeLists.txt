# The portable runtime lives in core/ and kernels/ at the repo root; the model
# files come from model/ (override with -DPLE_MODEL_DIR=... on idf.py).
set(ROOT ${CMAKE_CURRENT_LIST_DIR}/../../..)
if(NOT DEFINED PLE_MODEL_DIR)
    set(PLE_MODEL_DIR ${ROOT}/model)
endif()
if(NOT DEFINED PLE_KERNEL)
    set(PLE_KERNEL scalar)
endif()

idf_component_register(
    SRCS "main.c" "arena.c" "model_map.c" "worker.c" "oled.c"
         "${ROOT}/core/ple.c" "${ROOT}/core/tokenizer.c" "${ROOT}/core/bench.c"
         "${ROOT}/kernels/${PLE_KERNEL}.c"
    INCLUDE_DIRS "." "${ROOT}/core" "${ROOT}/kernels"
    REQUIRES esp_partition esp_timer spi_flash esp_driver_i2c esp_driver_gpio esp_hw_support
    EMBED_FILES "${PLE_MODEL_DIR}/tokenizer.bin"
    EMBED_TXTFILES "${PLE_MODEL_DIR}/golden.txt")
