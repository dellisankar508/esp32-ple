#pragma once
// Where hot code lives. Weights stream through the flash cache, so the kernels
// must not: on the ESP32 they go to IRAM, on the RP2350 to SRAM (.time_critical).
#if defined(ESP_PLATFORM)
#include "esp_attr.h"
#define PLE_HOT IRAM_ATTR
#elif defined(PLE_HOT_SECTION)
#define PLE_HOT __attribute__((section(PLE_HOT_SECTION)))
#else
#define PLE_HOT
#endif
