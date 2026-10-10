#pragma once
#include <stddef.h>
#include <stdint.h>

// Memory-maps the "model" data partition. Returns pointer to the raw bytes.
const uint8_t *model_map(size_t *size_out);
