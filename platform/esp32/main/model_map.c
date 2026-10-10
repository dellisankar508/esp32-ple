#include "model_map.h"
#include "esp_partition.h"
#include "esp_log.h"

static const char *TAG = "model";

const uint8_t *model_map(size_t *size_out) {
    const esp_partition_t *p = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!p) { ESP_LOGE(TAG, "model partition not found"); return NULL; }

    const void *ptr = NULL;
    esp_partition_mmap_handle_t h;
    esp_err_t err = esp_partition_mmap(p, 0, p->size, ESP_PARTITION_MMAP_DATA, &ptr, &h);
    if (err != ESP_OK) { ESP_LOGE(TAG, "mmap failed: %s", esp_err_to_name(err)); return NULL; }

    ESP_LOGI(TAG, "partition @0x%06lx size %lu B mapped at %p",
             (unsigned long)p->address, (unsigned long)p->size, ptr);
    if (size_out) *size_out = p->size;
    return (const uint8_t *)ptr;
}
