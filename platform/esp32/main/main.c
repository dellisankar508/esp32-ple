// ESP32-D0WD: boot benchmark (same method as every other target), then the OLED demo.
// Boot: model checksum, flash benchmark, golden check single-core and dual-core.
// Demo: stats screen, BOOT button streams a story, BOOT again returns to stats.
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_cpu.h"
#include "sdkconfig.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "arena.h"
#include "model_map.h"
#include "ple.h"
#include "tokenizer.h"
#include "bench.h"
#include "kern.h"
#include "worker.h"
#include "oled.h"
#include "font_body.h"
#include "font_title.h"

static const char *TAG = "ple";
#define CTX        128
#define PIN_SDA    21
#define PIN_SCL    22
#define PIN_BOOT   0
#define MAX_TOKENS 120

extern const uint8_t tokenizer_bin_start[] asm("_binary_tokenizer_bin_start");
extern const uint8_t tokenizer_bin_end[]   asm("_binary_tokenizer_bin_end");
extern const char    golden_txt_start[]    asm("_binary_golden_txt_start");

static uint32_t now_us(void) { return (uint32_t)esp_timer_get_time(); }

static ple_model_t m; static ple_state_t s; static tokenizer_t tk;
static float last_tps = 0, last_ms = 0;

static void show_stats(float tps, float ms) {
    char buf[32];
    const int lh = FONT_BODY_HEIGHT + 1;
    oled_clear();
    oled_text(&font_title, 0, 0, "ESP32  PLE LLM");
    oled_hline(14);
    oled_text(&font_body, 0, 17, "4.4M params, int4");
    oled_text(&font_body, 0, 17 + lh, "2.4MB flash, no PSRAM");
    snprintf(buf, sizeof buf, "%uKB RAM state", (unsigned)(ple_state_bytes(&m, CTX) / 1024));
    oled_text(&font_body, 0, 17 + 2 * lh, buf);
    if (tps > 0) snprintf(buf, sizeof buf, "%.1f tok/s   %.0f ms/tok", tps, ms);
    else         snprintf(buf, sizeof buf, "press BOOT for a story");
    oled_text(&font_body, 0, OLED_H - FONT_BODY_HEIGHT, buf);
    oled_flush();
}

static int ends_sentence(int id) {
    int len; const uint8_t *b = tokenizer_bytes(&tk, id, &len);
    return len && (b[len - 1] == '.' || b[len - 1] == '!' || b[len - 1] == '?');
}

static void emit(int id) {
    int len; const uint8_t *b = tokenizer_bytes(&tk, id, &len);
    fwrite(b, 1, len, stdout); fflush(stdout);
    con_put(b, len);
}

static void tell_story(const char *text) {
    int prompt[16];
    int np = tokenizer_encode(&tk, text, prompt, 16);
    oled_clear(); con_reset(&font_body, 0, 52); oled_hline(53);
    oled_text(&font_body, 0, OLED_H - FONT_BODY_HEIGHT, "generating...");
    printf("\n--- "); emit(prompt[0]);
    uint32_t rng = (uint32_t)esp_timer_get_time();
    int pos = 0, tok = prompt[0];
    int64_t t0 = esp_timer_get_time();
    for (int i = 0; pos < CTX - 1 && pos < MAX_TOKENS; i++) {
        ple_forward(&m, &s, tok, pos, NULL);
        int nxt = (i + 1 < np) ? prompt[i + 1] : ple_sample(s.logits, m.V, 0.8f, &rng);
        if (nxt == 0) break;
        emit(nxt); tok = nxt; pos++;
        if (pos > 90 && ends_sentence(nxt)) break;
        if (!gpio_get_level(PIN_BOOT)) break;               // button = stop early
    }
    int64_t dt = esp_timer_get_time() - t0;
    last_tps = pos * 1e6f / dt; last_ms = dt / 1000.0f / pos;
    char buf[32]; snprintf(buf, sizeof buf, "%d tok   %.1f tok/s", pos, last_tps);
    oled_clear_rect(OLED_H - FONT_BODY_HEIGHT, OLED_H);
    oled_text(&font_body, 0, OLED_H - FONT_BODY_HEIGHT, buf); oled_flush();
    printf("\n[%d tokens, %.2f tok/s]\n", pos, last_tps);
}

// CPU clock measured: cycle counter against the microsecond timer over 20 ms.
static float measure_cpu_mhz(void) {
    uint32_t c0 = esp_cpu_get_cycle_count();
    int64_t t0 = esp_timer_get_time(), t1;
    while ((t1 = esp_timer_get_time()) - t0 < 20000) { }
    uint32_t c1 = esp_cpu_get_cycle_count();
    return (float)(c1 - c0) / (float)(t1 - t0);
}

// Flash mode and clock, and the optimisation level, as configured in sdkconfig.
#if defined(CONFIG_ESPTOOLPY_FLASHMODE_QIO)
#define FLASH_MODE "QIO"
#elif defined(CONFIG_ESPTOOLPY_FLASHMODE_QOUT)
#define FLASH_MODE "QOUT"
#elif defined(CONFIG_ESPTOOLPY_FLASHMODE_DIO)
#define FLASH_MODE "DIO"
#else
#define FLASH_MODE "DOUT"
#endif
#if defined(CONFIG_ESPTOOLPY_FLASHFREQ_80M)
#define FLASH_FREQ "80 MHz"
#elif defined(CONFIG_ESPTOOLPY_FLASHFREQ_40M)
#define FLASH_FREQ "40 MHz"
#elif defined(CONFIG_ESPTOOLPY_FLASHFREQ_26M)
#define FLASH_FREQ "26 MHz"
#else
#define FLASH_FREQ "20 MHz"
#endif
#if defined(CONFIG_COMPILER_OPTIMIZATION_PERF)
#define OPT_FLAGS "-O2"
#elif defined(CONFIG_COMPILER_OPTIMIZATION_SIZE)
#define OPT_FLAGS "-Os"
#elif defined(CONFIG_COMPILER_OPTIMIZATION_NONE)
#define OPT_FLAGS "-O0"
#else
#define OPT_FLAGS "-Og"
#endif

static void run_bench(void) {
    bench_golden_t g;
    if (bench_parse_golden(golden_txt_start, &g)) { ESP_LOGW(TAG, "no golden.txt, skipping benchmark"); return; }
    bench_platform_t plat = {
        .board = "esp32",
        .cpu_mhz = measure_cpu_mhz(),
        .flash_desc = FLASH_MODE " " FLASH_FREQ,
        .cflags = OPT_FLAGS,
        .now = now_us,
        .parallel = worker_parallel,
        .reps = 3,
    };
    bench_run_all(&plat, &m, &s, &tk, &g);
}

static int button_pressed(void) {                           // debounced, waits for release
    if (gpio_get_level(PIN_BOOT)) return 0;
    vTaskDelay(pdMS_TO_TICKS(30));
    if (gpio_get_level(PIN_BOOT)) return 0;
    while (!gpio_get_level(PIN_BOOT)) vTaskDelay(pdMS_TO_TICKS(10));
    return 1;
}

void app_main(void) {
    arena_init(8 * 1024);
    size_t msize; const uint8_t *mb = model_map(&msize);
    if (ple_load(&m, mb, msize)) { ESP_LOGE(TAG, "ple_load failed - run flash_model.sh"); return; }
    if (tokenizer_load(&tk, tokenizer_bin_start, tokenizer_bin_end - tokenizer_bin_start, arena_alloc)) return;
    if (ple_state_init(&s, &m, CTX, arena_alloc)) { ESP_LOGE(TAG, "state alloc failed"); return; }
    worker_start();
    arena_report();
    run_bench();
    s.parallel = worker_parallel;

    gpio_config_t io = { .pin_bit_mask = 1ULL << PIN_BOOT, .mode = GPIO_MODE_INPUT, .pull_up_en = 1 };
    gpio_config(&io);
    if (oled_init(PIN_SDA, PIN_SCL, 0x3C)) ESP_LOGW(TAG, "OLED not found, serial only");
    show_stats(0, 0);

    // Stats screen <-> story, toggled by BOOT. First press starts a story.
    int showing_story = 0;
    for (;;) {
        if (button_pressed()) {
            if (!showing_story) { tell_story("Once upon a time"); showing_story = 1; }
            else { show_stats(last_tps, last_ms); showing_story = 0; }
        }
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}
