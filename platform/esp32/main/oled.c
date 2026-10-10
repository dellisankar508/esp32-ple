#include "oled.h"
#include <string.h>
#include <stdbool.h>
#include "driver/i2c_master.h"
#include "esp_log.h"

static const char *TAG = "oled";
static i2c_master_dev_handle_t dev;
static uint8_t fb[OLED_W * 8];       // page-major: fb[page*128 + x], bit = y within page
static uint8_t dirty;                // bit p = page p needs pushing

static int cmd(const uint8_t *c, int n) {
    uint8_t buf[16] = {0x00};
    memcpy(buf + 1, c, n);
    return i2c_master_transmit(dev, buf, n + 1, 100);
}

int oled_init(int sda, int scl, uint8_t addr) {
    i2c_master_bus_config_t bus = { .i2c_port = I2C_NUM_0, .sda_io_num = sda, .scl_io_num = scl,
        .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7, .flags.enable_internal_pullup = true };
    i2c_master_bus_handle_t bh;
    if (i2c_new_master_bus(&bus, &bh) != ESP_OK) return -1;
    i2c_device_config_t dc = { .dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = addr, .scl_speed_hz = 400000 };
    if (i2c_master_bus_add_device(bh, &dc, &dev) != ESP_OK) return -2;
    static const uint8_t init[] = {
        0xAE, 0xD5, 0x80, 0xA8, 0x3F, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00,
        0xA1, 0xC8, 0xDA, 0x12, 0x81, 0xCF, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0x2E, 0xAF };
    if (cmd(init, sizeof init) != ESP_OK) { ESP_LOGE(TAG, "no SSD1306 at 0x%02x", addr); return -3; }
    oled_clear(); oled_flush();
    return 0;
}

void oled_clear(void) { memset(fb, 0, sizeof fb); dirty = 0xFF; }

void oled_clear_rect(int y0, int y1) {
    for (int y = y0; y < y1; y++) {
        uint8_t mask = (uint8_t)~(1 << (y & 7)); uint8_t *p = fb + (y >> 3) * OLED_W;
        for (int x = 0; x < OLED_W; x++) p[x] &= mask;
        dirty |= 1 << (y >> 3);
    }
}

void oled_flush(void) {
    for (int page = 0; page < 8; page++) {
        if (!(dirty & (1 << page))) continue;
        uint8_t win[] = { 0x21, 0, 127, 0x22, (uint8_t)page, (uint8_t)page };
        cmd(win, sizeof win);
        uint8_t buf[OLED_W + 1]; buf[0] = 0x40;
        memcpy(buf + 1, fb + page * OLED_W, OLED_W);
        i2c_master_transmit(dev, buf, OLED_W + 1, 100);
    }
    dirty = 0;
}

static inline void px(int x, int y) {
    if ((unsigned)x < OLED_W && (unsigned)y < OLED_H) { fb[(y >> 3) * OLED_W + x] |= 1 << (y & 7); dirty |= 1 << (y >> 3); }
}

int oled_text_w(const font_t *f, const char *s) {
    int w = 0;
    for (; *s; s++) { int i = (uint8_t)*s - 32; if (i >= 0 && i < 95) w += f->w[i] + f->tracking; }
    return w;
}

int oled_text(const font_t *f, int x, int y, const char *s) {
    for (; *s; s++) {
        int i = (uint8_t)*s - 32;
        if (i < 0 || i >= 95) continue;
        const uint8_t *g = f->data + (size_t)f->off[i] * f->bpc;
        for (int cx = 0; cx < f->w[i]; cx++)
            for (int b = 0; b < f->bpc; b++) {
                uint8_t byte = g[cx * f->bpc + b];
                for (int bit = 0; byte; bit++, byte >>= 1)
                    if (byte & 1) px(x + cx, y + b * 8 + bit);
            }
        x += f->w[i] + f->tracking;
    }
    return x;
}

void oled_hline(int y) {
    uint8_t bit = 1 << (y & 7); uint8_t *p = fb + (y >> 3) * OLED_W;
    for (int x = 0; x < OLED_W; x++) p[x] |= bit;
    dirty |= 1 << (y >> 3);
}

// ------------------------------------------------------------ console
#define CON_MAX_LINES 6
#define CON_LINE_CH   40
static const font_t *cf;
static char lines[CON_MAX_LINES][CON_LINE_CH + 1];
static int c_y0, c_y1, c_nlines, c_lh, c_row, c_x;
static char word[CON_LINE_CH + 1]; static int wlen;
static int scrolled;

static void draw_row(int r) {
    int y = c_y0 + r * c_lh;
    oled_clear_rect(y, y + c_lh);
    oled_text(cf, 0, y, lines[r]);
}
static void redraw(void) {
    if (scrolled) { for (int r = 0; r < c_nlines; r++) draw_row(r); scrolled = 0; }
    else draw_row(c_row);
}
static void newline(void) {
    if (c_row < c_nlines - 1) c_row++;
    else { memmove(lines[0], lines[1], sizeof(lines[0]) * (c_nlines - 1)); memset(lines[c_nlines - 1], 0, sizeof lines[0]); scrolled = 1; }
    c_x = 0;
}
static void append(const char *s) { strncat(lines[c_row], s, CON_LINE_CH - strlen(lines[c_row])); c_x = oled_text_w(cf, lines[c_row]); }
static void put_word(void) {
    if (!wlen) return;
    word[wlen] = 0;
    int ww = oled_text_w(cf, word);
    int space = c_x ? oled_text_w(cf, " ") : 0;
    if (c_x + space + ww > OLED_W && c_x > 0) newline();
    else if (c_x) append(" ");
    append(word);
    wlen = 0;
}

void con_reset(const font_t *f, int y0, int y1) {
    cf = f; c_y0 = y0; c_y1 = y1; c_lh = f->height + 1;
    c_nlines = (y1 - y0) / c_lh; if (c_nlines > CON_MAX_LINES) c_nlines = CON_MAX_LINES;
    memset(lines, 0, sizeof lines); c_row = c_x = wlen = 0; scrolled = 1;
    oled_clear_rect(y0, y1); redraw();
}

void con_put(const uint8_t *b, int len) {
    int row0 = c_row;
    for (int i = 0; i < len; i++) {
        uint8_t ch = b[i];
        if (ch == ' ' || ch == '\n') { put_word(); if (ch == '\n' && c_x) newline(); }
        else if (ch >= 32 && ch < 127) { if (wlen < CON_LINE_CH) word[wlen++] = (char)ch; }
        else if (ch == 0x9C || ch == 0x9D) { if (wlen < CON_LINE_CH) word[wlen++] = '"'; }   // UTF-8 curly quotes -> "
    }
    put_word();
    if (c_row != row0 && !scrolled) draw_row(row0);
    redraw(); oled_flush();
}
