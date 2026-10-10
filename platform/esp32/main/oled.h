#pragma once
#include <stdint.h>
// SSD1306 128x64 over I2C, 1 KB framebuffer, proportional bitmap fonts (see ttf2font.py).
typedef struct {
    uint8_t  height, bpc, tracking;   // px height, bytes per column, gap after each glyph
    const uint8_t  *w;                // [95] glyph widths
    const uint16_t *off;              // [95] column offsets into data
    const uint8_t  *data;             // column-major glyph bitmaps, bit0 = top row
} font_t;

#define OLED_W 128
#define OLED_H 64

int  oled_init(int sda, int scl, uint8_t addr);      // addr usually 0x3C
void oled_clear(void);
void oled_clear_rect(int y0, int y1);                 // clear rows [y0, y1)
void oled_flush(void);                                // push dirty pages only
int  oled_text_w(const font_t *f, const char *s);
int  oled_text(const font_t *f, int x, int y, const char *s);   // returns end x
void oled_hline(int y);

// Scrolling story console: pixel word-wrap, scrolls when full.
void con_reset(const font_t *f, int y0, int y1);      // region rows [y0, y1)
void con_put(const uint8_t *bytes, int len);          // append token bytes, flushes the panel
