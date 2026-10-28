/* QEMU mps2-an505 (Cortex-M33 with the DSP extension): used only to check the Pico's
 * dsp kernel in emulation. Secure aliases of SSRAM1 (code + model) and SSRAM2/3 (data).
 * The names match sse300.ld so both share sections.ld. */
MEMORY {
    ITCM (rx)  : ORIGIN = 0x10000000, LENGTH = 512K
    DDR  (rwx) : ORIGIN = 0x10080000, LENGTH = 3584K
    DTCM (rwx) : ORIGIN = 0x38000000, LENGTH = 2048K
}
INCLUDE sections.ld
