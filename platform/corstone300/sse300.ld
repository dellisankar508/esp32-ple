/* Corstone-300 (SSE-300, Cortex-M55): the FVP and QEMU's mps3-an547. Secure aliases,
 * since both start the core in Secure state.
 *   ITCM 512 KB  code         DTCM 512 KB  data, heap, stack     DDR  model + test data */
MEMORY {
    ITCM (rx)  : ORIGIN = 0x10000000, LENGTH = 512K
    DTCM (rwx) : ORIGIN = 0x30000000, LENGTH = 512K
    DDR  (rwx) : ORIGIN = 0x70000000, LENGTH = 32M
}
INCLUDE sections.ld
