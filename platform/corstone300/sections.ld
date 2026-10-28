/* Shared section layout: code in ITCM, data/heap/stack in DTCM, blobs in DDR. */
ENTRY(Reset_Handler)
SECTIONS {
    .text : {
        KEEP(*(.vectors))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
        KEEP(*(.init)) KEEP(*(.fini))
        __init_array_start = .; KEEP(*(.init_array*)) __init_array_end = .;
        __preinit_array_start = .; KEEP(*(.preinit_array*)) __preinit_array_end = .;
        __fini_array_start = .; KEEP(*(.fini_array*)) __fini_array_end = .;
    } > ITCM
    .ARM.exidx : { *(.ARM.exidx*) } > ITCM

    .data : ALIGN(4) {
        __data_start = .;
        *(.data*)
        . = ALIGN(4);
        __data_end = .;
    } > DTCM AT > ITCM
    __data_load = LOADADDR(.data);

    .bss (NOLOAD) : ALIGN(4) {
        __bss_start = .;
        *(.bss*) *(COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > DTCM
    end = .;                                      /* newlib's heap starts here */
    __stack_top = ORIGIN(DTCM) + LENGTH(DTCM);
    __stack_limit = __stack_top - 16K;

    .blobs : ALIGN(16) { KEEP(*(.blobs)) } > DDR
}
