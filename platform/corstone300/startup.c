// Minimal Armv8-M / Armv8.1-M startup: vector table, FPU + Helium enable, C runtime
// init, semihosting stdio (newlib librdimon), then main(). Runs in Secure state, as the
// Corstone-300 FVP and QEMU's mps3-an547 / mps2-an505 come out of reset.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern uint32_t __stack_top[], __data_load[], __data_start[], __data_end[], __bss_start[], __bss_end[];
extern int main(void);
extern void initialise_monitor_handles(void);

#define SCB_CPACR (*(volatile uint32_t *)0xE000ED88)
#define SCB_CFSR  (*(volatile uint32_t *)0xE000ED28)
#define SCB_HFSR  (*(volatile uint32_t *)0xE000ED2C)
#define SCB_BFAR  (*(volatile uint32_t *)0xE000ED38)

void Reset_Handler(void) {
    SCB_CPACR |= 0xFu << 20;                     // CP10 + CP11 full access: FPU and MVE
    __asm volatile("dsb\n isb" ::: "memory");
    memcpy(__data_start, __data_load, (size_t)((char *)__data_end - (char *)__data_start));
    memset(__bss_start, 0, (size_t)((char *)__bss_end - (char *)__bss_start));
    initialise_monitor_handles();
    exit(main());
}

void Fault_Handler(void) {
    printf("\nFAULT | CFSR 0x%08lx HFSR 0x%08lx BFAR 0x%08lx\n",
           (unsigned long)SCB_CFSR, (unsigned long)SCB_HFSR, (unsigned long)SCB_BFAR);
    exit(3);
}

typedef void (*vec_t)(void);
__attribute__((section(".vectors"), used)) const vec_t vectors[16] = {
    (vec_t)__stack_top, Reset_Handler,
    Fault_Handler, Fault_Handler, Fault_Handler, Fault_Handler,   // NMI, HardFault, MemManage, BusFault
    Fault_Handler, Fault_Handler,                                  // UsageFault, SecureFault
    0, 0, 0, Fault_Handler, Fault_Handler, 0, Fault_Handler, Fault_Handler,
};

// newlib's exit() runs __libc_fini_array, which wants these (normally from crti.o).
void _init(void) {}
void _fini(void) {}
