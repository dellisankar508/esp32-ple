/* Model, tokenizer, golden file and kernel test vectors, embedded in the ELF in the
 * big memory region (.blobs: DDR on Corstone-300). The Makefile passes each path as a
 * quoted string (PLE_MODEL_BIN etc.). */

    .section .blobs, "a"
    .balign 16
    .global ple_model_start, ple_model_end
ple_model_start:
    .incbin PLE_MODEL_BIN
ple_model_end:

    .balign 16
    .global ple_kvec_start, ple_kvec_end
ple_kvec_start:
    .incbin PLE_KVEC_BIN
ple_kvec_end:

    .balign 4
    .global ple_tok_start, ple_tok_end
ple_tok_start:
    .incbin PLE_TOK_BIN
ple_tok_end:

    .global ple_golden_start
ple_golden_start:
    .incbin PLE_GOLDEN_TXT
    .byte 0
