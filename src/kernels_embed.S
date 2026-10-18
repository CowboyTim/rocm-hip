// kernels_embed.S -- the engine's GPU code objects (build/co/*.co, from the .hip sources) embedded in the
// library, so it needs no files at run time (kernel_loader.cpp loads them with hipModuleLoadData). Assembled
// with -DKERNEL_DIR=<absolute build/co>: .incbin looks in the working directory first, which may hold stale
// copies, so the paths are spelled out.

#define STR_(x) #x
#define STR(x) STR_(x)
#define EMBED(sym, file)              \
    .balign 4096;                     \
    .global sym;                      \
sym:                                  \
    .incbin STR(KERNEL_DIR/file);     \
    .global sym##_end;                \
sym##_end:

    .section .rodata.inference_kernels, "a", @progbits
EMBED(inference_co_matvec_q, matvec_q.co)
EMBED(inference_co_gpu_ops, gpu_ops.co)
EMBED(inference_co_deltanet, deltanet.co)
EMBED(inference_co_attention, attention.co)

    .section .note.GNU-stack, "", @progbits
