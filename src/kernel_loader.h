// kernel_loader.h -- loads the engine's GPU code objects by file name ("matvec_q.co", ...).
//
// By default from the copies embedded in the library (kernels_embed.S), so a host program needs no kernel files
// and no particular working directory. INFERENCE_KERNEL_DIR=<dir> loads <dir>/<name> instead (kernel
// development without relinking). A name that is not embedded is loaded from the working directory.

#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

hipError_t kernel_module_load(hipModule_t* module, const char* name);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_LOADER_H
