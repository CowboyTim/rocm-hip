// gpu_ops.cpp -- see gpu_ops.h.

#include "gpu_ops.h"

#include <stdio.h>

#define NORM_THREADS 1024  // must match gpu_ops.hip

static hipModule_t g_mod = nullptr;
static hipFunction_t g_norm = nullptr, g_swiglu = nullptr;

int gpu_ops_init(const char* co_path) {
    if (g_mod) return 0;
    if (hipModuleLoad(&g_mod, co_path) != hipSuccess) {
        fprintf(stderr, "[gpu_ops] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    if (hipModuleGetFunction(&g_norm, g_mod, "add_rmsnorm") != hipSuccess ||
        hipModuleGetFunction(&g_swiglu, g_mod, "swiglu") != hipSuccess) {
        fprintf(stderr, "[gpu_ops] kernels missing in %s\n", co_path);
        gpu_ops_free();
        return -1;
    }
    return 0;
}

void gpu_ops_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    g_norm = g_swiglu = nullptr;
}

hipError_t gpu_add_rmsnorm(float* x, const float* add, const float* w, float* y, int n, float eps, hipStream_t stream) {
    if (!g_norm) return hipErrorInvalidDeviceFunction;
    void* args[] = {&x, (void*)&add, (void*)&w, &y, &n, &eps};
    return hipModuleLaunchKernel(g_norm, 1, 1, 1, NORM_THREADS, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_swiglu(const float* g, const float* u, float* out, int n, hipStream_t stream) {
    if (!g_swiglu) return hipErrorInvalidDeviceFunction;
    void* args[] = {(void*)&g, (void*)&u, &out, &n};
    return hipModuleLaunchKernel(g_swiglu, (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}
