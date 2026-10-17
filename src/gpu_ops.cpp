// gpu_ops.cpp -- see gpu_ops.h.

#include "gpu_ops.h"

#include <stdio.h>

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

hipError_t gpu_add_rmsnorm(const float* x_in, const float* add, float* x_out, const float* w, float* y, int n, int T,
                           float eps, const qmatvec_q8* q8, hipStream_t stream) {
    if (!g_norm) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || (q8 && (n % 256 != 0 || (long long)n * T > q8->max_k)) || (add && x_out == x_in))
        return hipErrorInvalidValue;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* args[] = {(void*)&x_in, (void*)&add, &x_out, (void*)&w, &y, &n, &eps, &o};
    return hipModuleLaunchKernel(g_norm, (unsigned)((n + 255) / 256), (unsigned)T, 1, 256, 1, 1, 0, stream, args,
                                 nullptr);
}

hipError_t gpu_swiglu(const float* g, const float* u, float* out, int n, const qmatvec_q8* q8, hipStream_t stream) {
    if (!g_swiglu) return hipErrorInvalidDeviceFunction;
    if (q8 && (n % 256 != 0 || n > q8->max_k)) return hipErrorInvalidValue;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* args[] = {(void*)&g, (void*)&u, &out, &n, &o};
    return hipModuleLaunchKernel(g_swiglu, (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}
