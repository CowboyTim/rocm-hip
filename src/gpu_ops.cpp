// gpu_ops.cpp -- see gpu_ops.h.

#include "inference_log.h"
#include "kernel_loader.h"
#include "gpu_ops.h"

#include <stdio.h>

static hipModule_t g_mod = nullptr;
static hipFunction_t g_norm_rows = nullptr;
static hipFunction_t g_norm = nullptr, g_swiglu = nullptr, g_router = nullptr, g_combine = nullptr, g_group = nullptr;

int gpu_ops_init(const char* co_path) {
    if (g_mod) return 0;
    if (kernel_module_load(&g_mod, co_path) != hipSuccess) {
        INFER_LOG("[gpu_ops] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    if (hipModuleGetFunction(&g_norm, g_mod, "add_rmsnorm") != hipSuccess ||
        hipModuleGetFunction(&g_norm_rows, g_mod, "add_rmsnorm_rows") != hipSuccess ||
        hipModuleGetFunction(&g_swiglu, g_mod, "swiglu") != hipSuccess ||
        hipModuleGetFunction(&g_router, g_mod, "moe_router") != hipSuccess ||
        hipModuleGetFunction(&g_combine, g_mod, "moe_combine") != hipSuccess ||
        hipModuleGetFunction(&g_group, g_mod, "moe_group") != hipSuccess) {
        INFER_LOG("[gpu_ops] kernels missing in %s\n", co_path);
        gpu_ops_free();
        return -1;
    }
    return 0;
}

void gpu_ops_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    g_norm = g_norm_rows = g_swiglu = g_router = g_combine = g_group = nullptr;
}

hipError_t gpu_add_rmsnorm(const float* x_in, const float* add, float* x_out, const float* w, float* y, int n, int T,
                           float eps, const qmatvec_q8* q8, hipStream_t stream) {
    if (!g_norm) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || (q8 && (n % 256 != 0 || (long long)n * T > q8->max_k)) || (add && x_out == x_in))
        return hipErrorInvalidValue;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* args[] = {(void*)&x_in, (void*)&add, &x_out, (void*)&w, &y, &n, &eps, &o};
    // One token: spread the row over the GPU. Several: one workgroup per row (16 = NORM_MAXV in gpu_ops.hip).
    if (T > 1 && n % 256 == 0 && n <= 256 * 16)
        return hipModuleLaunchKernel(g_norm_rows, (unsigned)T, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
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

hipError_t gpu_moe_router(const float* logits, const float* x, const float* w_shexp, int* ids, float* wts, float* sg,
                          int n_expert, int n_used, int n, int T, hipStream_t stream) {
    if (!g_router) return hipErrorInvalidDeviceFunction;
    if (n_expert <= 0 || n_expert > 1024 || n_used <= 0 || n_used > 64 || n_used > n_expert || T <= 0)
        return hipErrorInvalidValue;
    void* args[] = {(void*)&logits, (void*)&x, (void*)&w_shexp, &ids, &wts, &sg, &n_expert, &n_used, &n};
    return hipModuleLaunchKernel(g_router, (unsigned)T, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_moe_combine(const float* down, const float* wts, const float* shared, const float* sg, float* out,
                           int n_used, int n, int T, hipStream_t stream) {
    if (!g_combine) return hipErrorInvalidDeviceFunction;
    void* args[] = {(void*)&down, (void*)&wts, (void*)&shared, (void*)&sg, &out, &n_used, &n};
    return hipModuleLaunchKernel(g_combine, (unsigned)((n + 255) / 256), (unsigned)T, 1, 256, 1, 1, 0, stream, args,
                                 nullptr);
}

hipError_t gpu_moe_group(const int* ids, int T, int n_used, int n_expert, int tile, int* tok_rows, int* pair_rows,
                         int* n_tiles, int* tile_e, int* tile_start, int* tile_n, hipStream_t stream) {
    if (!g_group) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || n_used <= 0 || n_expert <= 0 || n_expert > 1024 || tile <= 0) return hipErrorInvalidValue;
    void* args[] = {(void*)&ids, &T, &n_used, &n_expert, &tile, &tok_rows, &pair_rows, &n_tiles, &tile_e, &tile_start,
                    &tile_n};
    return hipModuleLaunchKernel(g_group, 1, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}
