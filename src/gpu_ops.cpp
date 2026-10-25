// gpu_ops.cpp -- see gpu_ops.h.

#include "inference_log.h"
#include "kernel_loader.h"
#include "gpu_ops.h"

#include <stdio.h>

static hipModule_t g_mod = nullptr;
static hipFunction_t g_norm_rows = nullptr;
static hipFunction_t g_norm = nullptr, g_swiglu = nullptr, g_router = nullptr, g_combine = nullptr, g_group = nullptr;
static hipFunction_t g_hc_norm = nullptr, g_hc_lowrank = nullptr, g_hc_mix = nullptr, g_ple_gate = nullptr,
                     g_ple_conv = nullptr, g_mtp_concat = nullptr;
static hipFunction_t g_add_normed = nullptr, g_geglu = nullptr, g_softcap = nullptr;

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
        hipModuleGetFunction(&g_group, g_mod, "moe_group") != hipSuccess ||
        hipModuleGetFunction(&g_hc_norm, g_mod, "hc_norm") != hipSuccess ||
        hipModuleGetFunction(&g_hc_lowrank, g_mod, "hc_lowrank") != hipSuccess ||
        hipModuleGetFunction(&g_hc_mix, g_mod, "hc_mix") != hipSuccess ||
        hipModuleGetFunction(&g_ple_gate, g_mod, "ple_gate") != hipSuccess ||
        hipModuleGetFunction(&g_ple_conv, g_mod, "ple_conv") != hipSuccess ||
        hipModuleGetFunction(&g_mtp_concat, g_mod, "mtp_concat") != hipSuccess ||
        hipModuleGetFunction(&g_add_normed, g_mod, "add_normed") != hipSuccess ||
        hipModuleGetFunction(&g_geglu, g_mod, "geglu") != hipSuccess ||
        hipModuleGetFunction(&g_softcap, g_mod, "softcap") != hipSuccess) {
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
    g_hc_norm = g_hc_lowrank = g_hc_mix = g_ple_gate = g_ple_conv = g_mtp_concat = nullptr;
    g_add_normed = g_geglu = g_softcap = nullptr;
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
    if (q8 && (n % 32 != 0 || n > q8->max_k)) return hipErrorInvalidValue;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* args[] = {(void*)&g, (void*)&u, &out, &n, &o};
    return hipModuleLaunchKernel(g_swiglu, (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_moe_router(const float* logits, const float* x, const float* w_shexp, int* ids, float* wts, float* sg,
                          int n_expert, int n_used, int n, int T, hipStream_t stream, int ids_stride) {
    if (!g_router) return hipErrorInvalidDeviceFunction;
    if (n_expert <= 0 || n_expert > 1024 || n_used <= 0 || n_used > 64 || n_used > n_expert || T <= 0)
        return hipErrorInvalidValue;
    if (ids_stride == 0) ids_stride = n_used;
    if (ids_stride < n_used) return hipErrorInvalidValue;
    void* args[] = {(void*)&logits, (void*)&x, (void*)&w_shexp, &ids, &wts, &sg, &n_expert, &n_used, &n, &ids_stride};
    return hipModuleLaunchKernel(g_router, (unsigned)T, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_moe_combine(const float* down, const float* wts, const float* shared, const float* sg, float* out,
                           int n_used, int n, int T, hipStream_t stream) {
    return gpu_moe_combine_scaled(down, wts, shared, sg, nullptr, nullptr, out, n_used, n, T, stream);
}

hipError_t gpu_moe_combine_scaled(const float* down, const float* wts, const float* shared, const float* sg,
                                  const int* ids, const float* exp_scale, float* out, int n_used, int n, int T,
                                  hipStream_t stream) {
    if (!g_combine) return hipErrorInvalidDeviceFunction;
    if (exp_scale && !ids) return hipErrorInvalidValue;
    void* args[] = {(void*)&down, (void*)&wts, (void*)&shared, (void*)&sg, &out, &n_used, &n, (void*)&ids,
                    (void*)&exp_scale};
    return hipModuleLaunchKernel(g_combine, (unsigned)((n + 255) / 256), (unsigned)T, 1, 256, 1, 1, 0, stream, args,
                                 nullptr);
}

hipError_t gpu_moe_group(const int* ids, int T, int n_used, int n_expert, int tile, int* tok_rows, int* pair_rows,
                         int* n_tiles, int* tile_e, int* tile_start, int* tile_n, hipStream_t stream) {
    if (!g_group) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || n_used <= 0 || n_expert <= 0 || n_expert > 1024 || tile <= 0) return hipErrorInvalidValue;
    void* args[] = {(void*)&ids, &T, &n_used, &n_expert, &tile, &tok_rows, &pair_rows, &n_tiles, &tile_e, &tile_start,
                    &tile_n};
    return hipModuleLaunchKernel(g_group, 1, 1, 1, 1024, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_hc_norm(const float* res_in, float* res_out, const float* cur, const float* inject, const float* w,
                       float* y, int n, int hc, int T, float eps, const qmatvec_q8* q8, hipStream_t stream) {
    if (!g_hc_norm) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || hc <= 0 || n % 256 != 0 || n > 256 * 16 || (q8 && (long long)n * hc * T > q8->max_k))
        return hipErrorInvalidValue;
    // Few rows (generating): spread each over n / 256 workgroups, which needs res_out != res_in when combining
    const unsigned split = T <= 4 && (!cur || res_out != res_in) ? (unsigned)(n / 256) : 1u;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* args[] = {(void*)&res_in, &res_out, (void*)&cur, (void*)&inject, (void*)&w, &y, &n, &hc, &eps, &o};
    return hipModuleLaunchKernel(g_hc_norm, split, (unsigned)(T * hc), 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_hc_lowrank(const float* lo, float* out, int n, int np, int T, int hc, const qmatvec_q8* q8,
                          hipStream_t stream) {
    if (!g_hc_lowrank) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || np < n || (q8 && (np % 32 != 0 || (long long)np * T > q8->max_k))) return hipErrorInvalidValue;
    float scale = 1.0f / (float)hc;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* args[] = {(void*)&lo, &out, &n, &np, &T, &scale, &o};
    const long long total = (long long)np * T;
    return hipModuleLaunchKernel(g_hc_lowrank, (unsigned)((total + 255) / 256), 1, 1, 256, 1, 1, 0, stream, args,
                                 nullptr);
}

hipError_t gpu_hc_mix(const float* xn, const float* gate, float* out, int n, int hc, int T, const qmatvec_q8* q8,
                      hipStream_t stream) {
    if (!g_hc_mix) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || (q8 && (n % 256 != 0 || (long long)n * T > q8->max_k))) return hipErrorInvalidValue;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* args[] = {(void*)&xn, (void*)&gate, &out, &n, &hc, &o};
    return hipModuleLaunchKernel(g_hc_mix, (unsigned)((n + 255) / 256), (unsigned)T, 1, 256, 1, 1, 0, stream, args,
                                 nullptr);
}

hipError_t gpu_ple_gate(const float* key, const float* value, const float* res, const float* wk, const float* wq,
                        const float* wc, float* gated, float* normed, int n, int hc, int T, float eps,
                        hipStream_t stream) {
    if (!g_ple_gate) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || n % 256 != 0 || n > 256 * 16) return hipErrorInvalidValue;
    void* args[] = {(void*)&key, (void*)&value, (void*)&res, (void*)&wk, (void*)&wq, (void*)&wc, &gated, &normed,
                    &n, &hc, &eps};
    return hipModuleLaunchKernel(g_ple_gate, (unsigned)T, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_ple_conv(const float* normed, const float* gated, float* res, float* state, const void* w, int m, int T,
                        int K, int dil, const deltanet_mid_states* state_mid, hipStream_t stream) {
    if (!g_ple_conv) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || K < 1 || dil < 1) return hipErrorInvalidValue;
    deltanet_mid_states mid = {{nullptr, nullptr, nullptr}};
    if (state_mid) mid = *state_mid;
    void* args[] = {(void*)&normed, (void*)&gated, &res, &state, (void*)&w, &m, &T, &K, &dil, &mid};
    return hipModuleLaunchKernel(g_ple_conv, (unsigned)((m + 255) / 256), 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_mtp_concat(const float* e, const float* h, float* out, int n, int hc, int T, hipStream_t stream) {
    if (!g_mtp_concat) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || hc <= 0 || n <= 0) return hipErrorInvalidValue;
    void* args[] = {(void*)&e, (void*)&h, &out, &n, &hc};
    return hipModuleLaunchKernel(g_mtp_concat, (unsigned)((2 * n + 255) / 256), (unsigned)(T * hc), 1, 256, 1, 1, 0,
                                 stream, args, nullptr);
}

hipError_t gpu_add_normed(const float* base, const float* a, const float* w, const float* scale, float* out, int n,
                          int T, float eps, hipStream_t stream) {
    if (!g_add_normed) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || n % 256 != 0 || n > 256 * 16 || a == out) return hipErrorInvalidValue;  // 16 = NORM_MAXV
    void* args[] = {(void*)&base, (void*)&a, (void*)&w, (void*)&scale, &out, &n, &eps};
    return hipModuleLaunchKernel(g_add_normed, (unsigned)T, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_geglu(const float* gate, const float* up, float* out, int rows, int h, int gate_stride, int up_stride,
                     const qmatvec_q8* q8, hipStream_t stream) {
    if (!g_geglu) return hipErrorInvalidDeviceFunction;
    int n = rows * h;
    if (rows <= 0 || h <= 0 || gate_stride < h || up_stride < h || (q8 && (n % 32 != 0 || n > q8->max_k)))
        return hipErrorInvalidValue;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* args[] = {(void*)&gate, (void*)&up, &out, &n, &h, &gate_stride, &up_stride, &o};
    return hipModuleLaunchKernel(g_geglu, (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t gpu_softcap(float* x, int n, float cap, hipStream_t stream) {
    if (!g_softcap) return hipErrorInvalidDeviceFunction;
    if (n <= 0 || cap <= 0.0f) return hipErrorInvalidValue;
    void* args[] = {&x, &n, &cap};
    return hipModuleLaunchKernel(g_softcap, (unsigned)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}
