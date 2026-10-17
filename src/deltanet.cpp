// deltanet.cpp -- see deltanet.h.

#include "deltanet.h"

#include <stdio.h>

static hipModule_t g_mod = nullptr;
static hipFunction_t g_pre = nullptr, g_rule = nullptr;

int deltanet_init(const char* co_path) {
    if (g_mod) return 0;
    if (hipModuleLoad(&g_mod, co_path) != hipSuccess) {
        fprintf(stderr, "[deltanet] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    if (hipModuleGetFunction(&g_pre, g_mod, "deltanet_pre") != hipSuccess ||
        hipModuleGetFunction(&g_rule, g_mod, "deltanet_rule") != hipSuccess) {
        fprintf(stderr, "[deltanet] kernels missing in %s\n", co_path);
        deltanet_free();
        return -1;
    }
    return 0;
}

void deltanet_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    g_pre = g_rule = nullptr;
}

hipError_t deltanet_pre(const float* qkv, float* conv_state, const float* conv_w, int d_conv, float* conv_out,
                        int n_k, int n_v, const float* alpha, const float* beta_in, const float* dt_bias,
                        const float* ssm_a, float* g_out, float* beta_out, float eps, hipStream_t stream) {
    if (!g_pre) return hipErrorInvalidDeviceFunction;
    if (d_conv < 2 || d_conv > 8 || n_v > DELTANET_HEAD_SIZE) return hipErrorInvalidValue;
    int n_qk = 2 * n_k;
    void* args[] = {(void*)&qkv, &conv_state, (void*)&conv_w, &d_conv, &conv_out, &n_qk, (void*)&alpha,
                    (void*)&beta_in, (void*)&dt_bias, (void*)&ssm_a, &g_out, &beta_out, &n_v, &eps};
    return hipModuleLaunchKernel(g_pre, (unsigned)(2 * n_k + n_v), 1, 1, DELTANET_HEAD_SIZE, 1, 1, 0, stream, args,
                                 nullptr);
}

hipError_t deltanet_rule(float* state, const float* conv_out, const float* g, const float* beta, const float* z,
                         const float* norm_w, float* out, int n_k, int n_v, float eps, hipStream_t stream) {
    if (!g_rule) return hipErrorInvalidDeviceFunction;
    void* args[] = {&state, (void*)&conv_out, (void*)&g, (void*)&beta, (void*)&z, (void*)&norm_w, &out, &n_k, &eps};
    return hipModuleLaunchKernel(g_rule, (unsigned)n_v, 1, 1, DELTANET_HEAD_SIZE, 1, 1, 0, stream, args, nullptr);
}
