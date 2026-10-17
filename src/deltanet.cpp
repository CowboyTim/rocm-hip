// deltanet.cpp -- see deltanet.h.

#include "deltanet.h"

#include <stdio.h>

static hipModule_t g_mod = nullptr;
static hipFunction_t g_pre = nullptr, g_conv = nullptr, g_conv_state = nullptr, g_rule = nullptr, g_norm = nullptr;

// Columns of a head per deltanet_rule workgroup: RULE_CW in deltanet.hip
static const int kRuleCols = 16;

int deltanet_init(const char* co_path) {
    if (g_mod) return 0;
    if (hipModuleLoad(&g_mod, co_path) != hipSuccess) {
        fprintf(stderr, "[deltanet] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    if (hipModuleGetFunction(&g_pre, g_mod, "deltanet_pre") != hipSuccess ||
        hipModuleGetFunction(&g_conv, g_mod, "deltanet_conv") != hipSuccess ||
        hipModuleGetFunction(&g_conv_state, g_mod, "deltanet_conv_state") != hipSuccess ||
        hipModuleGetFunction(&g_rule, g_mod, "deltanet_rule") != hipSuccess ||
        hipModuleGetFunction(&g_norm, g_mod, "deltanet_norm") != hipSuccess) {
        fprintf(stderr, "[deltanet] kernels missing in %s\n", co_path);
        deltanet_free();
        return -1;
    }
    return 0;
}

void deltanet_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    g_pre = g_conv = g_conv_state = g_rule = g_norm = nullptr;
}

hipError_t deltanet_pre(const float* qkv, float* conv_state, const float* conv_w, int d_conv, float* conv_out,
                        int n_k, int n_v, const float* alpha, const float* beta_in, const float* dt_bias,
                        const float* ssm_a, float* g_out, float* beta_out, float eps, int T, hipStream_t stream) {
    if (!g_pre) return hipErrorInvalidDeviceFunction;
    if (d_conv < 2 || d_conv > 8 || T <= 0) return hipErrorInvalidValue;
    int n_qk = 2 * n_k;
    const unsigned heads = (unsigned)(2 * n_k + n_v);
    if (T == 1) {
        void* args[] = {(void*)&qkv, &conv_state, (void*)&conv_w, &d_conv, &conv_out, &n_qk, (void*)&alpha,
                        (void*)&beta_in, (void*)&dt_bias, (void*)&ssm_a, &g_out, &beta_out, &n_v, &eps, &T};
        return hipModuleLaunchKernel(g_pre, heads, 1, 1, DELTANET_HEAD_SIZE, 1, 1, 0, stream, args, nullptr);
    }
    // Several tokens: all in parallel, then advance the conv state
    void* args[] = {(void*)&qkv, &conv_state, (void*)&conv_w, &d_conv, &conv_out, &n_qk, (void*)&alpha,
                    (void*)&beta_in, (void*)&dt_bias, (void*)&ssm_a, &g_out, &beta_out, &n_v, &eps};
    hipError_t e = hipModuleLaunchKernel(g_conv, heads, (unsigned)T, 1, DELTANET_HEAD_SIZE, 1, 1, 0, stream, args,
                                         nullptr);
    if (e != hipSuccess) return e;
    void* sargs[] = {(void*)&qkv, &conv_state, &d_conv, &T};
    return hipModuleLaunchKernel(g_conv_state, heads, 1, 1, DELTANET_HEAD_SIZE, 1, 1, 0, stream, sargs, nullptr);
}

hipError_t deltanet_rule(float* state, const float* conv_out, const float* g, const float* beta, const float* z,
                         const float* norm_w, float* out, int n_k, int n_v, float eps, int T, const qmatvec_q8* q8,
                         hipStream_t stream) {
    if (!g_rule) return hipErrorInvalidDeviceFunction;
    const long long n = (long long)T * n_v * DELTANET_HEAD_SIZE;
    if (T <= 0 || n % 256 != 0 || (q8 && n > q8->max_k)) return hipErrorInvalidValue;
    void* args[] = {&state, (void*)&conv_out, (void*)&g, (void*)&beta, &out, &n_k, &T};
    hipError_t e = hipModuleLaunchKernel(g_rule, (unsigned)n_v, DELTANET_HEAD_SIZE / kRuleCols, 1, 128, 1, 1, 0, stream,
                                         args, nullptr);
    if (e != hipSuccess) return e;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* nargs[] = {&out, (void*)&z, (void*)&norm_w, &eps, &o};
    return hipModuleLaunchKernel(g_norm, (unsigned)(n / 256), 1, 1, 256, 1, 1, 0, stream, nargs, nullptr);
}
