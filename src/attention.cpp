// attention.cpp -- see attention.h.

#include "attention.h"

#include <math.h>
#include <stdio.h>

static hipModule_t g_mod = nullptr;
static hipFunction_t g_prep = nullptr, g_partial = nullptr, g_combine = nullptr;

int attention_init(const char* co_path) {
    if (g_mod) return 0;
    if (hipModuleLoad(&g_mod, co_path) != hipSuccess) {
        fprintf(stderr, "[attention] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    if (hipModuleGetFunction(&g_prep, g_mod, "attn_prep") != hipSuccess ||
        hipModuleGetFunction(&g_partial, g_mod, "attn_partial") != hipSuccess ||
        hipModuleGetFunction(&g_combine, g_mod, "attn_combine") != hipSuccess) {
        fprintf(stderr, "[attention] kernels missing in %s\n", co_path);
        attention_free();
        return -1;
    }
    return 0;
}

void attention_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    g_prep = g_partial = g_combine = nullptr;
}

hipError_t attention_forward(const float* qfull, const float* k, const float* v, const float* q_norm,
                             const float* k_norm, __half* K, __half* V, int n_q, int n_kv, int n_rot, int pos0, int T,
                             float freq_base, float eps, float* q, float* part_ml, float* part_acc, float* out,
                             const qmatvec_q8* q8, hipStream_t stream) {
    if (!g_prep) return hipErrorInvalidDeviceFunction;
    if (n_kv <= 0 || n_q % n_kv != 0 || n_rot > ATTN_HEAD_DIM || n_rot % 2 != 0 || pos0 < 0 || T <= 0)
        return hipErrorInvalidValue;
    if (q8 && (long long)T * n_q * ATTN_HEAD_DIM > q8->max_k) return hipErrorInvalidValue;
    int kv_stride = n_kv * ATTN_HEAD_DIM;
    const int n_last = pos0 + T;  // positions the last token attends to

    void* a_prep[] = {(void*)&qfull, (void*)&k, (void*)&v, (void*)&q_norm, (void*)&k_norm, &q, &K, &V, &kv_stride,
                      &n_q, &n_kv, &n_rot, &pos0, &freq_base, &eps};
    hipError_t e = hipModuleLaunchKernel(g_prep, (unsigned)(n_q + n_kv), (unsigned)T, 1, ATTN_HEAD_DIM, 1, 1, 0, stream,
                                         a_prep, nullptr);
    if (e != hipSuccess) return e;

    int n_chunks = (n_last + ATTN_CHUNK - 1) / ATTN_CHUNK;
    float scale = 1.0f / sqrtf((float)ATTN_HEAD_DIM);
    const float* qc = q;
    void* a_part[] = {(void*)&qc, &K, &V, &kv_stride, &pos0, &n_q, &n_kv, &scale, &part_ml, &part_acc};
    e = hipModuleLaunchKernel(g_partial, (unsigned)n_q, (unsigned)n_chunks, (unsigned)T, ATTN_HEAD_DIM, 1, 1, 0, stream,
                              a_part, nullptr);
    if (e != hipSuccess) return e;

    qmatvec_q8_out o = qmatvec_q8_outputs(q8);
    void* a_comb[] = {&part_ml, &part_acc, &n_chunks, &n_q, (void*)&qfull, &out, &o};
    return hipModuleLaunchKernel(g_combine, (unsigned)n_q, (unsigned)T, 1, ATTN_HEAD_DIM, 1, 1, 0, stream, a_comb,
                                 nullptr);
}
