// attention.cpp -- see attention.h.

#include "inference_log.h"
#include "kernel_loader.h"
#include "attention.h"

#include <algorithm>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static hipModule_t g_mod = nullptr;
// [0]: head_dim 128 (attn_*_128), [1]: 256
static hipFunction_t g_prep[2], g_partial[2], g_combine[2], g_flash[2];

// attn_flash takes batches of at least this many tokens. QMATVEC_NO_WMMA (qmatvec.h) also turns it off, so
// that prefill can be compared bit for bit with token by token.
static const int kFlashMinT = 16;
static bool g_flash_off = false;

int attention_init(const char* co_path) {
    if (g_mod) return 0;
    if (kernel_module_load(&g_mod, co_path) != hipSuccess) {
        INFER_LOG("[attention] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    for (int i = 0; i < 2; ++i) {
        const int d = i ? 256 : 128;
        char n[4][32];
        snprintf(n[0], sizeof(n[0]), "attn_prep_%d", d);
        snprintf(n[1], sizeof(n[1]), "attn_partial_%d", d);
        snprintf(n[2], sizeof(n[2]), "attn_combine_%d", d);
        snprintf(n[3], sizeof(n[3]), "attn_flash_%d", d);
        if (hipModuleGetFunction(&g_prep[i], g_mod, n[0]) != hipSuccess ||
            hipModuleGetFunction(&g_partial[i], g_mod, n[1]) != hipSuccess ||
            hipModuleGetFunction(&g_combine[i], g_mod, n[2]) != hipSuccess ||
            hipModuleGetFunction(&g_flash[i], g_mod, n[3]) != hipSuccess) {
            INFER_LOG("[attention] kernels missing in %s\n", co_path);
            attention_free();
            return -1;
        }
    }
    g_flash_off = getenv("QMATVEC_NO_WMMA") != nullptr || getenv("ATTN_NO_FLASH") != nullptr;  // A/B: attention only
    return 0;
}

void attention_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    for (int i = 0; i < 2; ++i) g_prep[i] = g_partial[i] = g_combine[i] = g_flash[i] = nullptr;
}

hipError_t attention_forward(const float* qfull, const float* k, const float* v, const float* q_norm,
                             const float* k_norm, __half* K, __half* V, int kv_shift, long long kv_bstride, int n_q,
                             int n_kv, int n_rot, int pos0, int T,
                             float freq_base, float eps, float* q, float* part_ml, float* part_acc, float* out,
                             int head_dim, int gated, const qmatvec_q8* q8, hipStream_t stream) {
    const int hi = head_dim == 256 ? 1 : head_dim == 128 ? 0 : -1, D = head_dim;
    if (hi < 0) return hipErrorInvalidValue;
    if (!g_prep[hi]) return hipErrorInvalidDeviceFunction;
    const int HP = 256 / D;  // heads per attn_combine workgroup (one 256-value block)
    if (kv_shift < 0 || kv_shift > 30 || n_kv <= 0 || n_q % n_kv != 0 || n_q % HP != 0 || n_rot > D ||
        n_rot % 2 != 0 || pos0 < 0 || T <= 0)
        return hipErrorInvalidValue;
    if (q8 && (long long)T * n_q * D > q8->max_k) return hipErrorInvalidValue;
    int kv_stride = n_kv * D;
    const int n_last = pos0 + T;  // positions the last token attends to

    void* a_prep[] = {(void*)&qfull, (void*)&k, (void*)&v, (void*)&q_norm, (void*)&k_norm, &q, &K, &V, &kv_stride,
                      &kv_shift, &kv_bstride, &n_q, &n_kv, &n_rot, &pos0, &freq_base, &eps, &gated};
    hipError_t e = hipModuleLaunchKernel(g_prep[hi], (unsigned)(n_q + n_kv), (unsigned)T, 1, (unsigned)D, 1, 1, 0,
                                         stream,
                                         a_prep, nullptr);
    if (e != hipSuccess) return e;

    // Partial-result slots the caller's buffers hold (attention_part_slots with n_ctx >= n_last)
    const int nc = (n_last + ATTN_CHUNK - 1) / ATTN_CHUNK;
    const long long slots = std::max<long long>(nc, (long long)T * std::min(nc, ATTN_MAX_SPLITS));
    float scale = 1.0f / sqrtf((float)D);
    const float* qc = q;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);

    if (T >= kFlashMinT && !g_flash_off && (n_q / n_kv) % 4 == 0) {
        // Up to ATTN_MAX_SPLITS ranges per token, each a multiple of 16 positions
        int n_split = std::min(nc, ATTN_MAX_SPLITS);
        int chunk = ((n_last + n_split - 1) / n_split + 15) / 16 * 16;
        n_split = (n_last + chunk - 1) / chunk;
        void* a_flash[] = {(void*)&qc, &K, &V, &kv_stride, &kv_shift, &kv_bstride, &pos0, &T, &n_q, &n_kv, &chunk, &scale, &part_ml, &part_acc};
        e = hipModuleLaunchKernel(g_flash[hi], (unsigned)(n_q / 4), (unsigned)((T + 15) / 16), (unsigned)n_split,
                                  128, 1,
                                  1, 0, stream, a_flash, nullptr);
        if (e != hipSuccess) return e;
        void* a_comb[] = {&part_ml, &part_acc, &n_split, &n_q, (void*)&qfull, &out, &o, &gated};
        return hipModuleLaunchKernel(g_combine[hi], (unsigned)(n_q / HP), (unsigned)T, 1, 256, 1, 1, 0, stream, a_comb,
                                     nullptr);
    }

    // attn_partial: one 256-position chunk per slot, so the tokens run in groups of slots / nc
    const int group = (int)std::max<long long>(1, slots / nc);
    for (int z0 = 0; z0 < T; z0 += group) {
        const int tg = std::min(group, T - z0), p0 = pos0 + z0;
        const int n_chunks = (p0 + tg + ATTN_CHUNK - 1) / ATTN_CHUNK;
        const float* qz = qc + (size_t)z0 * n_q * D;
        const float* qfz = qfull + (size_t)z0 * (gated ? 2 : 1) * n_q * D;
        float* outz = out + (size_t)z0 * n_q * D;
        qmatvec_q8_out oz = o;  // quantized blocks of tokens z0 ..: one 256-value block per HP heads
        if (oz.qK) {
            const size_t blk = (size_t)z0 * n_q / HP;
            oz.q0 += blk * 256;
            oz.qK += blk * 256;
            oz.d0 += blk * 8;
            oz.dK += blk;
            oz.bsK += blk * 16;
        }
        void* a_part[] = {(void*)&qz, &K, &V, &kv_stride, &kv_shift, &kv_bstride, (void*)&p0, &n_q, &n_kv, &scale, &part_ml, &part_acc};
        e = hipModuleLaunchKernel(g_partial[hi], (unsigned)n_q, (unsigned)n_chunks, (unsigned)tg, ATTN_CHUNK, 1, 1, 0,
                                  stream, a_part, nullptr);
        if (e != hipSuccess) return e;
        int nch = n_chunks;
        void* a_comb[] = {&part_ml, &part_acc, &nch, &n_q, (void*)&qfz, &outz, &oz, &gated};
        e = hipModuleLaunchKernel(g_combine[hi], (unsigned)(n_q / HP), (unsigned)tg, 1, 256, 1, 1, 0, stream, a_comb,
                                  nullptr);
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}
