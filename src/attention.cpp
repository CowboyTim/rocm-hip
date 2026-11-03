// attention.cpp -- see attention.h.

#include "inference_log.h"
#include "kernel_loader.h"
#include "attention.h"

#include <algorithm>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static hipModule_t g_mod = nullptr;
// [0]: head_dim 128 (attn_*_128), [1]: 256, [2]: 512 (no attn_flash_512), [3]: 64
#define ATTN_NDIMS 4
static int attn_dim(int i) { return i < 3 ? 128 << i : 64; }
static hipFunction_t g_prep[ATTN_NDIMS], g_partial[ATTN_NDIMS], g_combine[ATTN_NDIMS], g_flash[ATTN_NDIMS];
static hipFunction_t g_partial_q8[ATTN_NDIMS], g_flash_q8[ATTN_NDIMS];
static hipFunction_t g_bidir[ATTN_NDIMS];  // attn_bidir_<D> (attention_opts.bidir)  // the same reading a Q8_0 KV cache (attention_opts.kv_q8)
// attn_flash_128_g2: 2 KV heads per workgroup, for a query / KV head ratio of 2 (Qwen3); attn_flash_256_h2t2
// (ratio 2: Gemma-4's sliding layers) and attn_flash_512_h2s2 (Gemma-4's global layers, D split in 2)
static hipFunction_t g_flash128_g2 = nullptr, g_flash256_h2t2 = nullptr, g_flash512_h2s2 = nullptr;
static hipFunction_t g_flash128_g2_q8 = nullptr, g_flash256_h2t2_q8 = nullptr, g_flash512_h2s2_q8 = nullptr;
// Other ratios: attn_flash_<D>_h2t2 (D 64 / 128, ratio a multiple of 2), attn_flash_<D>_h1t4 (D 64 / 128 / 256, any);
// [i] as g_prep, [k][1] the Q8_0-cache variants
static hipFunction_t g_flash_h2t2[ATTN_NDIMS][2], g_flash_h1t4[ATTN_NDIMS][2];
// attn_partial_gqa{8,16}[s]_<D> (D 64 / 128 / 256, F16 cache): attn_partial with the query heads of a KV head
// together; [i][0 / 1] up to 8 / 16 query heads per KV head, [i][2 / 3] the same over 64-position chunks
static hipFunction_t g_partial_gqa[ATTN_NDIMS][4];
static bool g_gqa_off = false;

// attn_flash takes batches of at least this many tokens. QMATVEC_NO_WMMA (qmatvec.h) also turns it off, so
// that prefill can be compared bit for bit with token by token.
static const int kFlashMinT = 16;
// attn_flash splits a token's positions until the launch has this many workgroups (4 per WGP of the 20)
static const long long kFlashWgs = 160;
static bool g_flash_off = false;

int attention_init(const char* co_path) {
    if (g_mod) return 0;
    if (kernel_module_load(&g_mod, co_path) != hipSuccess) {
        INFER_LOG("[attention] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    for (int i = 0; i < ATTN_NDIMS; ++i) {
        const int d = attn_dim(i);
        char n[6][40];
        snprintf(n[0], sizeof(n[0]), "attn_prep_%d", d);
        snprintf(n[1], sizeof(n[1]), "attn_partial_%d", d);
        snprintf(n[2], sizeof(n[2]), "attn_combine_%d", d);
        snprintf(n[3], sizeof(n[3]), "attn_flash_%d", d);
        snprintf(n[4], sizeof(n[4]), "attn_partial_%d_q8", d);
        snprintf(n[5], sizeof(n[5]), "attn_flash_%d_q8", d);
        if (hipModuleGetFunction(&g_prep[i], g_mod, n[0]) != hipSuccess ||
            hipModuleGetFunction(&g_partial[i], g_mod, n[1]) != hipSuccess ||
            hipModuleGetFunction(&g_combine[i], g_mod, n[2]) != hipSuccess ||
            hipModuleGetFunction(&g_partial_q8[i], g_mod, n[4]) != hipSuccess ||
            (d < 512 && (hipModuleGetFunction(&g_flash[i], g_mod, n[3]) != hipSuccess ||
                         hipModuleGetFunction(&g_flash_q8[i], g_mod, n[5]) != hipSuccess))) {
            INFER_LOG("[attention] kernels missing in %s\n", co_path);
            attention_free();
            return -1;
        }
    }
    for (int i = 0; i < ATTN_NDIMS; ++i) {
        char n[48];
        snprintf(n, sizeof(n), "attn_bidir_%d", attn_dim(i));
        if (hipModuleGetFunction(&g_bidir[i], g_mod, n) != hipSuccess) g_bidir[i] = nullptr;
        for (int q = 0; q < 2; ++q) {
            snprintf(n, sizeof(n), "attn_flash_%d_h2t2%s", attn_dim(i), q ? "_q8" : "");
            if (hipModuleGetFunction(&g_flash_h2t2[i][q], g_mod, n) != hipSuccess) g_flash_h2t2[i][q] = nullptr;
            snprintf(n, sizeof(n), "attn_flash_%d_h1t4%s", attn_dim(i), q ? "_q8" : "");
            if (hipModuleGetFunction(&g_flash_h1t4[i][q], g_mod, n) != hipSuccess) g_flash_h1t4[i][q] = nullptr;
            for (int sm = 0; sm < 2; ++sm) {
                snprintf(n, sizeof(n), "attn_partial_gqa%d%s_%d", q ? 16 : 8, sm ? "s" : "", attn_dim(i));
                hipFunction_t& f = g_partial_gqa[i][2 * sm + q];
                if (hipModuleGetFunction(&f, g_mod, n) != hipSuccess) f = nullptr;
            }
        }
    }
    if (hipModuleGetFunction(&g_flash128_g2, g_mod, "attn_flash_128_g2") != hipSuccess) g_flash128_g2 = nullptr;
    if (hipModuleGetFunction(&g_flash256_h2t2, g_mod, "attn_flash_256_h2t2") != hipSuccess) g_flash256_h2t2 = nullptr;
    if (hipModuleGetFunction(&g_flash512_h2s2, g_mod, "attn_flash_512_h2s2") != hipSuccess) g_flash512_h2s2 = nullptr;
    if (hipModuleGetFunction(&g_flash128_g2_q8, g_mod, "attn_flash_128_g2_q8") != hipSuccess) g_flash128_g2_q8 = nullptr;
    if (hipModuleGetFunction(&g_flash256_h2t2_q8, g_mod, "attn_flash_256_h2t2_q8") != hipSuccess)
        g_flash256_h2t2_q8 = nullptr;
    if (hipModuleGetFunction(&g_flash512_h2s2_q8, g_mod, "attn_flash_512_h2s2_q8") != hipSuccess)
        g_flash512_h2s2_q8 = nullptr;
    g_flash_off = getenv("QMATVEC_NO_WMMA") != nullptr || getenv("ATTN_NO_FLASH") != nullptr;  // A/B: attention only
    g_gqa_off = getenv("ATTN_NO_GQA") != nullptr;  // A/B: attn_partial per query head
    return 0;
}

void attention_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    for (int i = 0; i < ATTN_NDIMS; ++i) {
        g_prep[i] = g_partial[i] = g_combine[i] = g_flash[i] = g_partial_q8[i] = g_flash_q8[i] = g_bidir[i] = nullptr;
        g_flash_h2t2[i][0] = g_flash_h2t2[i][1] = g_flash_h1t4[i][0] = g_flash_h1t4[i][1] = nullptr;
        for (int k = 0; k < 4; ++k) g_partial_gqa[i][k] = nullptr;
    }
    g_flash128_g2 = g_flash256_h2t2 = g_flash512_h2s2 = nullptr;
    g_flash128_g2_q8 = g_flash256_h2t2_q8 = g_flash512_h2s2_q8 = nullptr;
}

hipError_t attention_forward(const float* qfull, const float* k, const float* v, const float* q_norm,
                             const float* k_norm, __half* K, __half* V, int kv_shift, long long kv_bstride, int n_q,
                             int n_kv, int n_rot, int pos0, int T,
                             float freq_base, float eps, float* q, float* part_ml, float* part_acc, float* out,
                             int head_dim, int gated, const qmatvec_q8* q8, hipStream_t stream) {
    attention_opts o = {};
    o.head_dim = head_dim;
    o.gated = gated;
    return attention_forward_opts(qfull, k, v, q_norm, k_norm, K, V, kv_shift, kv_bstride, n_q, n_kv, n_rot, pos0, T,
                                  freq_base, eps, q, part_ml, part_acc, out, &o, q8, stream);
}

hipError_t attention_forward_opts(const float* qfull, const float* k, const float* v, const float* q_norm,
                                  const float* k_norm, __half* K, __half* V, int kv_shift, long long kv_bstride,
                                  int n_q, int n_kv, int n_rot, int pos0, int T, float freq_base, float eps, float* q,
                                  float* part_ml, float* part_acc, float* out, const attention_opts* opts,
                                  const qmatvec_q8* q8, hipStream_t stream) {
    if (!opts) return hipErrorInvalidValue;
    const int D = opts->head_dim, hi = D == 128 ? 0 : D == 256 ? 1 : D == 512 ? 2 : D == 64 ? 3 : -1;
    int gated = opts->gated, window = opts->window, v_norm = opts->v_norm, qk_norm = !opts->no_qk_norm;
    int kv_q8 = opts->kv_q8;
    int rot_pairs = opts->rot_pairs > 0 ? opts->rot_pairs : n_rot / 2;
    if (hi < 0) return hipErrorInvalidValue;
    if (!g_prep[hi]) return hipErrorInvalidDeviceFunction;
    // Q8_0 rows: kv_stride halves hold kv_stride * 32 / 17 values (the caller passes the cache's row stride)
    int kv_stride = opts->kv_stride > 0 ? opts->kv_stride : kv_q8 ? n_kv * D * 17 / 32 : n_kv * D;
    const long long row_values = kv_q8 ? (long long)kv_stride * 32 / 17 : kv_stride;
    if (kv_shift < 0 || kv_shift > 30 || n_kv <= 0 || n_q % n_kv != 0 || (q8 && (long long)n_q * D % 256 != 0) ||
        n_rot > D || n_rot % 2 != 0 || rot_pairs > n_rot / 2 || pos0 < 0 || T <= 0 || window < 0 ||
        row_values < (long long)n_kv * D || (kv_q8 && ((long long)kv_stride * 32 % 17 != 0 || row_values % 256 != 0)))
        return hipErrorInvalidValue;
    if (q8 && (long long)T * n_q * D > q8->max_k) return hipErrorInvalidValue;
    const int NB = (n_q * D + 255) / 256;  // attn_combine workgroups per token (256 values each)
    const int n_last = pos0 + T;  // positions the last token attends to

    void* a_prep[] = {(void*)&qfull, (void*)&k, (void*)&v, (void*)&q_norm, (void*)&k_norm, &q, &K, &V, &kv_stride,
                      &kv_shift, &kv_bstride, &n_q, &n_kv, &n_rot, &pos0, &freq_base, &eps, &gated, &rot_pairs,
                      &v_norm, &qk_norm, &kv_q8, (void*)&opts->bq, (void*)&opts->bk, (void*)&opts->bv};
    hipError_t e = hipModuleLaunchKernel(g_prep[hi], (unsigned)(n_q + n_kv), (unsigned)T, 1, (unsigned)D, 1, 1, 0,
                                         stream,
                                         a_prep, nullptr);
    if (e != hipSuccess) return e;

    if (opts->bidir) {  // embedding models: every token of the batch sees the whole batch (or its window)
        if (pos0 != 0 || T > 2048 || kv_q8 || gated || !g_bidir[hi]) return hipErrorInvalidValue;
        const float sc = opts->scale != 0.0f ? opts->scale : 1.0f / sqrtf((float)D);
        int half = window / 2;
        void* a_bidir[] = {&q, &K, &V, &kv_stride, &kv_shift, &kv_bstride, &T, &n_q, &n_kv, (void*)&sc, &half, &out};
        e = hipModuleLaunchKernel(g_bidir[hi], (unsigned)n_q, (unsigned)T, 1, (unsigned)D, 1, 1, 0, stream, a_bidir,
                                  nullptr);
        if (e != hipSuccess || !q8) return e;
        return qmatvec_quantize(out, T * n_q * D, q8, stream);
    }

    // Partial-result slots the caller's buffers hold (attention_part_slots with n_ctx >= n_last)
    const int nc = (n_last + ATTN_CHUNK - 1) / ATTN_CHUNK;
    const long long slots = std::max(opts->part_slots,
                                     std::max<long long>(nc, (long long)T * std::min(nc, ATTN_MAX_SPLITS)));
    float scale = opts->scale != 0.0f ? opts->scale : 1.0f / sqrtf((float)D);
    const float* qc = q;
    qmatvec_q8_out o = qmatvec_q8_outputs(q8);

    // attn_flash: NH query heads x TT tiles of 16 tokens per workgroup, sharing one KV head: 4 x 1 (ratio a
    // multiple of 4), head_dim 128 with ratio 2: 4 heads of two KV heads, 256: 2 x 2, 512: 2 x 1 with the D
    // output dims split over DS = 2 workgroups; other ratios 2 x 2 (even) or 1 x 4 (odd: Qwen2.5-0.5B / 7B's 7).
    const int ratio = n_q / n_kv;
    hipFunction_t flash = nullptr;
    int nh = 4, tt = 1, ds = 1;
    if (D < 512 && ratio % 4 == 0) flash = kv_q8 ? g_flash_q8[hi] : g_flash[hi];
    else if (D == 128 && ratio == 2 && n_q % 4 == 0) flash = kv_q8 ? g_flash128_g2_q8 : g_flash128_g2;
    else if (D == 256 && ratio % 2 == 0) flash = kv_q8 ? g_flash256_h2t2_q8 : g_flash256_h2t2, nh = 2, tt = 2;
    else if (D == 512 && ratio % 2 == 0) flash = kv_q8 ? g_flash512_h2s2_q8 : g_flash512_h2s2, nh = 2, ds = 2;
    else if (ratio % 2 == 0 && g_flash_h2t2[hi][kv_q8]) flash = g_flash_h2t2[hi][kv_q8], nh = 2, tt = 2;
    else if (g_flash_h1t4[hi][kv_q8]) flash = g_flash_h1t4[hi][kv_q8], nh = 1, tt = 4;
    if (T >= kFlashMinT && !g_flash_off && flash) {
        // Ranges of positions per token (each a multiple of 16 positions, at most ATTN_MAX_SPLITS): only as many
        // as it takes to give the GPU ~kFlashWgs workgroups. A prefill batch has enough without splitting, and
        // every split costs partial results written and merged (one split: Gemma-4 prefill +10%, Qwen3.6 +4%);
        // a few tokens at a long context need them
        const long long wgs = (long long)(n_q / nh * ds) * ((T + 16 * tt - 1) / (16 * tt));
        int n_split = 1;
        while (n_split < std::min(nc, ATTN_MAX_SPLITS) && wgs * n_split < kFlashWgs) n_split *= 2;
        n_split = std::min(n_split, std::min(nc, ATTN_MAX_SPLITS));
        int chunk = ((n_last + n_split - 1) / n_split + 15) / 16 * 16;
        n_split = (n_last + chunk - 1) / chunk;
        void* a_flash[] = {(void*)&qc, &K, &V, &kv_stride, &kv_shift, &kv_bstride, &pos0, &T, &n_q, &n_kv, &chunk,
                           &scale, &part_ml, &part_acc, &window};
        e = hipModuleLaunchKernel(flash, (unsigned)(n_q / nh * ds), (unsigned)((T + 16 * tt - 1) / (16 * tt)),
                                  (unsigned)n_split, (unsigned)(32 * nh * tt), 1, 1, 0, stream, a_flash, nullptr);
        if (e != hipSuccess) return e;
        void* a_comb[] = {&part_ml, &part_acc, &n_split, &n_q, (void*)&qfull, &out, &o, &gated};
        return hipModuleLaunchKernel(g_combine[hi], (unsigned)NB, (unsigned)T, 1, 256, 1, 1, 0, stream, a_comb,
                                     nullptr);
    }

    // attn_partial: one chunk per slot, so the tokens run in groups of slots / chunks. Several query heads per KV head:
    // attn_partial_gqa (K / V read once for the group, not ratio times), over 64-position chunks when the slots
    // hold them (a decode step needs the 4x workgroups: Qwen3-30B at 2.4k positions has 4 KV heads x 10 chunks)
    const bool gqa_ok = !kv_q8 && !g_gqa_off && ratio >= 2 && ratio <= 16 && g_partial_gqa[hi][ratio > 8];
    const int nc_s = (n_last + 63) / 64;
    const bool small = gqa_ok && g_partial_gqa[hi][2 + (ratio > 8)] && slots >= nc_s;
    const int pw = small ? 64 : ATTN_CHUNK;
    const hipFunction_t gqa = gqa_ok ? g_partial_gqa[hi][(small ? 2 : 0) + (ratio > 8)] : nullptr;
    const int group = (int)std::max<long long>(1, slots / (small ? nc_s : nc));
    for (int z0 = 0; z0 < T; z0 += group) {
        const int tg = std::min(group, T - z0), p0 = pos0 + z0;
        const int n_chunks = (p0 + tg + pw - 1) / pw;
        const float* qz = qc + (size_t)z0 * n_q * D;
        const float* qfz = qfull + (size_t)z0 * (gated ? 2 : 1) * n_q * D;
        float* outz = out + (size_t)z0 * n_q * D;
        qmatvec_q8_out oz = o;  // quantized blocks of tokens z0 ..: NB 256-value blocks per token
        if (oz.xh) oz.xh += (size_t)z0 * NB * 256;  // (the 16-bit copy: with or without the int8 forms)
        if (oz.qK) {
            const size_t blk = (size_t)z0 * NB;
            oz.q0 += blk * 256;
            oz.qK += blk * 256;
            oz.d0 += blk * 8;
            oz.dK += blk;
            oz.bsK += blk * 16;
            oz.s0 += blk * 8;
        }
        void* a_part[] = {(void*)&qz, &K, &V, &kv_stride, &kv_shift, &kv_bstride, (void*)&p0, &n_q, &n_kv, &scale,
                          &part_ml, &part_acc, &window};
        if (gqa)
            e = hipModuleLaunchKernel(gqa, (unsigned)n_kv, (unsigned)n_chunks, (unsigned)tg, ATTN_CHUNK, 1, 1, 0,
                                      stream, a_part, nullptr);
        else
            e = hipModuleLaunchKernel(kv_q8 ? g_partial_q8[hi] : g_partial[hi], (unsigned)n_q, (unsigned)n_chunks,
                                      (unsigned)tg, ATTN_CHUNK, 1, 1, 0, stream, a_part, nullptr);
        if (e != hipSuccess) return e;
        int nch = n_chunks;
        void* a_comb[] = {&part_ml, &part_acc, &nch, &n_q, (void*)&qfz, &outz, &oz, &gated};
        e = hipModuleLaunchKernel(g_combine[hi], (unsigned)NB, (unsigned)tg, 1, 256, 1, 1, 0, stream, a_comb,
                                  nullptr);
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}
