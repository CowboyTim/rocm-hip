// qmatvec.cpp — see qmatvec.h.

#include "inference_log.h"
#include "kernel_loader.h"
#include "qmatvec.h"
#include "quant_types.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QMV_ROWS_PER_BLOCK 4  // must match ROWS_PER_BLOCK in matvec_q.hip
#define QMV_F32_PARTS 4       // must match F32_PARTS in matvec_q.hip
#define QMV_MAX_TYPE 128

static hipModule_t g_mod = nullptr;
static hipFunction_t g_fn[QMV_MAX_TYPE];
static hipFunction_t g_fn_q8[QMV_MAX_TYPE];  // int8 kernels, matvec_q8_<type>
static hipFunction_t g_quant = nullptr;
static hipFunction_t g_mm_f32 = nullptr;  // matmul_f32: batched F32
static hipFunction_t g_quant_rows_q8_0 = nullptr, g_mv_q8_0s = nullptr;  // int8 copy of a float LM head
static hipFunction_t g_fn_mm[QMV_MAX_TYPE];  // batched int8 kernels, matmul_q8_<type>
static hipFunction_t g_fn_q8_id[QMV_MAX_TYPE];  // expert-indexed int8 kernels, matvec_q8_id_<type>
static hipFunction_t g_fn_id[QMV_MAX_TYPE];     // expert-indexed float kernels, matvec_id_<type>
static hipFunction_t g_fn_mm_id[QMV_MAX_TYPE];  // expert-tiled batched int8 kernels, matmul_q8_id_<type>
static hipFunction_t g_fn_wmma[QMV_MAX_TYPE];   // WMMA tiled int8 kernels, matmul_wmma_<type> (Phase 20 W2)
static hipFunction_t g_fn_wmma_id[QMV_MAX_TYPE];  // WMMA on MoE expert tiles, matmul_wmma_id_<type> (W4)
static hipFunction_t g_fn_mmf[QMV_MAX_TYPE];     // batched float kernels, matmul_f_<type>
static hipFunction_t g_lr_fast = nullptr;         // matvec_q8_lr_q4_0_rocmfp4_fast
static hipFunction_t g_fn_q8t[QMV_MAX_TYPE];     // int8 kernels for 2..4 tokens, matvec_q8t_<type>
static bool g_use_wmma = true;
static int g_wmma_min_t = 32;  // smaller batches, and N < 64 (a 64-row tile mostly padding), stay on matmul_q8_<type>
#define QMV_MM_TB 16      // tokens per tile, must match MM_TB in matvec_q.hip
#define QMV_WM_ROWS 128       // weight rows per workgroup of matmul_wmma[_id]_<rocmfp4 type>, WM_ROWS in matvec_q.hip
#define QMV_WM_THREADS 256    // its block size
#define QMV_WMMA_ID_TB 64  // entries per expert tile of matmul_wmma_id_<type> (4 sub-tiles of 16)

// The int8 kernels of the K-quants read the activations' Q8_K form; ROCmFP4, ROCmFP4_FAST and Q8_0 weights the
// Q8_0 form
static bool reads_q8k(int type) { return type == QT_Q4_K || type == QT_Q5_K || type == QT_Q6_K; }

int qmatvec_init(const char *co_path) {
    if (g_mod) return 0;
    if (kernel_module_load(&g_mod, co_path) != hipSuccess) {
        INFER_LOG("[qmatvec] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    memset(g_fn, 0, sizeof(g_fn));
    memset(g_fn_q8, 0, sizeof(g_fn_q8));
    memset(g_fn_mm, 0, sizeof(g_fn_mm));
    memset(g_fn_q8_id, 0, sizeof(g_fn_q8_id));
    memset(g_fn_id, 0, sizeof(g_fn_id));
    memset(g_fn_mm_id, 0, sizeof(g_fn_mm_id));
    memset(g_fn_wmma, 0, sizeof(g_fn_wmma));
    memset(g_fn_wmma_id, 0, sizeof(g_fn_wmma_id));
    memset(g_fn_mmf, 0, sizeof(g_fn_mmf));
    memset(g_fn_q8t, 0, sizeof(g_fn_q8t));
    for (int t = 0; t < QMV_MAX_TYPE; ++t) {
        const qt_traits *tr = qt_get_traits(t);
        if (!tr) continue;
        char name[96];
        snprintf(name, sizeof(name), "matvec_%s", tr->name);
        if (hipModuleGetFunction(&g_fn[t], g_mod, name) != hipSuccess) g_fn[t] = nullptr;
        snprintf(name, sizeof(name), "matvec_q8_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_q8[t], g_mod, name) != hipSuccess) g_fn_q8[t] = nullptr;
        snprintf(name, sizeof(name), "matmul_q8_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_mm[t], g_mod, name) != hipSuccess) g_fn_mm[t] = nullptr;
        snprintf(name, sizeof(name), "matvec_q8_id_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_q8_id[t], g_mod, name) != hipSuccess) g_fn_q8_id[t] = nullptr;
        snprintf(name, sizeof(name), "matvec_id_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_id[t], g_mod, name) != hipSuccess) g_fn_id[t] = nullptr;
        snprintf(name, sizeof(name), "matmul_q8_id_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_mm_id[t], g_mod, name) != hipSuccess) g_fn_mm_id[t] = nullptr;
        snprintf(name, sizeof(name), "matmul_wmma_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_wmma[t], g_mod, name) != hipSuccess) g_fn_wmma[t] = nullptr;
        snprintf(name, sizeof(name), "matmul_wmma_id_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_wmma_id[t], g_mod, name) != hipSuccess) g_fn_wmma_id[t] = nullptr;
        snprintf(name, sizeof(name), "matvec_q8t_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_q8t[t], g_mod, name) != hipSuccess) g_fn_q8t[t] = nullptr;
        snprintf(name, sizeof(name), "matmul_f_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_mmf[t], g_mod, name) != hipSuccess) g_fn_mmf[t] = nullptr;
    }
    if (hipModuleGetFunction(&g_quant, g_mod, "quantize_q8") != hipSuccess) g_quant = nullptr;
    if (hipModuleGetFunction(&g_mm_f32, g_mod, "matmul_f32") != hipSuccess) g_mm_f32 = nullptr;
    if (hipModuleGetFunction(&g_lr_fast, g_mod, "matvec_q8_lr_q4_0_rocmfp4_fast") != hipSuccess) g_lr_fast = nullptr;
    if (hipModuleGetFunction(&g_quant_rows_q8_0, g_mod, "quantize_rows_q8_0") != hipSuccess) g_quant_rows_q8_0 = nullptr;
    if (hipModuleGetFunction(&g_mv_q8_0s, g_mod, "matvec_q8_0s") != hipSuccess) g_mv_q8_0s = nullptr;
    g_use_wmma = getenv("QMATVEC_NO_WMMA") == nullptr && getenv("QMATVEC_NO_WMMA_MM") == nullptr;  // A/B: all / matmuls only
    if (const char *v = getenv("QMATVEC_WMMA_MIN_T")) g_wmma_min_t = atoi(v);  // tests: route small batches to WMMA
    return 0;
}

void qmatvec_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    memset(g_fn, 0, sizeof(g_fn));
    memset(g_fn_q8, 0, sizeof(g_fn_q8));
    memset(g_fn_mm, 0, sizeof(g_fn_mm));
    memset(g_fn_q8_id, 0, sizeof(g_fn_q8_id));
    memset(g_fn_id, 0, sizeof(g_fn_id));
    memset(g_fn_mm_id, 0, sizeof(g_fn_mm_id));
    memset(g_fn_wmma, 0, sizeof(g_fn_wmma));
    memset(g_fn_wmma_id, 0, sizeof(g_fn_wmma_id));
    memset(g_fn_mmf, 0, sizeof(g_fn_mmf));
    memset(g_fn_q8t, 0, sizeof(g_fn_q8t));
    g_quant = nullptr;
    g_mm_f32 = g_quant_rows_q8_0 = g_mv_q8_0s = g_lr_fast = nullptr;
}

int qmatvec_supported(int type) {
    return type >= 0 && type < QMV_MAX_TYPE && g_fn[type] != nullptr;
}

hipError_t qmatvec_launch(int type, const void *W, size_t row_bytes, const float *x, float *y, int N, int K,
                          hipStream_t stream) {
    if (!qmatvec_supported(type)) return hipErrorInvalidDeviceFunction;
    const qt_traits *tr = qt_get_traits(type);
    if (K % tr->blck_size != 0 || row_bytes != qt_row_size(type, K)) return hipErrorInvalidValue;
    long long rb = (long long)row_bytes;
    void *args[] = {(void *)&W, &rb, (void *)&x, (void *)&y, &N, &K};
    if (type == QT_F32)  // one row per workgroup (matvec_f32)
        return hipModuleLaunchKernel(g_fn[type], (unsigned)N, 1, 1, 32 * QMV_F32_PARTS, 1, 1, 0, stream, args, nullptr);
    const unsigned grid = (unsigned)((N + QMV_ROWS_PER_BLOCK - 1) / QMV_ROWS_PER_BLOCK);
    return hipModuleLaunchKernel(g_fn[type], grid, 1, 1, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0, stream, args, nullptr);
}

#define QMV_MF32_TT 8     // must match MF32_TT in matvec_q.hip
#define QMV_MF32_ROWS 16  // must match MF32_ROWS in matvec_q.hip
#define QMV_MMF_TT 8   // must match MMF_TT in matvec_q.hip

hipError_t qmatvec_launch_batch(int type, const void *W, size_t row_bytes, const float *x, float *y, int N, int K,
                                int T, hipStream_t stream) {
    if (T > 1 && type == QT_F32 && g_mm_f32 && K % (4 * QMV_F32_PARTS) == 0 && row_bytes == (size_t)K * 4 &&
        (((uintptr_t)W | (uintptr_t)x) & 15) == 0) {
        long long rb = (long long)row_bytes;
        void *args[] = {(void *)&W, &rb, (void *)&x, (void *)&y, &N, &K, &T};
        return hipModuleLaunchKernel(g_mm_f32, (unsigned)((N + QMV_MF32_ROWS - 1) / QMV_MF32_ROWS),
                                     (unsigned)((T + QMV_MF32_TT - 1) / QMV_MF32_TT), 1,
                                     32 * QMV_F32_PARTS, 1, 1, 0, stream, args, nullptr);
    }
    if (T > 1 && type >= 0 && type < QMV_MAX_TYPE && g_fn_mmf[type]) {  // same sums as T launches of matvec_<type>
        const qt_traits *tr = qt_get_traits(type);
        if (K % tr->blck_size != 0 || row_bytes != qt_row_size(type, K)) return hipErrorInvalidValue;
        long long rb = (long long)row_bytes;
        void *args[] = {(void *)&W, &rb, (void *)&x, (void *)&y, &N, &K, &T};
        return hipModuleLaunchKernel(g_fn_mmf[type], (unsigned)((N + QMV_ROWS_PER_BLOCK - 1) / QMV_ROWS_PER_BLOCK),
                                     (unsigned)((T + QMV_MMF_TT - 1) / QMV_MMF_TT), 1, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0,
                                     stream, args, nullptr);
    }
    hipError_t e = hipSuccess;
    for (int t = 0; t < T && e == hipSuccess; ++t)
        e = qmatvec_launch(type, W, row_bytes, x + (size_t)t * K, y + (size_t)t * N, N, K, stream);
    return e;
}

int qmatvec_q8_0s_source_supported(int type, int K) {
    return g_quant_rows_q8_0 && g_mv_q8_0s && K % 256 == 0 && (type == QT_F32 || type == QT_F16 || type == QT_BF16);
}

hipError_t qmatvec_quantize_rows_q8_0(int type, const void *W, size_t row_bytes, int N, int K, int8_t *qs, uint16_t *d,
                                      hipStream_t stream) {
    if (!qmatvec_q8_0s_source_supported(type, K)) return hipErrorInvalidDeviceFunction;
    if (row_bytes != qt_row_size(type, K) || ((uintptr_t)qs & 15) != 0) return hipErrorInvalidValue;
    long long rb = (long long)row_bytes;
    void *args[] = {(void *)&W, &rb, &type, &K, &qs, &d};
    return hipModuleLaunchKernel(g_quant_rows_q8_0, (unsigned)N, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t qmatvec_launch_q8_0s(const int8_t *qs, const uint16_t *d, const qmatvec_q8 *q, float *y, int N, int K,
                                hipStream_t stream) {
    if (!g_mv_q8_0s) return hipErrorInvalidDeviceFunction;
    if (K % 256 != 0 || K > q->max_k || ((uintptr_t)qs & 15) != 0) return hipErrorInvalidValue;
    const int8_t *q8 = q->q0;
    const float *d8 = q->d0;
    void *args[] = {(void *)&qs, (void *)&d, (void *)&q8, (void *)&d8, &y, &N, &K};
    return hipModuleLaunchKernel(g_mv_q8_0s, (unsigned)((N + QMV_ROWS_PER_BLOCK - 1) / QMV_ROWS_PER_BLOCK), 1, 1,
                                 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0, stream, args, nullptr);
}

// ---------------------------------------------------------------- int8 path

static size_t align16(size_t n) { return (n + 15) & ~(size_t)15; }

size_t qmatvec_q8_bytes(int max_k) {
    const size_t k = (size_t)max_k;
    return 2 * align16(k) + align16(k / 32 * 4) + align16(k / 256 * 4) + align16(k / 16 * 4);
}

void qmatvec_q8_bind(qmatvec_q8 *q, void *mem, int max_k) {
    const size_t k = (size_t)max_k;
    char *p = (char *)mem;
    q->q0 = (int8_t *)p;
    p += align16(k);
    q->qK = (int8_t *)p;
    p += align16(k);
    q->d0 = (float *)p;
    p += align16(k / 32 * 4);
    q->dK = (float *)p;
    p += align16(k / 256 * 4);
    q->bsK = (int32_t *)p;
    q->max_k = max_k;
}

// ROCmFP4 weights read only the Q8_0 side of the quantized input (32-value blocks), so their rows need not be
// whole 256-value Q8_K blocks: e.g. the qwen4exp experts' K = 640. K-quants read Q8_K and need K % 256 == 0.
int qmatvec_q8_supported(int type, int K) {
    if (type < 0 || type >= QMV_MAX_TYPE || !g_fn_q8[type] || !g_quant) return 0;
    if (type == QT_Q4_0_ROCMFP4 || type == QT_Q8_0) return K % 64 == 0;  // lanes take block pairs (36 / 68 bytes)
    if (type == QT_Q4_0_ROCMFP4_FAST) return K % 128 == 0;  // lanes take 68-byte block quads
    return K % 256 == 0;
}

hipError_t qmatvec_quantize(const float *x, int K, const qmatvec_q8 *q, hipStream_t stream) {
    if (!g_quant) return hipErrorInvalidDeviceFunction;
    if (K <= 0 || K % 32 != 0 || K > q->max_k) return hipErrorInvalidValue;
    qmatvec_q8_out o = qmatvec_q8_outputs(q);
    void *args[] = {(void *)&x, &K, &o};
    return hipModuleLaunchKernel(g_quant, (unsigned)((K + 255) / 256), 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

// Lanes per row for the int8 matvec kernels: the smallest power of two (>= 4, >= 8 for K-quants, which use 8
// lanes per 256-value block) that still gives every lane a task, so short rows (K = 512 MoE experts) don't
// leave most of the wave idle. See Q8_ROW_INDEX in matvec_q.hip.
static int q8_lanes_per_row(int type, int K, long long rows) {
    const bool kq = reads_q8k(type);
    const int tasks = type == QT_Q4_0_ROCMFP4 || type == QT_Q8_0 ? K / 64
                      : type == QT_Q4_0_ROCMFP4_FAST          ? K / 128
                                                              : K / 256 * 8;
    int L = 32;
    while (L > (kq ? 8 : 4) && L / 2 >= tasks) L /= 2;
    // ROCmFP4_FAST: at most 8 lanes per row, i.e. several 68-byte quads per lane. Measured on the 35B: z
    // (4096 x 2048) 168 -> 184 GB/s, ssm_out (2048 x 4096) 160 -> 190; plain ROCmFP4 is best at 32. Matrices
    // of few rows (all rows of the launch) need more lanes to keep enough loads in flight: the qwen4exp mixers'
    // down + inject (324 x 10240) ran at 118 GB/s with 8.
    const int cap = rows >= 4096 ? 8 : rows >= 1024 ? 16 : 32;
    if (type == QT_Q4_0_ROCMFP4_FAST && L > cap) L = cap;
    return L;
}

hipError_t qmatvec_launch_q8_pair(int type, const void *W, int N, float *y, const void *W2, int N2, float *y2,
                                  size_t row_bytes, const qmatvec_q8 *q, int K, hipStream_t stream) {
    const void *Ws[] = {W, W2};
    const int Ns[] = {N, N2};
    float *ys[] = {y, y2};
    return qmatvec_launch_q8_multi(type, 2, Ws, Ns, ys, row_bytes, q, K, stream);
}

int qmatvec_q8_lr_supported(int type, int K) {
    return g_lr_fast && type == QT_Q4_0_ROCMFP4_FAST && K % 128 == 0 && K <= 1024;  // LR_MAXK in matvec_q.hip
}

hipError_t qmatvec_launch_q8_lr(int type, const void *W, size_t row_bytes, const float *lo, int n, int K, float scale,
                                float *y, int N, hipStream_t stream) {
    if (!qmatvec_q8_lr_supported(type, K)) return hipErrorInvalidDeviceFunction;
    if (n > K || row_bytes != qt_row_size(type, K) || row_bytes % 4 != 0 || ((uintptr_t)W & 3) != 0)
        return hipErrorInvalidValue;
    long long rb = (long long)row_bytes;
    int L = q8_lanes_per_row(type, K, N);
    void *args[] = {(void *)&W, &rb, (void *)&lo, &n, &K, &scale, &y, &N, &L};
    const int rows = QMV_ROWS_PER_BLOCK * 32 / L;
    return hipModuleLaunchKernel(g_lr_fast, (unsigned)((N + rows - 1) / rows), 1, 1, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0,
                                 stream, args, nullptr);
}

hipError_t qmatvec_launch_q8_multi(int type, int n, const void *const *Ws, const int *Ns, float *const *ys,
                                   size_t row_bytes, const qmatvec_q8 *q, int K, hipStream_t stream) {
    return qmatvec_launch_q8_multi_t(type, n, Ws, Ns, ys, row_bytes, q, K, 1, stream);
}

int qmatvec_q8_few_supported(int type, int K, int T) {
    return T == 1 ? qmatvec_q8_supported(type, K)
                  : T >= 2 && T <= 4 && type >= 0 && type < QMV_MAX_TYPE && g_fn_q8t[type] &&
                        qmatvec_q8_supported(type, K);
}

hipError_t qmatvec_launch_q8_multi_t(int type, int n, const void *const *Ws, const int *Ns, float *const *ys,
                                     size_t row_bytes, const qmatvec_q8 *q, int K, int T, hipStream_t stream) {
    if (!qmatvec_q8_few_supported(type, K, T)) return hipErrorInvalidDeviceFunction;
    if (n < 1 || n > 4 || row_bytes != qt_row_size(type, K) || row_bytes % 4 != 0 || (long long)K * T > q->max_k)
        return hipErrorInvalidValue;
    const void *W[4];
    float *y[4];
    int bound[4], Nt = 0;
    for (int i = 0; i < 4; ++i) {
        const int j = i < n ? i : n - 1;  // unused slots repeat the last matrix, with no rows
        if (((uintptr_t)Ws[j] & 3) != 0) return hipErrorInvalidValue;
        W[i] = Ws[j];
        y[i] = ys[j];
        if (i < n) Nt += Ns[i];
        bound[i] = Nt;
    }
    const bool kq = reads_q8k(type);  // K-quants read Q8_K, ROCmFP4 / Q8_0 weights Q8_0
    const int8_t *q8 = kq ? q->qK : q->q0;
    const float *d8 = kq ? q->dK : q->d0;
    const int32_t *bs = q->bsK;
    long long rb = (long long)row_bytes;
    int L = q8_lanes_per_row(type, K, Nt);
    void *args[] = {(void *)&W[0], (void *)&W[1], (void *)&W[2], (void *)&W[3], &bound[0], &bound[1], &bound[2], &rb,
                    (void *)&q8, (void *)&d8, (void *)&bs, (void *)&y[0], (void *)&y[1], (void *)&y[2], (void *)&y[3],
                    &Nt, &K, &L, &T};
    const int rows = QMV_ROWS_PER_BLOCK * 32 / L;
    const unsigned grid = (unsigned)((Nt + rows - 1) / rows);
    return hipModuleLaunchKernel(T == 1 ? g_fn_q8[type] : g_fn_q8t[type], grid, 1, 1, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0,
                                 stream, args, nullptr);
}

hipError_t qmatvec_launch_q8(int type, const void *W, size_t row_bytes, const qmatvec_q8 *q, float *y, int N, int K,
                             hipStream_t stream) {
    return qmatvec_launch_q8_pair(type, W, N, y, W, 0, y, row_bytes, q, K, stream);
}

hipError_t qmatvec_launch_q8_batch(int type, const void *W, size_t row_bytes, const qmatvec_q8 *q, float *y,
                                   long long ldy, int N, int K, int T, hipStream_t stream) {
    if (!qmatvec_q8_supported(type, K) || !g_fn_mm[type]) return hipErrorInvalidDeviceFunction;
    if (T <= 0 || row_bytes != qt_row_size(type, K) || row_bytes % 4 != 0 || ((uintptr_t)W & 3) != 0 ||
        (long long)T * K > q->max_k || ldy < N)
        return hipErrorInvalidValue;
    if (T > 1 && ldy == N && qmatvec_q8_few_supported(type, K, T)) {  // a few tokens: weights decoded once
        const void *Ws[] = {W};
        const int Ns[] = {N};
        float *ys[] = {y};
        return qmatvec_launch_q8_multi_t(type, 1, Ws, Ns, ys, row_bytes, q, K, T, stream);
    }
    const bool kq = reads_q8k(type);
    const int8_t *q8 = kq ? q->qK : q->q0;
    const float *d8 = kq ? q->dK : q->d0;
    const int32_t *bs = q->bsK;
    long long rb = (long long)row_bytes;
    if (g_use_wmma && g_fn_wmma[type] && T >= g_wmma_min_t && N >= 64 && K % 128 == 0) {
        // tiles of 64 tokens x 128 rows (ROCmFP4 types, 256 threads) or 64 rows (K-quants, 128 threads); the
        // K-quant kernels also take the Q8_K group sums (for the mins)
        void *wargs_q80[] = {(void *)&W, &rb, (void *)&q8, (void *)&d8, (void *)&y, &ldy, &N, &K, &T};
        void *wargs_q8k[] = {(void *)&W, &rb, (void *)&q8, (void *)&d8, (void *)&bs, (void *)&y, &ldy, &N, &K, &T};
        void **wargs = kq ? wargs_q8k : wargs_q80;
        const int rows = kq ? 64 : QMV_WM_ROWS, threads = kq ? 128 : QMV_WM_THREADS;
        return hipModuleLaunchKernel(g_fn_wmma[type], (unsigned)((N + rows - 1) / rows), (unsigned)((T + 63) / 64), 1,
                                     (unsigned)threads, 1, 1, 0, stream, wargs, nullptr);
    }
    void *args[] = {(void *)&W, &rb, (void *)&q8, (void *)&d8, (void *)&bs, (void *)&y, &ldy, &N, &K, &T};
    const unsigned gx = (unsigned)((N + QMV_ROWS_PER_BLOCK - 1) / QMV_ROWS_PER_BLOCK);
    const unsigned gy = (unsigned)((T + QMV_MM_TB - 1) / QMV_MM_TB);
    return hipModuleLaunchKernel(g_fn_mm[type], gx, gy, 1, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0, stream, args, nullptr);
}

hipError_t qmatvec_launch_q8_id(int type, const void *W, size_t row_bytes, const int32_t *ids, int n_used,
                                int x_per_slot, const qmatvec_q8 *q, float *y, int N, int K, int T, hipStream_t stream) {
    return qmatvec_launch_q8_id_x(type, W, nullptr, nullptr, nullptr, row_bytes, ids, n_used, x_per_slot, q, y,
                                  nullptr, N, K, T, stream);
}

hipError_t qmatvec_launch_q8_id_pair(int type, const void *W, const void *W2, size_t row_bytes, const int32_t *ids,
                                     int n_used, int x_per_slot, const qmatvec_q8 *q, float *y, float *y2, int N, int K,
                                     int T, hipStream_t stream) {
    return qmatvec_launch_q8_id_x(type, W, W2, nullptr, nullptr, row_bytes, ids, n_used, x_per_slot, q, y, y2, N, K,
                                  T, stream);
}

hipError_t qmatvec_launch_q8_id_x(int type, const void *W, const void *W2, const void *Wx, const void *W2x,
                                  size_t row_bytes, const int32_t *ids, int n_used, int x_per_slot,
                                  const qmatvec_q8 *q, float *y, float *y2, int N, int K, int T, hipStream_t stream) {
    if (!qmatvec_q8_supported(type, K) || !g_fn_q8_id[type]) return hipErrorInvalidDeviceFunction;
    const long long in_rows = (long long)T * (x_per_slot ? n_used : 1);
    if (T <= 0 || n_used <= 0 || row_bytes != qt_row_size(type, K) || row_bytes % 4 != 0 || ((uintptr_t)W & 3) != 0 ||
        in_rows * K > q->max_k)
        return hipErrorInvalidValue;
    const bool kq = reads_q8k(type);
    const int8_t *q8 = kq ? q->qK : q->q0;
    const float *d8 = kq ? q->dK : q->d0;
    const int32_t *bs = q->bsK;
    long long rb = (long long)row_bytes;
    int L = q8_lanes_per_row(type, K, (long long)N * n_used * (W2 ? 2 : 1) * T);
    if ((((uintptr_t)W2 | (uintptr_t)Wx | (uintptr_t)W2x) & 3) != 0) return hipErrorInvalidValue;
    void *args[] = {(void *)&W, (void *)&W2, &rb, (void *)&ids, &n_used, &x_per_slot, (void *)&q8, (void *)&d8,
                    (void *)&bs, (void *)&y, (void *)&y2, &N, &K, &L, (void *)&Wx, (void *)&W2x};
    const int rows = QMV_ROWS_PER_BLOCK * 32 / L;
    const unsigned gx = (unsigned)((N + rows - 1) / rows);
    return hipModuleLaunchKernel(g_fn_q8_id[type], gx, (unsigned)(n_used * (W2 ? 2 : 1)), (unsigned)T,
                                 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0, stream, args, nullptr);
}

hipError_t qmatvec_launch_id(int type, const void *W, size_t row_bytes, const int32_t *ids, int n_used, int x_per_slot,
                             const float *x, float *y, int N, int K, int T, hipStream_t stream) {
    if (type < 0 || type >= QMV_MAX_TYPE || !g_fn_id[type]) return hipErrorInvalidDeviceFunction;
    const qt_traits *tr = qt_get_traits(type);
    if (T <= 0 || n_used <= 0 || K % tr->blck_size != 0 || row_bytes != qt_row_size(type, K)) return hipErrorInvalidValue;
    long long rb = (long long)row_bytes;
    void *args[] = {(void *)&W, &rb, (void *)&ids, &n_used, &x_per_slot, (void *)&x, (void *)&y, &N, &K};
    const unsigned gx = (unsigned)((N + QMV_ROWS_PER_BLOCK - 1) / QMV_ROWS_PER_BLOCK);
    return hipModuleLaunchKernel(g_fn_id[type], gx, (unsigned)n_used, (unsigned)T, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0,
                                 stream, args, nullptr);
}

static bool wmma_id_ok(int type, int N, int K) {
    return g_use_wmma && g_fn_wmma_id[type] && N >= 64 && K % 128 == 0;
}

int qmatvec_id_batch_tile(int type, int N, int K) {
    if (type < 0 || type >= QMV_MAX_TYPE) return QMV_MM_TB;
    return wmma_id_ok(type, N, K) ? QMV_WMMA_ID_TB : QMV_MM_TB;
}

int qmatvec_q8_id_batch_supported(int type, int K) {
    return qmatvec_q8_supported(type, K) && type >= 0 && type < QMV_MAX_TYPE && g_fn_mm_id[type] != nullptr;
}

hipError_t qmatvec_launch_q8_id_batch(int type, const void *W, size_t row_bytes, const qmatvec_q8 *q, float *y, int N,
                                      int K, int max_tiles, int tile, const int32_t *n_tiles, const int32_t *tile_e,
                                      const int32_t *tile_start, const int32_t *tile_n, const int32_t *in_rows,
                                      const int32_t *out_rows, hipStream_t stream) {
    if (!qmatvec_q8_id_batch_supported(type, K)) return hipErrorInvalidDeviceFunction;
    if (max_tiles <= 0 || tile <= 0 || tile > qmatvec_id_batch_tile(type, N, K) || row_bytes != qt_row_size(type, K) ||
        row_bytes % 4 != 0 || ((uintptr_t)W & 3) != 0)
        return hipErrorInvalidValue;
    const bool kq = reads_q8k(type);
    const int8_t *q8 = kq ? q->qK : q->q0;
    const float *d8 = kq ? q->dK : q->d0;
    const int32_t *bs = q->bsK;
    long long rb = (long long)row_bytes;
    void *args[] = {(void *)&W, &rb, (void *)&q8, (void *)&d8, (void *)&bs, (void *)&y, &N, &K, (void *)&n_tiles,
                    (void *)&tile_e, (void *)&tile_start, (void *)&tile_n, (void *)&in_rows, (void *)&out_rows};
    if (wmma_id_ok(type, N, K)) {  // up to 64 entries x 128 expert rows (ROCmFP4 / Q8_0) or 64 (K-quants)
        const int rows = kq ? 64 : QMV_WM_ROWS, threads = kq ? 128 : QMV_WM_THREADS;
        return hipModuleLaunchKernel(g_fn_wmma_id[type], (unsigned)((N + rows - 1) / rows), (unsigned)max_tiles, 1,
                                     (unsigned)threads, 1, 1, 0, stream, args, nullptr);
    }
    const unsigned gx = (unsigned)((N + QMV_ROWS_PER_BLOCK - 1) / QMV_ROWS_PER_BLOCK);
    return hipModuleLaunchKernel(g_fn_mm_id[type], gx, (unsigned)max_tiles, 1, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0, stream,
                                 args, nullptr);
}
