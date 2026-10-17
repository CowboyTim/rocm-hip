// qmatvec.cpp — see qmatvec.h.

#include "qmatvec.h"
#include "quant_types.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define QMV_ROWS_PER_BLOCK 4  // must match ROWS_PER_BLOCK in matvec_q.hip
#define QMV_MAX_TYPE 128

static hipModule_t g_mod = nullptr;
static hipFunction_t g_fn[QMV_MAX_TYPE];
static hipFunction_t g_fn_q8[QMV_MAX_TYPE];  // int8 kernels, matvec_q8_<type>
static hipFunction_t g_quant = nullptr;

int qmatvec_init(const char *co_path) {
    if (g_mod) return 0;
    if (hipModuleLoad(&g_mod, co_path) != hipSuccess) {
        fprintf(stderr, "[qmatvec] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    memset(g_fn, 0, sizeof(g_fn));
    memset(g_fn_q8, 0, sizeof(g_fn_q8));
    for (int t = 0; t < QMV_MAX_TYPE; ++t) {
        const qt_traits *tr = qt_get_traits(t);
        if (!tr) continue;
        char name[96];
        snprintf(name, sizeof(name), "matvec_%s", tr->name);
        if (hipModuleGetFunction(&g_fn[t], g_mod, name) != hipSuccess) g_fn[t] = nullptr;
        snprintf(name, sizeof(name), "matvec_q8_%s", tr->name);
        if (hipModuleGetFunction(&g_fn_q8[t], g_mod, name) != hipSuccess) g_fn_q8[t] = nullptr;
    }
    if (hipModuleGetFunction(&g_quant, g_mod, "quantize_q8") != hipSuccess) g_quant = nullptr;
    return 0;
}

void qmatvec_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    memset(g_fn, 0, sizeof(g_fn));
    memset(g_fn_q8, 0, sizeof(g_fn_q8));
    g_quant = nullptr;
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
    const unsigned grid = (unsigned)((N + QMV_ROWS_PER_BLOCK - 1) / QMV_ROWS_PER_BLOCK);
    return hipModuleLaunchKernel(g_fn[type], grid, 1, 1, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0, stream, args, nullptr);
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

int qmatvec_q8_supported(int type, int K) {
    if (type < 0 || type >= QMV_MAX_TYPE || !g_fn_q8[type] || !g_quant || K % 256 != 0) return 0;
    return type == QT_Q4_0_ROCMFP4 ? K % 64 == 0 : 1;  // ROCmFP4 lanes take 36-byte block pairs
}

hipError_t qmatvec_quantize(const float *x, int K, const qmatvec_q8 *q, hipStream_t stream) {
    if (!g_quant) return hipErrorInvalidDeviceFunction;
    if (K <= 0 || K % 256 != 0 || K > q->max_k) return hipErrorInvalidValue;
    qmatvec_q8_out o = qmatvec_q8_outputs(q);
    void *args[] = {(void *)&x, &o};
    return hipModuleLaunchKernel(g_quant, (unsigned)(K / 256), 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}

hipError_t qmatvec_launch_q8_pair(int type, const void *W, int N, float *y, const void *W2, int N2, float *y2,
                                  size_t row_bytes, const qmatvec_q8 *q, int K, hipStream_t stream) {
    if (!qmatvec_q8_supported(type, K)) return hipErrorInvalidDeviceFunction;
    if (row_bytes != qt_row_size(type, K) || row_bytes % 4 != 0 || ((uintptr_t)W & 3) != 0 || ((uintptr_t)W2 & 3) != 0 ||
        K > q->max_k)
        return hipErrorInvalidValue;
    const bool kq = type != QT_Q4_0_ROCMFP4;  // K-quants read Q8_K, ROCmFP4 reads Q8_0
    const int8_t *q8 = kq ? q->qK : q->q0;
    const float *d8 = kq ? q->dK : q->d0;
    const int32_t *bs = q->bsK;
    long long rb = (long long)row_bytes;
    int N1 = N, Nt = N + N2;
    void *args[] = {(void *)&W, (void *)&W2, &N1, &rb, (void *)&q8, (void *)&d8, (void *)&bs, (void *)&y, (void *)&y2,
                    &Nt, &K};
    const unsigned grid = (unsigned)((Nt + QMV_ROWS_PER_BLOCK - 1) / QMV_ROWS_PER_BLOCK);
    return hipModuleLaunchKernel(g_fn_q8[type], grid, 1, 1, 32 * QMV_ROWS_PER_BLOCK, 1, 1, 0, stream, args, nullptr);
}

hipError_t qmatvec_launch_q8(int type, const void *W, size_t row_bytes, const qmatvec_q8 *q, float *y, int N, int K,
                             hipStream_t stream) {
    return qmatvec_launch_q8_pair(type, W, N, y, W, 0, y, row_bytes, q, K, stream);
}
