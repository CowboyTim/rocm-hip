// qmatvec.cpp — see qmatvec.h.

#include "qmatvec.h"
#include "quant_types.h"

#include <stdio.h>
#include <string.h>

#define QMV_ROWS_PER_BLOCK 4  // must match ROWS_PER_BLOCK in matvec_q.hip
#define QMV_MAX_TYPE 128

static hipModule_t g_mod = nullptr;
static hipFunction_t g_fn[QMV_MAX_TYPE];

int qmatvec_init(const char *co_path) {
    if (g_mod) return 0;
    if (hipModuleLoad(&g_mod, co_path) != hipSuccess) {
        fprintf(stderr, "[qmatvec] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    memset(g_fn, 0, sizeof(g_fn));
    for (int t = 0; t < QMV_MAX_TYPE; ++t) {
        const qt_traits *tr = qt_get_traits(t);
        if (!tr) continue;
        char name[96];
        snprintf(name, sizeof(name), "matvec_%s", tr->name);
        if (hipModuleGetFunction(&g_fn[t], g_mod, name) != hipSuccess) g_fn[t] = nullptr;
    }
    return 0;
}

void qmatvec_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    memset(g_fn, 0, sizeof(g_fn));
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
