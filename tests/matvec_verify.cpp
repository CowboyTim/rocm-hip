// matvec_verify — Phase 20 Step 3b: GPU matvec on raw GGUF blocks vs CPU reference.
//
//   matvec_verify <model.gguf> [tensor ...]
//
// Pins the whole GGUF mapping read-only (zero-copy) and runs matvec_<type> on real 2-D weight
// tensors: by default the first tensor of every type in the file. The reference decodes each row
// with qt_to_float (bit-exact vs ggml, see dequant_verify) and accumulates in double.
// Pass: |gpu - ref| <= 1e-5 * sum_k |w_k x_k| for every row. Run from build/co (loads matvec_q.co).
//
// Types with an int8 kernel are also run through qmatvec_quantize + qmatvec_launch_q8, against a reference
// whose x is quantized on the CPU the way ggml does it (Q8_0 for ROCmFP4, Q8_K for K-quants) and decoded
// back, so it checks the quantizer and the integer dot products together.

#include "gguf_mmap.h"
#include "qmatvec.h"
#include "quant_types.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// /dev/shm-backed, hipHostRegister'ed buffer (project core pattern)
struct ShmBuf {
    void *p = nullptr;
    size_t n = 0;
    bool alloc(const char *name, size_t bytes) {
        std::string path = std::string("/dev/shm/") + name;
        int fd = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0 || ftruncate(fd, (off_t)bytes) != 0) return false;
        p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        unlink(path.c_str());
        if (p == MAP_FAILED) return false;
        n = bytes;
        return hipHostRegister(p, bytes, hipHostRegisterMapped) == hipSuccess;
    }
    ~ShmBuf() {
        if (p && p != MAP_FAILED) {
            (void)hipHostUnregister(p);
            munmap(p, n);
        }
    }
};

// x quantized and decoded back as ggml's CPU reference does for this weight type's vec_dot_type:
// Q8_0 (x86 SIMD quantize_row_q8_0) for ROCmFP4 and Q8_0, Q8_K (quantize_row_q8_K_ref) for K-quants
static void q8_roundtrip(int wtype, const float *x, float *out, int K) {
    if (wtype >= QT_Q2_K && wtype <= QT_Q6_K) {
        for (int b = 0; b < K; b += 256) {
            float amax = 0.0f, mx = 0.0f;
            for (int j = 0; j < 256; ++j)
                if (std::fabs(x[b + j]) > amax) { amax = std::fabs(x[b + j]); mx = x[b + j]; }
            if (amax == 0.0f) { for (int j = 0; j < 256; ++j) out[b + j] = 0.0f; continue; }
            const float iscale = -127.0f / mx, d = 1.0f / iscale;
            for (int j = 0; j < 256; ++j) out[b + j] = d * (float)std::min(127, (int)nearbyintf(iscale * x[b + j]));
        }
    } else {
        for (int b = 0; b < K; b += 32) {
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(x[b + j]));
            const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
            const float dh = __half2float(__float2half(amax / 127.0f));
            for (int j = 0; j < 32; ++j) out[b + j] = dh * nearbyintf(x[b + j] * id);
        }
    }
}

static ShmBuf g_q8mem;
static qmatvec_q8 g_q8;

static bool run_one(gguf_model_t *g, int64_t idx, ShmBuf &xb, ShmBuf &yb) {
    const char *name = gguf_tensor_name(g, idx);
    const int type = gguf_tensor_type(g, idx);
    const int64_t *ne = gguf_tensor_ne(g, idx);
    const qt_traits *tr = qt_get_traits(type);
    const int K = (int)ne[0];
    const int N = (int)ne[1];
    if (!tr || !qmatvec_supported(type)) {
        printf("  [FAIL] %-32s type %d has no matvec kernel\n", name, type);
        return false;
    }
    const size_t rb = qt_row_size(type, K);
    size_t bytes = 0;
    const uint8_t *W = (const uint8_t *)gguf_get_tensor_ptr(g, name, &bytes);
    if ((size_t)K * sizeof(float) > xb.n || (size_t)N * sizeof(float) > yb.n) {
        printf("  [SKIP] %-32s shape %dx%d exceeds test buffers\n", name, N, K);
        return true;
    }

    float *x = (float *)xb.p, *y = (float *)yb.p;
    uint32_t s = 12345u;
    for (int k = 0; k < K; ++k) {
        s = s * 1664525u + 1013904223u;
        x[k] = ((int)(s >> 9) - (1 << 22)) / (float)(1 << 22);  // [-1, 1)
    }
    memset(y, 0, (size_t)N * sizeof(float));

    hipError_t e = qmatvec_launch(type, W, rb, x, y, N, K, 0);
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("  [FAIL] %-32s launch: %s\n", name, hipGetErrorString(e));
        return false;
    }

    std::vector<float> row((size_t)K);
    // The BF16 / F16 kernels round x to BF16 / F16 first, as ggml (the type is its own vec_dot_type)
    std::vector<float> xr(x, x + K);
    if (type == QT_F16)
        for (int k = 0; k < K; ++k) xr[k] = __half2float(__float2half(xr[k]));
    if (type == QT_BF16)
        for (int k = 0; k < K; ++k) {
            uint32_t u;
            memcpy(&u, &xr[k], 4);
            u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
            memcpy(&xr[k], &u, 4);
        }
    double worst = 0.0;
    int bad = 0, worst_n = 0;
    for (int n = 0; n < N; ++n) {
        qt_to_float(type, W + (size_t)n * rb, row.data(), K);
        double ref = 0.0, mag = 0.0;
        for (int k = 0; k < K; ++k) {
            ref += (double)row[k] * xr[k];
            mag += std::fabs((double)row[k] * xr[k]);
        }
        const double rel = std::fabs(y[n] - ref) / (mag + 1e-30);
        if (rel > worst) {
            worst = rel;
            worst_n = n;
        }
        if (rel > 1e-5) ++bad;
    }
    printf("  [%s] %-32s %-18s %6d x %-6d worst rel err %.2e (row %d)%s\n", bad ? "FAIL" : "PASS", name, tr->name,
           N, K, worst, worst_n, bad ? " <-- rows over 1e-5" : "");
    if (!qmatvec_q8_supported(type, K) || K > g_q8.max_k) return bad == 0;

    // int8 path
    memset(y, 0, (size_t)N * sizeof(float));
    e = qmatvec_quantize(x, K, &g_q8, 0);
    if (e == hipSuccess) e = qmatvec_launch_q8(type, W, rb, &g_q8, y, N, K, 0);
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        printf("  [FAIL] %-32s int8 launch: %s\n", name, hipGetErrorString(e));
        return false;
    }
    std::vector<float> xq((size_t)K);
    q8_roundtrip(type, x, xq.data(), K);
    double worst8 = 0.0;
    int bad8 = 0, worst8_n = 0;
    // Q4_1 / Q5_1: the offsets use ggml's F16-rounded Q8_1 block sums (qmatvec_q8.s0), which this exact reference
    // does not round
    const double tol8 = type == QT_Q4_1 || type == QT_Q5_1 ? 5e-4 : 1e-5;
    for (int n = 0; n < N; ++n) {
        qt_to_float(type, W + (size_t)n * rb, row.data(), K);
        double ref = 0.0, mag = 0.0;
        for (int k = 0; k < K; ++k) {
            ref += (double)row[k] * xq[k];
            mag += std::fabs((double)row[k] * xq[k]);
        }
        const double rel = std::fabs(y[n] - ref) / (mag + 1e-30);
        if (rel > worst8) {
            worst8 = rel;
            worst8_n = n;
        }
        if (rel > tol8) ++bad8;
    }
    printf("  [%s] %-32s %-18s int8 path (q8 x)   worst rel err %.2e (row %d)%s\n", bad8 ? "FAIL" : "PASS", name,
           tr->name, worst8, worst8_n, bad8 ? " <-- rows over 1e-5" : "");

    // Batched (prefill) kernels on T = 11 tokens (a full tile of 16 is not reached; dot4 path, must be
    // bit-identical to the single-token matvecs) and T = 70 (a full 64-token WMMA tile plus a partial one where
    // the type has a WMMA kernel: same integer sums, different float accumulation order, so 1e-5 relative)
    bool okb_all = true;
    for (const int T : {11, 70}) {
        if ((size_t)T * K * sizeof(float) > xb.n || (size_t)T * N * sizeof(float) > yb.n || T * K > g_q8.max_k ||
            (T > 11 && N > 65536)) {
            printf("  [SKIP] %-32s batched check, %d tokens, exceeds test buffers\n", name, T);
            continue;
        }
        std::vector<float> xs((size_t)T * K), single((size_t)T * N);
        uint32_t s2 = 777u;
        for (size_t i = 0; i < xs.size(); ++i) {
            s2 = s2 * 1664525u + 1013904223u;
            xs[i] = ((int)(s2 >> 9) - (1 << 22)) / (float)(1 << 22);
        }
        for (int t = 0; t < T; ++t) {
            memcpy(x, &xs[(size_t)t * K], (size_t)K * sizeof(float));
            e = qmatvec_quantize(x, K, &g_q8, 0);
            if (e == hipSuccess) e = qmatvec_launch_q8(type, W, rb, &g_q8, y, N, K, 0);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) { printf("  [FAIL] %-32s int8 launch: %s\n", name, hipGetErrorString(e)); return false; }
            memcpy(&single[(size_t)t * N], y, (size_t)N * sizeof(float));
        }
        memcpy(x, xs.data(), xs.size() * sizeof(float));
        e = qmatvec_quantize(x, T * K, &g_q8, 0);
        if (e == hipSuccess) e = qmatvec_launch_q8_batch(type, W, rb, &g_q8, y, N, N, K, T, 0);
        if (e == hipSuccess) e = hipDeviceSynchronize();
        if (e != hipSuccess) { printf("  [FAIL] %-32s batched launch: %s\n", name, hipGetErrorString(e)); return false; }
        // Row scale: sum_k |w_k| over the row (|x| <= 1), so the bound is relative to the size of the terms
        std::vector<double> rmag((size_t)N);
        for (int n = 0; n < N; ++n) {
            qt_to_float(type, W + (size_t)n * rb, row.data(), K);
            double m = 0.0;
            for (int k = 0; k < K; ++k) m += std::fabs((double)row[k]);
            rmag[n] = m;
        }
        double worstb = 0.0;
        for (int t = 0; t < T; ++t)
            for (int n = 0; n < N; ++n) {
                const size_t i = (size_t)t * N + n;
                worstb = std::max(worstb, std::fabs((double)y[i] - single[i]) / (rmag[n] + 1e-30));
            }
        const bool exact = T == 11;
        const bool okb = exact ? worstb == 0.0 : worstb <= 1e-5;
        okb_all = okb_all && okb;
        printf("  [%s] %-32s %-18s batched, %d tokens  worst diff vs single %.2e of sum|w|%s\n", okb ? "PASS" : "FAIL",
               name, tr->name, T, worstb, exact ? " (must be 0)" : "");
    }
    return bad == 0 && bad8 == 0 && okb_all;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [tensor ...]\n", argv[0]);
        return 2;
    }
    if (qmatvec_init("matvec_q.co") != 0) return 2;
    gguf_model_t *g = gguf_mmap_open(argv[1]);
    if (!g) return 2;
    size_t map_size = 0;
    void *base = (void *)gguf_map_base(g, &map_size);
    hipError_t e = hipHostRegister(base, map_size, hipHostRegisterReadOnly);
    if (e != hipSuccess) {
        fprintf(stderr, "hipHostRegister(GGUF map, ReadOnly) failed: %s\n", hipGetErrorString(e));
        return 2;
    }

    ShmBuf xb, yb;
    if (!xb.alloc("matvec_verify_x.bin", 64u << 20) || !yb.alloc("matvec_verify_y.bin", 64u << 20)) {
        fprintf(stderr, "cannot allocate /dev/shm buffers\n");
        return 2;
    }

    std::vector<int64_t> todo;
    if (argc > 2) {
        for (int a = 2; a < argc; ++a)
            for (int64_t i = 0; i < gguf_n_tensors(g); ++i)
                if (strcmp(gguf_tensor_name(g, i), argv[a]) == 0) todo.push_back(i);
    } else {
        bool seen[256] = {false};
        for (int64_t i = 0; i < gguf_n_tensors(g); ++i) {
            const int t = gguf_tensor_type(g, i);
            const int64_t *ne = gguf_tensor_ne(g, i);
            if (t < 0 || t >= 256 || seen[t] || ne[1] <= 1) continue;  // 2-D weights only
            seen[t] = true;
            todo.push_back(i);
        }
    }

    const int max_k = (1 << 16) * 80;  // the 70-token batched check of K up to 65536
    if (!g_q8mem.alloc("matvec_verify_q8.bin", qmatvec_q8_bytes(max_k))) {
        fprintf(stderr, "cannot allocate /dev/shm buffers\n");
        return 2;
    }
    qmatvec_q8_bind(&g_q8, g_q8mem.p, max_k);

    int fail = 0;
    for (int64_t i : todo) fail += !run_one(g, i, xb, yb);
    (void)hipHostUnregister(base);
    gguf_mmap_close(g);
    qmatvec_free();
    printf("matvec_verify: %zu tensors, %d failed\n", todo.size(), fail);
    return fail ? 1 : 0;
}
