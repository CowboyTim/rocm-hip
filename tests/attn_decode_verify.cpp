// attn_decode_verify: decode attention (attn_partial and the per-KV-head attn_partial_gqa) over an F16, Q8_0 or Q4_0 KV
// cache vs a CPU reference computed from the cache's own values and the prepped q, for the head sizes, query / KV
// head ratios and sliding windows the models use. No model needed; run from build/co (attention.co).
//
// Each case fills a cache by 512-token batches, then decodes 1 and 3 tokens and checks every output; both kernel
// paths (ATTN_NO_GQA=1 for the per-head one). Then times the 1-token step over 24 cache copies (past the 32 MB MALL,
// as a model's layers are).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "attention.h"

#define CK(x)                                                                             \
    do {                                                                                  \
        hipError_t e_ = (x);                                                              \
        if (e_ != hipSuccess) {                                                           \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_));     \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

template <class T>
static T* alloc(size_t n) {
    void* p;
    CK(hipHostMalloc(&p, n * sizeof(T) + 256, hipHostMallocNonCoherent));
    memset(p, 0, n * sizeof(T));
    return (T*)p;
}

static bool g_timing = true;

static float h2f(uint16_t h) { return __half2float(*(__half*)&h); }

struct Cfg {
    const char* name;
    int D, n_q, n_kv, npos, window;
};

// Largest error of the decode tokens' outputs relative to the largest reference value; *us: one decode step
static double run(const Cfg& c, int kq, double* us) {
    const int D = c.D, n_q = c.n_q, n_kv = c.n_kv, n_ctx = 4096, rows = 1024, T0 = 512, TD = 3, L = g_timing ? 24 : 1;
    // inputs uniform in [-1.7, 1.7) (about unit variance) from an LCG: std::normal_distribution took most of the run
    uint32_t rng = 1234;
    auto nd = [&]() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f) * 1.7f; };
    const int kv_stride = kq ? n_kv * D * (kq == 1 ? 17 : 9) / 32 : n_kv * D;  // F16, Q8_0, Q4_0 rows
    const size_t layer = (size_t)n_ctx * kv_stride;
    __half* K = alloc<__half>(layer * L);
    __half* V = alloc<__half>(layer * L);
    float* qfull = alloc<float>((size_t)T0 * n_q * D);
    float* k = alloc<float>((size_t)T0 * n_kv * D);
    float* v = alloc<float>((size_t)T0 * n_kv * D);
    float* qn = alloc<float>(D);
    float* kn = alloc<float>(D);
    for (int i = 0; i < D; ++i) qn[i] = 1.0f, kn[i] = 1.0f;
    float* q = alloc<float>((size_t)T0 * n_q * D);
    float* out = alloc<float>((size_t)T0 * n_q * D);
    float* pml = alloc<float>(attention_part_ml_floats(n_q, n_ctx, rows));
    float* pacc = alloc<float>(attention_part_acc_floats_d(n_q, n_ctx, rows, D));
    attention_opts o = {};
    o.head_dim = D;
    o.kv_quant = kq;
    o.window = c.window;
    o.part_slots = (long long)attention_part_slots(n_ctx, rows);  // as the engine's scratch: 64-position chunks
    o.kv_stride = kv_stride;
    auto step = [&](int p0, int T, __half* Kl, __half* Vl) {
        CK(attention_forward_opts(qfull, k, v, qn, kn, Kl, Vl, 30, 0, n_q, n_kv, D, p0, T, 1e6f, 1e-6f, q, pml, pacc,
                                  out, &o, nullptr, 0));
    };
    auto fill = [&](int T) {
        for (size_t i = 0; i < (size_t)T * n_q * D; ++i) qfull[i] = nd();
        for (size_t i = 0; i < (size_t)T * n_kv * D; ++i) k[i] = nd(), v[i] = nd();
    };
    for (int p0 = 0; p0 < c.npos; p0 += T0) {
        const int T = std::min(T0, c.npos - p0);
        fill(T);
        step(p0, T, K, V);
    }
    auto val = [&](const __half* C, int p, int e) -> float {
        const char* row = (const char*)(C + (size_t)p * kv_stride);
        if (!kq) return h2f(((const uint16_t*)row)[e]);
        const int drow = kv_stride * 32 / (kq == 1 ? 17 : 9);
        if (kq == 1) return (float)((const int8_t*)row)[e] * h2f(((const uint16_t*)(row + drow))[e / 32]);
        const uint8_t byte = ((const uint8_t*)row)[(e >> 3) * 4 + (e & 3)];  // (attention.hip's Q4_0 layout)
        return (float)(((e & 4) ? byte >> 4 : byte & 15) - 8) * h2f(((const uint16_t*)(row + drow / 2))[e / 32]);
    };
    double maxerr = 0.0, maxref = 0.0;
    const int R = n_q / n_kv;
    std::vector<double> s(c.npos + TD), acc(D);
    for (int T : {1, TD}) {  // decode 1 token at npos, then TD tokens from npos (rewriting its row)
        fill(T);
        step(c.npos, T, K, V);
        CK(hipDeviceSynchronize());
        for (int z = 0; z < T; ++z) {
            const int n = c.npos + z + 1, lo = c.window > 0 ? std::max(0, n - c.window) : 0;
            const float* qz = q + (size_t)z * n_q * D;
            const float* oz = out + (size_t)z * n_q * D;
            for (int h = 0; h < n_q; ++h) {
                const int g = h / R;
                double m = -1e30, l = 0.0;
                for (int p = lo; p < n; ++p) {
                    double d = 0.0;
                    for (int i = 0; i < D; ++i) d += (double)qz[h * D + i] * val(K, p, g * D + i);
                    s[p] = d / sqrt((double)D);
                    m = std::max(m, s[p]);
                }
                for (int i = 0; i < D; ++i) acc[i] = 0.0;
                for (int p = lo; p < n; ++p) {
                    const double e = exp(s[p] - m);
                    l += e;
                    for (int i = 0; i < D; ++i) acc[i] += e * val(V, p, g * D + i);
                }
                // a quantized cache holds V Hadamard-rotated in blocks of 64 (attention.hip hadamard_rot): the
                // output is rotated back (the transform is its own inverse)
                if (kq)
                    for (int b = 0; b < D; b += 64)
                        for (int st = 1; st < 64; st <<= 1)
                            for (int i = b; i < b + 64; ++i)
                                if (!(i & st)) {
                                    const double x = acc[i], y = acc[i | st];
                                    acc[i] = (x + y) / sqrt(2.0), acc[i | st] = (x - y) / sqrt(2.0);
                                }
                for (int i = 0; i < D; ++i) {
                    maxerr = std::max(maxerr, fabs(acc[i] / l - oz[h * D + i]));
                    maxref = std::max(maxref, fabs(acc[i] / l));
                }
            }
        }
    }
    // Timing: the 1-token step over L copies of the cache (each step rewrites row npos of its copy)
    *us = 0.0;
    if (!g_timing) {
        for (void* p : {(void*)K, (void*)V, (void*)qfull, (void*)k, (void*)v, (void*)qn, (void*)kn, (void*)q,
                        (void*)out, (void*)pml, (void*)pacc})
            CK(hipHostFree(p));
        return maxerr / std::max(maxref, 1e-9);
    }
    for (int l = 1; l < L; ++l) memcpy(K + l * layer, K, layer * 2), memcpy(V + l * layer, V, layer * 2);
    const int reps = 20 * L;
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    for (int pass = 0; pass < 2; ++pass) {  // (the first warms up)
        CK(hipEventRecord(e0, 0));
        for (int r = 0; r < reps; ++r) step(c.npos, 1, K + (r % L) * layer, V + (r % L) * layer);
        CK(hipEventRecord(e1, 0));
        CK(hipEventSynchronize(e1));
    }
    float ms = 0.0f;
    CK(hipEventElapsedTime(&ms, e0, e1));
    *us = 1000.0 * ms / reps;
    CK(hipEventDestroy(e0));
    CK(hipEventDestroy(e1));
    for (void* p : {(void*)K, (void*)V, (void*)qfull, (void*)k, (void*)v, (void*)qn, (void*)kn, (void*)q, (void*)out,
                    (void*)pml, (void*)pacc})
        CK(hipHostFree(p));
    return maxerr / std::max(maxref, 1e-9);
}

int main(int argc, char** argv) {
    // [attention.co] [--no-timing]: the checks only (tests/regress.sh), not the timing loops
    const char* co = "attention.co";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-timing")) g_timing = false;
        else co = argv[i];
    }
    const Cfg cfgs[] = {
        {"Qwen2.5-0.5B", 64, 14, 2, 2400, 0},  // ratio 7
        {"Qwen2.5-7B", 128, 28, 4, 2400, 0},
        {"Qwen3-30B-A3B", 128, 32, 4, 2400, 0},  // ratio 8
        {"Qwen3-0.6B", 128, 16, 8, 2400, 0},     // ratio 2
        {"Qwen3.6-35B", 256, 16, 2, 2400, 0},
        {"ratio 16", 128, 64, 4, 2400, 0},
        {"D 256 ratio 2", 256, 16, 8, 3000, 0},
        {"window 1024", 256, 16, 8, 3000, 1024},
        {"window 1024", 128, 32, 4, 3000, 1024},
        {"short", 128, 32, 4, 100, 0},
        {"short", 128, 32, 4, 300, 0},
    };
    int bad = 0;
    printf("%-14s %3s %3s %3s %5s %5s %-5s %9s %9s %9s %9s\n", "shape", "D", "n_q", "kv", "pos", "win", "cache",
           "err kv-hd", "err head", "us kv-hd", "us head");
    for (const Cfg& c : cfgs)
        for (int kq = 0; kq < 3; ++kq) {
            if (kq && c.n_kv * c.D % 256 != 0) continue;  // (quantized cache rows are 256-value multiples)
            double us[2], err[2];
            for (int off = 0; off < 2; ++off) {  // per KV head (the default), per query head
                if (off) setenv("ATTN_NO_GQA", "1", 1);
                else unsetenv("ATTN_NO_GQA");
                attention_free();
                if (attention_init(co) != 0) {
                    fprintf(stderr, "cannot load %s\n", co);
                    return 2;
                }
                err[off] = run(c, kq, &us[off]);
            }
            const bool ok = err[0] < 1e-5 && err[1] < 1e-5;
            bad += !ok;
            printf("%-14s %3d %3d %3d %5d %5d %-5s %9.2e %9.2e %9.1f %9.1f %s\n", c.name, c.D, c.n_q, c.n_kv, c.npos,
                   c.window, kq == 2 ? "Q4_0" : kq ? "Q8_0" : "F16", err[0], err[1], us[0], us[1], ok ? "ok" : "FAIL");
        }
    unsetenv("ATTN_NO_GQA");
    attention_free();
    printf(bad ? "%d FAILED\n" : "all passed\n", bad);
    return bad != 0;
}
